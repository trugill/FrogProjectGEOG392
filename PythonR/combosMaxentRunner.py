#!/usr/bin/env python3
"""
maxent_combo_caller.py
======================
Command-line MaxEnt driver: runs one MaxEnt model per combination FOLDER found in
a combos directory. The variables for each model are read from the folder name
by splitting on "_" (e.g. "BIO11_BIO16_BIO18" -> BIO11, BIO16, BIO18).

Python analogue of MaxentCaller.R (runMaxent / buildLayerArgs), adapted to drive
a pre-existing combos tree such as:

    D:/Frogs/Models/combos/BIO11_BIO16
    D:/Frogs/Models/combos/BIO11_BIO16_BIO18
    ...

with all layers living in a single flat folder:

    D:/Frogs/ASC/ASC_selected/BIO11.ASC, BIO16.ASC, BIO18.ASC, BIO3.ASC, ...

Zero-argument usage (edit the DEFAULT_* constants below first):
    python maxent_combo_caller.py

Typical explicit usage:
    python maxent_combo_caller.py ^
        --jar D:/Frogs/maxent/maxent.jar ^
        --samples D:/Frogs/CSV/points.refined_trimdupes_grid.csv ^
        --layers D:/Frogs/ASC/ASC_selected ^
        --combos D:/Frogs/Models/combos ^
        --output-format logistic --betamultiplier 1.0 ^
        --jackknife --random-test-percentage 25

IMPORTANT -- WHY EDITING A DATACLASS DEFAULT MAY DO NOTHING:
    main() rebuilds MaxentSettings entirely from parsed CLI args, so the
    dataclass field defaults are decorative at runtime. To change behaviour
    permanently, edit the DEFAULT_* constants below (they feed argparse), NOT
    the dataclass fields.

TROUBLESHOOTING "Error occurred during initialization of VM":
    That message comes from Java itself, BEFORE maxent.jar loads. It almost
    always means "Could not reserve enough space for object heap" -- i.e. the
    -Xmx value is larger than the JVM can reserve. A 32-bit JVM typically caps
    out near 1200-1500 MB no matter how much RAM the machine has. This script
    probes the heap up front and steps it down automatically, but the real fix
    is to install 64-bit Java (Adoptium Temurin 17/21) and re-run.

WHERE JACKKNIFE RESULTS APPEAR:
    There is no separate "jackknife file". Look for:
      * maxentResults.csv -> columns "Training gain without BIO11",
        "Training gain with only BIO11" (and Test gain / AUC variants when
        randomtestpoints > 0). These columns are the definitive proof.
      * plots/<species>_training.png -> the red/blue bar charts
        (requires plots=true AND pictures=true).
      * Runtime roughly triples: jackknife trains 2n+1 models per combo.
"""

from __future__ import annotations          # modern type hints on older Pythons

import argparse                             # CLI parsing
import csv                                  # run_summary.csv / combinations.csv
import itertools                            # pair/subset enumeration
import re                                   # log grepping, name normalization
import shutil                               # locate `java` on PATH
import subprocess                           # launch java -jar maxent.jar
import sys                                  # exit codes, stderr
import time                                 # per-run wall clock
from concurrent.futures import ThreadPoolExecutor, as_completed  # optional parallelism
from dataclasses import dataclass, field    # tidy settings container
from pathlib import Path                    # all path handling
from typing import Iterable, Sequence

# =============================================================================
# SITE DEFAULTS -- edit these so the script can be run with no arguments at all
# =============================================================================
DEFAULT_JAR     = Path("D:/Frogs/maxent/maxent.jar")                        # maxent.jar
DEFAULT_LAYERS  = Path("D:/Frogs/ASC/ASC_selected")                         # flat .asc folder
DEFAULT_COMBOS  = Path("D:/Frogs/Models/combos")                            # one subfolder per model
DEFAULT_SAMPLES = Path("D:/Frogs/CSV/points.refined_trimdupes_grid.csv")    # occurrence CSV

# --- Model-parameter defaults ------------------------------------------------
# These DO take effect, because they are wired into the argparse defaults
# (unlike the MaxentSettings dataclass defaults, which main() overwrites).
DEFAULT_JACKKNIFE = True    # <<< JACKKNIFE: measure variable importance
                            #     (trains 2n+1 models per combo -- ~3x slower)
DEFAULT_TEST_PCT  = 25      # <<< JACKKNIFE: % of points withheld as test data.
                            #     >0 unlocks the Test-gain/AUC jackknife bars;
                            #     0 gives the training-gain chart only.
DEFAULT_PLOTS     = True    # <<< JACKKNIFE: plots= must be true for the PNGs

# Accepted raster extensions, compared case-insensitively because this dataset
# uses uppercase ".ASC" on disk while MaxEnt conventionally writes ".asc".
ASC_SUFFIXES = {".asc"}

# Rasters that MaxEnt writes even for partially failed runs -- these must NOT
# count as evidence that a model succeeded.
NON_PREDICTION_ASC = re.compile(r"_clamping|_novel", re.IGNORECASE)

# Log lines indicating MaxEnt complained, mirroring the R grep() call.
ERROR_PATTERNS = re.compile(r"error|exception|failed|cannot|missing", re.IGNORECASE)

# <<< JACKKNIFE: maxentResults.csv header fragments that only exist when the
# jackknife actually ran ("Training gain without BIO11", "... with only BIO11").
JACKKNIFE_RESULT_COL = re.compile(r"without |with only ", re.IGNORECASE)

# Fingerprints of a JVM that died before maxent.jar was ever loaded. These are
# fatal for the WHOLE batch (not just one combo), so they get special handling.
JVM_STARTUP_PATTERNS = re.compile(
    r"initialization of VM"
    r"|Could not reserve enough space"
    r"|Could not create the Java Virtual Machine"
    r"|Unrecognized option"
    r"|Invalid maximum heap size"
    r"|Unable to allocate",
    re.IGNORECASE)

# Heap sizes (MB) tried in order when the requested size cannot be reserved.
# `None` means "emit no -Xmx flag at all and let the JVM choose its default".
HEAP_FALLBACK_LADDER: list[int | None] = [1536, 1024, 768, 512, None]


class JvmStartupError(RuntimeError):
    """Raised when Java fails to boot -- aborts the batch instead of looping."""


# =============================================================================
# JAVA PREFLIGHT
# =============================================================================
def java_version_banner(java_exe: str) -> str:
    """
    Return the `java -version` banner (Java prints it to stderr, not stdout).

    Used to surface the JVM vendor/version AND, critically, whether this is a
    32-bit VM -- the usual reason a 2 GB heap request fails outright.
    """
    try:
        proc = subprocess.run([java_exe, "-version"],
                              capture_output=True, text=True, check=False)
    except OSError as exc:
        return f"<could not execute {java_exe}: {exc}>"
    return ((proc.stderr or "") + (proc.stdout or "")).strip()


def jvm_is_32bit(banner: str) -> bool:
    """
    Heuristically detect a 32-bit JVM from its version banner.

    64-bit VMs announce "64-Bit Server VM"; 32-bit ones say "Client VM" or
    "32-Bit". A 32-bit VM cannot usually reserve more than ~1.2-1.5 GB of heap.
    """
    if re.search(r"64-bit", banner, re.IGNORECASE):
        return False
    return bool(re.search(r"32-bit|Client VM", banner, re.IGNORECASE))


def heap_is_reservable(java_exe: str, mb: int | None) -> bool:
    """
    Probe whether the JVM can actually start with this heap size.

    Starting `java -Xmx<N>m -version` is instantaneous and allocates the heap
    reservation, so it is a faithful (and cheap) test of whether a real MaxEnt
    run with the same flag would survive startup.
    """
    cmd = [java_exe]
    if mb:                                           # mb=None/0 -> no -Xmx flag
        cmd.append(f"-Xmx{mb}m")
    cmd.append("-version")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError:
        return False
    combined = (proc.stderr or "") + (proc.stdout or "")
    # Success = clean exit AND no JVM startup complaint in the output.
    return proc.returncode == 0 and not JVM_STARTUP_PATTERNS.search(combined)


def negotiate_heap(java_exe: str, requested_mb: int | None,
                   allow_fallback: bool = True) -> int | None:
    """
    Settle on the largest heap this JVM will actually accept.

    Tries the requested size first, then walks HEAP_FALLBACK_LADDER downward.
    Raises JvmStartupError if even a flag-less JVM refuses to boot, because at
    that point nothing MaxEnt-related can possibly work.
    """
    banner = java_version_banner(java_exe)
    print(f"[java] {java_exe}", flush=True)
    for line in banner.splitlines()[:3]:             # 3 lines = version/runtime/VM
        print(f"[java] {line.strip()}", flush=True)
    if jvm_is_32bit(banner):
        print("[java] !! This looks like a 32-BIT JVM: heap is capped near "
              "~1200-1500 MB. Install 64-bit Java (Temurin 17/21) for large "
              "raster stacks.", flush=True)

    # Build the candidate list: requested first, then progressively smaller.
    candidates: list[int | None] = [requested_mb]
    if allow_fallback:
        candidates += [mb for mb in HEAP_FALLBACK_LADDER
                       if mb is None or requested_mb is None or mb < requested_mb]

    tried: list[str] = []
    for mb in candidates:
        label = f"-Xmx{mb}m" if mb else "(JVM default heap)"
        if heap_is_reservable(java_exe, mb):
            if mb != requested_mb:
                print(f"[java] requested heap unavailable; using {label}", flush=True)
            else:
                print(f"[java] heap OK: {label}", flush=True)
            return mb
        tried.append(label)
        print(f"[java] cannot reserve {label}", flush=True)

    raise JvmStartupError(
        "Java will not start with any heap size (tried: "
        + ", ".join(tried) + "). Run `java -version` manually to diagnose; "
        "a reinstall of 64-bit Java is usually the fix.\n--- java -version said ---\n"
        + banner)


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class MaxentSettings:
    """
    All MaxEnt/Java knobs that stay constant across every combination run.

    WARNING: these field defaults are NOT the effective defaults at runtime --
    main() constructs this object from parsed CLI args, so every field is
    explicitly supplied. Change the DEFAULT_* module constants instead.
    """

    jar: Path                                  # path to maxent.jar
    java_memory_mb: int | None = 2048          # -XmxNNNNm; None = JVM default
    java_exe: str = "java"                     # override if java isn't on PATH

    # --- Window / run behaviour ---------------------------------------------
    visible: bool = True                       # visible=true -> MaxEnt GUI is SHOWN
    autorun: bool = True                       # start without clicking "Run"
    warnings: bool = False                     # suppress blocking warning dialogs
    redo_if_exists: bool = True                # overwrite stale outputs

    # --- Output / reporting options -----------------------------------------
    output_format: str = "logistic"            # logistic | cloglog | cumulative | raw
    response_curves: bool = True               # responsecurves=
    jackknife: bool = DEFAULT_JACKKNIFE        # <<< JACKKNIFE: jackknife=
    pictures: bool = True                      # pictures= (write PNG maps)
    plots: bool = DEFAULT_PLOTS                # <<< JACKKNIFE: plots= must be
                                               #     true for the bar-chart PNGs
    write_plot_data: bool = False              # writeplotdata=

    # --- Regularization / features ------------------------------------------
    betamultiplier: float = 1.0                # global regularization multiplier
    linear: bool = True                        # feature classes
    quadratic: bool = True
    product: bool = True
    threshold: bool = True
    hinge: bool = True

    # --- Replication / test split -------------------------------------------
    replicates: int = 1                        # 1 = single model, no replication
    replicate_type: str = "crossvalidate"      # crossvalidate | bootstrap | subsample
    random_test_percentage: int = DEFAULT_TEST_PCT  # <<< JACKKNIFE:
                                               #     randomtestpoints=; 0 = none.
                                               #     >0 unlocks Test gain / AUC bars.

    # --- Misc ----------------------------------------------------------------
    random_seed: bool = False                  # randomseed=
    maximum_iterations: int | None = None      # maximumiterations=
    nodata: int = -9999                        # must match the NAflag of your .asc
    projection_layers: Path | None = None      # projectionlayers= (mask / future clim)
    threads: int = 1                           # MaxEnt-internal threads per model
    extra_args: list[str] = field(default_factory=list)  # raw passthrough args

    def validate(self) -> None:
        """Fail fast and loudly, like the R `stop("MaxEnt jar not found")` guard."""
        if not self.jar.is_file():
            raise FileNotFoundError(f"MaxEnt jar not found: {self.jar}")
        if shutil.which(self.java_exe) is None:
            raise FileNotFoundError(f"Java executable not found on PATH: {self.java_exe}")

    def java_args(self) -> list[str]:
        """
        JVM-level arguments that must precede `-jar`.

        NOTE: this uses the STANDARD `-Xmx` flag. The R original used the legacy
        `-mx` alias; on some JVMs that is rejected outright, producing exactly
        the "Error occurred during initialization of VM" message.
        """
        args: list[str] = []
        if self.java_memory_mb:                     # None/0 -> let the JVM decide
            args.append(f"-Xmx{self.java_memory_mb}m")
        return args

    def base_args(self) -> list[str]:
        """Run-mode tokens shared by every model (visibility, autorun, overwrite)."""
        def tf(flag: bool) -> str:                 # MaxEnt wants lowercase true/false
            return "true" if flag else "false"
        return [
            f"redoifexists={tf(self.redo_if_exists)}",
            f"autorun={tf(self.autorun)}",
            f"visible={tf(self.visible)}",         # TRUE here: show the MaxEnt screen
            f"warnings={tf(self.warnings)}",
        ]

    def model_args(self) -> list[str]:
        """Translate the dataclass fields into MaxEnt `key=value` tokens."""
        def tf(flag: bool) -> str:
            return "true" if flag else "false"

        args = [
            f"outputformat={self.output_format}",   # logistic response surface, etc.
            f"responsecurves={tf(self.response_curves)}",
            f"jackknife={tf(self.jackknife)}",      # <<< JACKKNIFE: the actual switch
            f"pictures={tf(self.pictures)}",
            f"plots={tf(self.plots)}",              # <<< JACKKNIFE: bar charts live here
            f"writeplotdata={tf(self.write_plot_data)}",
            f"betamultiplier={self.betamultiplier}",
            f"linear={tf(self.linear)}",
            f"quadratic={tf(self.quadratic)}",
            f"product={tf(self.product)}",
            f"threshold={tf(self.threshold)}",
            f"hinge={tf(self.hinge)}",
            f"randomseed={tf(self.random_seed)}",
            f"nodata={self.nodata}",
            f"threads={self.threads}",
        ]
        if self.replicates and self.replicates > 1:
            # Only emit replication args when actually replicating; MaxEnt writes a
            # different output layout (averaged grids) once replicates > 1.
            args.append(f"replicates={self.replicates}")
            args.append(f"replicatetype={self.replicate_type}")
        if self.maximum_iterations is not None:
            args.append(f"maximumiterations={self.maximum_iterations}")
        if self.projection_layers is not None:
            # Project the fitted model onto a second (e.g. larger/future) layer set.
            args.append(f"projectionlayers={self.projection_layers}")
        # <<< JACKKNIFE: withhold points so the jackknife can report Test gain /
        # AUC, not just Training gain. MaxEnt ignores this when replicates > 1
        # (the replicate folds supply the test data instead).
        if self.random_test_percentage and self.random_test_percentage > 0:
            args.append(f"randomtestpoints={self.random_test_percentage}")
        args.extend(self.extra_args)                # anything passed verbatim
        return args


# =============================================================================
# LAYER DISCOVERY AND TOGGLING
# =============================================================================
def discover_layers(layers_dir: Path) -> list[str]:
    """
    Return the sorted on-disk variable names (file stems) for every raster in the
    layer folder. Matching is case-insensitive so "BIO11.ASC" is found just as
    readily as "bio11.asc".
    """
    if not layers_dir.is_dir():
        raise NotADirectoryError(f"Layers directory does not exist: {layers_dir}")
    names = sorted(p.stem for p in layers_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in ASC_SUFFIXES)
    if not names:
        raise FileNotFoundError(f"No .asc/.ASC files found in: {layers_dir}")
    return names


def build_layer_lookup(all_layers: Sequence[str]) -> dict[str, str]:
    """
    Case-insensitive map of normalized token -> real on-disk stem.

    Folder names and filenames can disagree on case (folder "BIO11" vs file
    "bio11.asc"), so every toggle argument is emitted using the REAL stem.
    """
    return {name.strip().upper(): name for name in all_layers}


def resolve_variables(tokens: Sequence[str],
                      lookup: dict[str, str]) -> tuple[list[str], list[str]]:
    """
    Map folder-name tokens onto real layer stems.

    Returns (resolved, unresolved) so the caller can skip and report any combo
    folder that references a layer which isn't present in the layer directory.
    """
    resolved: list[str] = []
    unresolved: list[str] = []
    for tok in tokens:
        hit = lookup.get(tok.strip().upper())
        if hit is None:
            unresolved.append(tok)
        elif hit not in resolved:                 # de-dupe, keep folder-name order
            resolved.append(hit)
    return resolved, unresolved


def build_layer_args(selected: Sequence[str],
                     all_layers: Sequence[str],
                     toggle_flag: str = "togglelayerselected") -> list[str]:
    """
    Switch OFF every layer that is not part of this combination.

    MaxEnt is always handed the whole `environmentallayers` directory, so the
    only way to vary the variable set is to deselect the unwanted layers.
    `togglelayerselected=<name>` removes a layer from the model;
    `togglelayertype=<name>` (used in MaxentCaller.R) merely flips
    continuous<->categorical, which does NOT reduce the variable set.
    """
    chosen = set(selected)                                   # O(1) membership tests
    return [f"{toggle_flag}={name}" for name in all_layers if name not in chosen]


# =============================================================================
# COMBINATION DISCOVERY (folder names -> variable sets)
# =============================================================================
def parse_folder_variables(folder: Path) -> list[str]:
    """
    Derive the variable list for a combo folder.

    Preference order:
      1. "_variables.txt" manifest, if present (authoritative, underscore-safe).
      2. Split the folder name on "_".

    CAVEAT: splitting on "_" is only reversible when variable names themselves
    contain no underscores -- which is true for this dataset (BIO3, BIO11, ...).
    If you later add layers like "spectre_human_biomes_60m", generate
    "_variables.txt" manifests instead, because the split will shred those names.
    """
    manifest = folder / "_variables.txt"
    if manifest.is_file():
        vars_from_file = [ln.strip() for ln in
                          manifest.read_text(encoding="utf-8-sig").splitlines()
                          if ln.strip()]
        if vars_from_file:
            return vars_from_file
    # Fall back to the folder name; filter out empty tokens from stray "__".
    return [tok for tok in folder.name.split("_") if tok]


def discover_combo_folders(combos_dir: Path,
                           lookup: dict[str, str]) -> list[tuple[Path, list[str]]]:
    """
    Treat EVERY subfolder of `combos_dir` as a model to run, deriving its
    variables from the folder name (or its manifest). Folders referencing an
    unknown layer are reported and skipped instead of being run incorrectly.
    """
    if not combos_dir.is_dir():
        raise NotADirectoryError(f"Combos directory does not exist: {combos_dir}")

    combos: list[tuple[Path, list[str]]] = []
    skipped: list[tuple[str, list[str]]] = []

    for folder in sorted(p for p in combos_dir.iterdir() if p.is_dir()):
        tokens = parse_folder_variables(folder)
        if not tokens:
            skipped.append((folder.name, ["<no parsable variables>"]))
            continue
        resolved, unresolved = resolve_variables(tokens, lookup)
        if unresolved:
            skipped.append((folder.name, unresolved))
            continue
        combos.append((folder, resolved))

    for name, missing in skipped:
        print(f"[warn] skipping '{name}': no layer for {', '.join(missing)}",
              file=sys.stderr)
    if not combos:
        raise FileNotFoundError(
            f"No runnable combination folders found in {combos_dir}")
    return combos


# =============================================================================
# OPTIONAL: generate combination folders from scratch (R combos() analogue)
# =============================================================================
def load_correlation_matrix(csv_path: Path) -> dict[tuple[str, str], float]:
    """
    Read a Pearson matrix CSV (as written by RFunctions$correlation) into a
    {(var_a, var_b): r} lookup. Column names are not sanitized, so dots and
    underscores in layer names survive (the check.names=FALSE intent).
    """
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:   # strip any BOM
        reader = csv.reader(fh)
        header = next(reader)
        cols = [re.sub(r"^pearson[. ]correlation[. ]coefficient[. ]", "", c).strip()
                for c in header[1:]]
        matrix: dict[tuple[str, str], float] = {}
        for row in reader:
            if not row:
                continue
            row_name = row[0].strip()
            for col_name, value in zip(cols, row[1:]):
                try:
                    matrix[(row_name, col_name)] = float(value)
                except (TypeError, ValueError):
                    continue                     # NA / blank cells are skipped
    return matrix


def generate_combo_folders(combos_dir: Path,
                           layers: Sequence[str],
                           min_size: int,
                           max_size: int,
                           required: Sequence[str] = (),
                           correlation_csv: Path | None = None,
                           correlation_threshold: float = 0.80
                           ) -> list[tuple[Path, list[str]]]:
    """
    Create "VAR1_VAR2_..." folders for every qualifying subset, so the same
    folder-driven runner can execute them. Only used with --mode generate.
    """
    max_size = min(max_size, len(layers))         # can't pick more than exist
    matrix = load_correlation_matrix(correlation_csv) if correlation_csv else None

    made: list[tuple[Path, list[str]]] = []
    for size in range(max(1, min_size), max_size + 1):
        for combo in itertools.combinations(layers, size):
            # Hard requirement: keep only combos containing ALL required vars.
            if required and not {r.upper() for r in required}.issubset(
                    {c.upper() for c in combo}):
                continue
            # Optional multicollinearity screen: drop combos with a hot pair.
            if matrix is not None:
                hot = False
                for a, b in itertools.combinations(combo, 2):
                    r = matrix.get((a, b), matrix.get((b, a)))
                    if r is not None and abs(r) >= correlation_threshold:
                        hot = True
                        break
                if hot:
                    continue
            folder = combos_dir / "_".join(combo)   # name encodes the variable set
            folder.mkdir(parents=True, exist_ok=True)
            made.append((folder, list(combo)))
    return made


# =============================================================================
# JACKKNIFE VERIFICATION
# =============================================================================
def jackknife_in_results(output_dir: Path) -> bool:
    """
    <<< JACKKNIFE
    True if maxentResults.csv contains jackknife columns ("... without BIO11",
    "... with only BIO11").

    This is the authoritative check that the jackknife actually RAN, rather than
    merely that jackknife=true was requested -- which is exactly the failure
    mode where the flag silently never reached MaxEnt.
    """
    results_csv = output_dir / "maxentResults.csv"
    if not results_csv.is_file():
        return False
    try:
        with results_csv.open(newline="", encoding="utf-8-sig") as fh:
            header = next(csv.reader(fh), [])
    except OSError:
        return False
    return any(JACKKNIFE_RESULT_COL.search(col) for col in header)


# =============================================================================
# SINGLE MAXENT RUN
# =============================================================================
def run_maxent(settings: MaxentSettings,
               samples: Path,
               layers_dir: Path,
               output_dir: Path,
               selected: Sequence[str],
               all_layers: Sequence[str],
               toggle_flag: str,
               dry_run: bool = False,
               log_tail_lines: int = 25) -> dict:
    """
    Execute one MaxEnt model and return a result record.

    Mirrors MaxentCaller$runMaxent: ensures the output dir exists, writes the
    exact command for replay, captures stdout/stderr, reports errors, and
    verifies that a prediction .asc was actually produced.
    """
    output_dir.mkdir(parents=True, exist_ok=True)   # recursive=TRUE equivalent

    cmd = [settings.java_exe]
    cmd += settings.java_args()                      # -Xmx... (must precede -jar)
    cmd += [
        "-jar", str(settings.jar),
        f"samplesfile={samples}",                    # occurrence CSV (species,lat,lon)
        f"environmentallayers={layers_dir}",         # full layer dir, then toggle off
        f"outputdirectory={output_dir}",             # results land in the combo folder
    ]
    cmd += settings.base_args()                                  # visible/autorun/etc.
    cmd += build_layer_args(selected, all_layers, toggle_flag)    # variable subsetting
    cmd += settings.model_args()                                 # shared model params

    # Persist the command so any run can be inspected or replayed by hand.
    # Grep this file for "jackknife=" to confirm what was actually requested.
    (output_dir / "maxent_command.txt").write_text("\n".join(cmd), encoding="utf-8")

    record = {
        "combo": output_dir.name,
        "variables": ";".join(selected),
        "n_variables": len(selected),
        "status": "dry-run",
        "returncode": None,
        "seconds": 0.0,
        "prediction_asc": "",
        "jackknife_ok": False,                       # <<< JACKKNIFE: verified below
    }
    if dry_run:
        print(" ".join(cmd))                         # show, don't launch
        return record

    # <<< JACKKNIFE: jackknife needs >=2 variables to have anything to leave out.
    if settings.jackknife and len(selected) < 2:
        print(f"    !! {output_dir.name}: jackknife requested but only "
              f"{len(selected)} variable(s) -- MaxEnt will skip it", flush=True)

    jk_note = " +jackknife" if settings.jackknife else ""
    print(f"[run] {output_dir.name}: {len(selected)} vars "
          f"({', '.join(selected)}){jk_note}", flush=True)
    start = time.time()
    try:
        # The MaxEnt window will appear because visible=true; autorun=true means
        # it starts immediately and closes itself when finished.
        proc = subprocess.run(cmd,
                              capture_output=True,   # grab stdout+stderr together
                              text=True,             # decode to str
                              check=False)           # inspect returncode ourselves
    except OSError as exc:                           # java missing, exec failure, etc.
        record["status"] = f"launch-error: {exc}"
        print(f"    !! launch error: {exc}", flush=True)
        return record
    record["seconds"] = round(time.time() - start, 1)
    record["returncode"] = proc.returncode

    # Save the full console output next to the model, like maxent_stdout.log in R.
    combined = (proc.stdout or "") + (proc.stderr or "")
    log_path = output_dir / "maxent_stdout.log"
    log_path.write_text(combined, encoding="utf-8")

    # ---- JVM-level failure: fatal for the entire batch, not just this combo --
    # The JVM prints these BEFORE maxent.jar loads, so retrying other combos
    # with identical java args is guaranteed to fail the same way.
    if JVM_STARTUP_PATTERNS.search(combined):
        record["status"] = "jvm-error"
        print("    !! Java failed to start -- this is a JVM problem, "
              "not a MaxEnt/model problem:", flush=True)
        for ln in combined.splitlines()[:log_tail_lines]:
            if ln.strip():
                print(f"       | {ln.rstrip()}", flush=True)
        raise JvmStartupError(
            f"JVM startup failure while running {output_dir.name}. "
            f"See {log_path}. Lower --memory, or install 64-bit Java.")

    # Surface MaxEnt's own complaints (model-level problems).
    bad_lines = [ln for ln in combined.splitlines() if ERROR_PATTERNS.search(ln)]
    for ln in bad_lines[:10]:
        print(f"    !! {ln.strip()}", flush=True)

    # The real success test: a non-diagnostic prediction raster must exist.
    predictions = [p.name for p in output_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in ASC_SUFFIXES
                   and not NON_PREDICTION_ASC.search(p.name)]
    if predictions:
        record["status"] = "ok"
        record["prediction_asc"] = sorted(predictions)[0]
        print(f"    ok ({record['seconds']}s) -> {record['prediction_asc']}", flush=True)

        # <<< JACKKNIFE: confirm the request actually produced importance columns.
        record["jackknife_ok"] = jackknife_in_results(output_dir)
        if settings.jackknife and not record["jackknife_ok"]:
            print("    !! jackknife=true was sent but maxentResults.csv has no "
                  "'without'/'with only' columns -- check the command file",
                  flush=True)
        elif record["jackknife_ok"]:
            print("    jackknife columns present in maxentResults.csv", flush=True)
    else:
        record["status"] = "no-prediction"
        print(f"    !! No prediction .asc in {output_dir} "
              f"(exit code {proc.returncode})", flush=True)
        # Dump the head of the log so the cause is visible immediately, instead
        # of only the lines that happened to match ERROR_PATTERNS.
        head = [ln for ln in combined.splitlines() if ln.strip()][:log_tail_lines]
        if head:
            print("       --- maxent_stdout.log (first lines) ---", flush=True)
            for ln in head:
                print(f"       | {ln.rstrip()}", flush=True)
        print(f"       see {output_dir / 'maxent_command.txt'}", flush=True)
        print(f"       see {log_path}", flush=True)
    return record


def already_done(output_dir: Path) -> bool:
    """Resume support: treat a combo as complete if it has a prediction raster."""
    if not output_dir.is_dir():
        return False
    return any(p.is_file() and p.suffix.lower() in ASC_SUFFIXES
               and not NON_PREDICTION_ASC.search(p.name)
               for p in output_dir.iterdir())


# =============================================================================
# BATCH DRIVER
# =============================================================================
def write_summary(combos_dir: Path, results: Sequence[dict]) -> Path:
    """Write run_summary.csv -- one row per model, sorted for easy diffing."""
    summary_path = combos_dir / "run_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["combo", "variables", "n_variables",
                                                "status", "returncode", "seconds",
                                                "prediction_asc",
                                                "jackknife_ok"])  # <<< JACKKNIFE
        writer.writeheader()
        for row in sorted(results, key=lambda r: r["combo"]):
            writer.writerow(row)
    print(f"[info] summary written: {summary_path}", flush=True)
    return summary_path


def run_all_combinations(settings: MaxentSettings,
                         samples: Path,
                         layers_dir: Path,
                         combos_dir: Path,
                         combos: Sequence[tuple[Path, list[str]]],
                         all_layers: Sequence[str],
                         toggle_flag: str,
                         jobs: int = 1,
                         skip_existing: bool = False,
                         dry_run: bool = False,
                         write_manifests: bool = True,
                         continue_on_jvm_error: bool = False) -> list[dict]:
    """Loop (or thread-pool) over every combination folder and collect results."""

    # Write the combo index up front so the mapping survives a crashed batch.
    with (combos_dir / "combinations.csv").open("w", newline="",
                                                encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["combo_dir", "n_variables", "variables"])
        for folder, variables in combos:
            writer.writerow([folder.name, len(variables), ";".join(variables)])

    pending: list[tuple[Path, list[str]]] = []
    for folder, variables in combos:
        if write_manifests:
            # Drop an underscore-safe manifest so future runs (and the R
            # pipeline's getComboVariables) never have to guess from the name.
            (folder / "_variables.txt").write_text("\n".join(variables) + "\n",
                                                   encoding="utf-8")
        if skip_existing and already_done(folder):
            print(f"[skip] {folder.name} already has a prediction raster", flush=True)
            continue
        pending.append((folder, variables))

    heap_label = (f"-Xmx{settings.java_memory_mb}m" if settings.java_memory_mb
                  else "JVM default heap")
    # <<< JACKKNIFE: echo the effective setting so it is never a mystery again.
    jk_label = (f"jackknife=ON (test={settings.random_test_percentage}%)"
                if settings.jackknife else "jackknife=off")
    print(f"[info] {len(combos)} combination folder(s); {len(pending)} to run; "
          f"jobs={jobs}; visible={settings.visible}; {heap_label}; {jk_label}",
          flush=True)
    if settings.jackknife:
        # 2n+1 models per combo: warn so the runtime is not a surprise.
        est = sum(2 * len(v) + 1 for _, v in pending)
        print(f"[info] jackknife trains 2n+1 models per combo "
              f"(~{est} model fits total across {len(pending)} combos)", flush=True)

    results: list[dict] = []
    try:
        if jobs <= 1:
            # Sequential path: one MaxEnt window at a time, easiest to watch/debug.
            for folder, variables in pending:
                try:
                    results.append(run_maxent(settings, samples, layers_dir, folder,
                                              variables, all_layers, toggle_flag,
                                              dry_run))
                except JvmStartupError:
                    # Record the failure, then stop (or keep going if forced).
                    results.append({"combo": folder.name,
                                    "variables": ";".join(variables),
                                    "n_variables": len(variables),
                                    "status": "jvm-error", "returncode": None,
                                    "seconds": 0.0, "prediction_asc": "",
                                    "jackknife_ok": False})
                    if not continue_on_jvm_error:
                        raise
        else:
            # Parallel path: each run is a separate JVM *and* a separate visible
            # window, so total RAM is roughly jobs * java_memory_mb.
            with ThreadPoolExecutor(max_workers=jobs) as pool:
                futures = {
                    pool.submit(run_maxent, settings, samples, layers_dir, folder,
                                variables, all_layers, toggle_flag, dry_run): folder.name
                    for folder, variables in pending
                }
                for fut in as_completed(futures):
                    try:
                        results.append(fut.result())
                    except JvmStartupError:
                        results.append({"combo": futures[fut], "variables": "",
                                        "n_variables": 0, "status": "jvm-error",
                                        "returncode": None, "seconds": 0.0,
                                        "prediction_asc": "", "jackknife_ok": False})
                        if not continue_on_jvm_error:
                            raise
    finally:
        # Always leave a summary behind, even on an aborted batch.
        write_summary(combos_dir, results)

    return results


# =============================================================================
# CLI
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a MaxEnt model for every combination folder in a directory, "
                    "deriving variables from the folder name (split on '_').")

    # --- I/O (all defaulted so the script can run with no arguments) ---------
    p.add_argument("--jar", type=Path, default=DEFAULT_JAR,
                   help=f"Path to maxent.jar (default: {DEFAULT_JAR})")
    p.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES,
                   help=f"Occurrence CSV: species,latitude,longitude "
                        f"(default: {DEFAULT_SAMPLES})")
    p.add_argument("--layers", type=Path, default=DEFAULT_LAYERS,
                   help=f"Folder of .asc environmental layers (default: {DEFAULT_LAYERS})")
    p.add_argument("--combos", type=Path, default=DEFAULT_COMBOS,
                   help=f"Folder containing one subfolder per combination "
                        f"(default: {DEFAULT_COMBOS})")

    # --- Combination source --------------------------------------------------
    p.add_argument("--mode", choices=["folders", "generate"], default="folders",
                   help="folders = run every existing subfolder of --combos (default); "
                        "generate = create the subfolders first, then run them")
    p.add_argument("--min-size", type=int, default=2,
                   help="[generate] smallest combination size")
    p.add_argument("--max-size", type=int, default=7,
                   help="[generate] largest combination size")
    p.add_argument("--required", nargs="*", default=[],
                   help="[generate] variables that MUST appear in every combination")
    p.add_argument("--correlation-csv", type=Path,
                   help="[generate] Pearson matrix CSV used to drop collinear combos")
    p.add_argument("--correlation-threshold", type=float, default=0.80,
                   help="[generate] exclude combos with any |r| >= this value")
    p.add_argument("--no-manifests", action="store_true",
                   help="Do not write _variables.txt into each combo folder")
    p.add_argument("--toggle-flag", default="togglelayerselected",
                   choices=["togglelayerselected", "togglelayertype"],
                   help="How unused layers are switched off (default correctly "
                        "deselects; togglelayertype replicates MaxentCaller.R)")

    # --- Model parameters ----------------------------------------------------
    p.add_argument("--output-format", default="logistic",
                   choices=["logistic", "cloglog", "cumulative", "raw"],
                   help="Response/output scale (default: logistic)")
    p.add_argument("--no-response-curves", action="store_true",
                   help="Disable response curves (on by default)")

    # <<< JACKKNIFE: an explicit on/off pair sharing one dest, so DEFAULT_JACKKNIFE
    # is honoured and the old "store_true always wins = False" bug is gone.
    p.add_argument("--jackknife", dest="jackknife", action="store_true",
                   default=DEFAULT_JACKKNIFE,
                   help=f"jackknife=true: measure variable importance "
                        f"(trains 2n+1 models per combo -- roughly 3x slower). "
                        f"Default: {DEFAULT_JACKKNIFE}")
    p.add_argument("--no-jackknife", dest="jackknife", action="store_false",
                   help="Disable the jackknife variable-importance runs")
    p.add_argument("--random-test-percentage", type=int, default=DEFAULT_TEST_PCT,
                   help=f"randomtestpoints=N. Needed for Test-gain/AUC jackknife "
                        f"bars; 0 = training-gain chart only "
                        f"(default: {DEFAULT_TEST_PCT})")
    p.add_argument("--no-plots", dest="plots", action="store_false",
                   default=DEFAULT_PLOTS,
                   help="Disable plots= (also removes the jackknife bar-chart PNGs)")

    p.add_argument("--no-pictures", action="store_true",
                   help="Disable PNG map output (on by default)")
    p.add_argument("--write-plot-data", action="store_true", help="writeplotdata=true")
    p.add_argument("--betamultiplier", type=float, default=1.0,
                   help="Regularization multiplier (default: 1.0)")
    p.add_argument("--no-linear", action="store_true", help="Disable linear features")
    p.add_argument("--no-quadratic", action="store_true")
    p.add_argument("--no-product", action="store_true")
    p.add_argument("--no-threshold", action="store_true")
    p.add_argument("--no-hinge", action="store_true")
    p.add_argument("--replicates", type=int, default=1, help="1 = no replication")
    p.add_argument("--replicate-type", default="crossvalidate",
                   choices=["crossvalidate", "bootstrap", "subsample"])
    p.add_argument("--maximum-iterations", type=int)
    p.add_argument("--nodata", type=int, default=-9999,
                   help="Must match the NAflag of your .asc layers")
    p.add_argument("--projection-layers", type=Path,
                   help="Optional folder to project the model onto (mask/future)")

    # --- Runtime -------------------------------------------------------------
    p.add_argument("--hidden", action="store_true",
                   help="Run headless (visible=false); default is a VISIBLE MaxEnt window")
    p.add_argument("--java", default="java", help="Java executable")
    p.add_argument("--memory", type=int, default=2048,
                   help="JVM heap MB per run (-Xmx). Use 0 for the JVM default.")
    p.add_argument("--no-heap-flag", action="store_true",
                   help="Emit no -Xmx at all; let the JVM choose its own heap")
    p.add_argument("--no-heap-fallback", action="store_true",
                   help="Do not auto-reduce the heap if the requested size fails")
    p.add_argument("--skip-java-check", action="store_true",
                   help="Skip the java -version / heap preflight probe")
    p.add_argument("--continue-on-jvm-error", action="store_true",
                   help="Keep going after a JVM startup failure (default: abort)")
    p.add_argument("--threads", type=int, default=1, help="MaxEnt threads per model")
    p.add_argument("--jobs", type=int, default=1,
                   help="Concurrent MaxEnt processes (each opens its own window)")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip combos that already produced a prediction .asc")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without launching Java")
    p.add_argument("--extra", nargs="*", default=[],
                   help="Raw MaxEnt key=value args appended verbatim")
    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    # None => no -Xmx flag at all; otherwise the requested MB.
    requested_heap: int | None = None if (args.no_heap_flag or args.memory <= 0) \
        else args.memory

    # Assemble the shared settings object from CLI flags.
    # NOTE: every field is supplied explicitly here, which is why the dataclass
    # defaults never take effect at runtime -- change DEFAULT_* instead.
    settings = MaxentSettings(
        jar=args.jar,
        java_memory_mb=requested_heap,
        java_exe=args.java,
        visible=not args.hidden,                     # VISIBLE unless --hidden given
        output_format=args.output_format,
        response_curves=not args.no_response_curves,
        jackknife=args.jackknife,                    # <<< JACKKNIFE: now defaults True
        pictures=not args.no_pictures,
        plots=args.plots,                            # <<< JACKKNIFE
        write_plot_data=args.write_plot_data,
        betamultiplier=args.betamultiplier,
        linear=not args.no_linear,
        quadratic=not args.no_quadratic,
        product=not args.no_product,
        threshold=not args.no_threshold,
        hinge=not args.no_hinge,
        replicates=args.replicates,
        replicate_type=args.replicate_type,
        random_test_percentage=args.random_test_percentage,   # <<< JACKKNIFE
        maximum_iterations=args.maximum_iterations,
        nodata=args.nodata,
        projection_layers=args.projection_layers,
        threads=args.threads,
        extra_args=list(args.extra),
    )

    try:
        if not args.dry_run:
            settings.validate()                      # jar + java sanity check
            if not args.samples.is_file():
                raise FileNotFoundError(
                    f"Samples file not found: {args.samples} "
                    "(pass --samples, or edit DEFAULT_SAMPLES at the top of this script)")

            # ---- JVM PREFLIGHT ---------------------------------------------
            # Catch "Error occurred during initialization of VM" ONCE, here,
            # instead of 33 times inside the model loop.
            if not args.skip_java_check:
                settings.java_memory_mb = negotiate_heap(
                    settings.java_exe, requested_heap,
                    allow_fallback=not args.no_heap_fallback)

        # <<< JACKKNIFE: flag combinations that quietly produce fewer charts.
        if settings.jackknife:
            if settings.replicates > 1 and settings.random_test_percentage > 0:
                print("[warn] replicates > 1 makes MaxEnt ignore randomtestpoints; "
                      "the replicate folds supply the test data instead",
                      file=sys.stderr)
            elif settings.replicates <= 1 and settings.random_test_percentage <= 0:
                print("[warn] jackknife with no test split: you will get the "
                      "TRAINING-gain chart only (no Test gain / AUC bars). "
                      "Pass --random-test-percentage 25 to enable them",
                      file=sys.stderr)
            if not settings.plots or not settings.pictures:
                print("[warn] jackknife bar-chart PNGs need both plots=true and "
                      "pictures=true; importance columns will still be written "
                      "to maxentResults.csv", file=sys.stderr)

        # Many visible windows at once is usually a mistake -- warn, don't block.
        if settings.visible and args.jobs > 1:
            print(f"[warn] --jobs {args.jobs} with a visible MaxEnt window will open "
                  f"{args.jobs} GUIs at once; consider --hidden for batch runs",
                  file=sys.stderr)

        all_layers = discover_layers(args.layers)     # real on-disk stems
        lookup = build_layer_lookup(all_layers)       # case-insensitive resolver
        print(f"[info] {len(all_layers)} layer(s) in {args.layers}: "
              f"{', '.join(all_layers)}", flush=True)

        if args.mode == "generate":
            args.combos.mkdir(parents=True, exist_ok=True)
            combos = generate_combo_folders(args.combos, all_layers,
                                            args.min_size, args.max_size,
                                            args.required,
                                            args.correlation_csv,
                                            args.correlation_threshold)
            print(f"[info] generated {len(combos)} combination folder(s)", flush=True)
        else:
            # Default path: every folder already sitting in --combos is a model.
            combos = discover_combo_folders(args.combos, lookup)
            print(f"[info] found {len(combos)} combination folder(s) in {args.combos}",
                  flush=True)

        results = run_all_combinations(settings, args.samples, args.layers, args.combos,
                                       combos, all_layers, args.toggle_flag,
                                       jobs=max(1, args.jobs),
                                       skip_existing=args.skip_existing,
                                       dry_run=args.dry_run,
                                       write_manifests=not args.no_manifests,
                                       continue_on_jvm_error=args.continue_on_jvm_error)

        failures = [r for r in results if r["status"] not in ("ok", "dry-run")]
        # <<< JACKKNIFE: report how many runs genuinely produced importance data.
        if settings.jackknife and not args.dry_run:
            jk_ok = sum(1 for r in results if r.get("jackknife_ok"))
            print(f"[done] jackknife columns found in {jk_ok}/{len(results)} "
                  f"maxentResults.csv file(s)", flush=True)
        print(f"[done] {len(results) - len(failures)} succeeded, {len(failures)} failed",
              flush=True)
        return 0 if not failures else 2               # non-zero exit for scripting/CI

    except JvmStartupError as exc:                    # dedicated, actionable message
        print(f"[fatal] {exc}", file=sys.stderr)
        print("[hint] Try: --memory 1024   (or --no-heap-flag)", file=sys.stderr)
        print("[hint] Check `java -version`; install 64-bit Temurin 17/21 if 32-bit.",
              file=sys.stderr)
        return 3
    except Exception as exc:                          # one tidy error line, no traceback
        print(f"[fatal] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())