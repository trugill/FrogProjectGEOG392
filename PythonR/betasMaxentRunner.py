#!/usr/bin/env python3
"""
maxent_beta_sweep.py
====================
Command-line MaxEnt driver for REGULARIZATION (beta multiplier) optimization.

Runs the SAME fixed variable set over and over, changing only `betamultiplier`,
writing each run into its own folder:

    D:/Frogs/Models/betas/Beta1.0
    D:/Frogs/Models/betas/Beta1.5
    ...
    D:/Frogs/Models/betas/Beta5.0

Variables are FIXED (no combinations) -- the winning set from the combination
step. Every raster in --layers that is NOT in that set is toggled off, so the
model composition is identical across all betas and the only thing varying is
regularization strength.

Python analogue of MaxentCaller.R + Step12Config (beta optimization).

Zero-argument usage (edit the DEFAULT_* constants below first):
    python maxent_beta_sweep.py

Typical explicit usage:
    python maxent_beta_sweep.py ^
        --jar D:/Frogs/maxent/maxent.jar ^
        --samples D:/Frogs/CSV/points.refined_trimdupes_grid.csv ^
        --layers D:/Frogs/ASC/ASC_selected ^
        --betas-dir D:/Frogs/Models/betas ^
        --variables BIO11 BIO16 BIO18 BIO3 BIO8 ^
        --min-beta 1.0 --max-beta 5.0 --beta-step 0.5

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

WHAT TO LOOK AT WHEN IT FINISHES:
    beta_comparison.csv in the betas root -- one row per beta with training
    gain, AUC, sample size, and the non-zero parameter count from the .lambdas
    file. Parameter count should DECREASE as beta increases; if it doesn't, the
    betamultiplier argument is not reaching MaxEnt.
"""

from __future__ import annotations          # modern type hints on older Pythons

import argparse                             # CLI parsing
import csv                                  # summary / comparison CSV writing
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
DEFAULT_JAR       = Path("D:/Frogs/maxent/maxent.jar")                      # maxent.jar
DEFAULT_LAYERS    = Path("D:/Frogs/ASC/ASC_selected")                       # flat .asc folder
DEFAULT_BETAS_DIR = Path("D:/Frogs/Models/betas")                           # one subfolder per beta
DEFAULT_SAMPLES   = Path("D:/Frogs/CSV/points.refined_trimdupes_grid.csv")  # occurrence CSV

# --- The winning variable set from the combination-selection step ------------
# FIXED for every run in this sweep. Any raster in DEFAULT_LAYERS that is not
# listed here (e.g. BIO4) gets toggled OFF, so all betas see the same 5 layers.
DEFAULT_VARIABLES = ["BIO11", "BIO16", "BIO18", "BIO3", "BIO8"]

# --- Beta ladder ------------------------------------------------------------
DEFAULT_MIN_BETA  = 1.0     # first betamultiplier tested
DEFAULT_MAX_BETA  = 5.0     # last betamultiplier tested (inclusive)
DEFAULT_BETA_STEP = 0.5     # increment -> 1.0, 1.5, 2.0, ... 5.0 (9 runs)
BETA_FOLDER_FMT   = "Beta{:.1f}"   # "Beta1.0" -- one decimal, matches existing tree

# --- Model-parameter defaults ------------------------------------------------
# These DO take effect, because they are wired into the argparse defaults
# (unlike the MaxentSettings dataclass defaults, which main() overwrites).
DEFAULT_JACKKNIFE = True    # measure variable importance (trains 2n+1 models: SLOW)
DEFAULT_PLOTS     = True    # plots= must be true for the jackknife bar-chart PNGs
DEFAULT_TEST_PCT  = 0       # randomtestpoints=. KEEP 0 for AICc-based beta
                            # selection: withholding points changes both the
                            # likelihood and the parameter count, making scores
                            # incomparable. Set 25 only if you want AUC instead.

# Accepted raster extensions, compared case-insensitively because this dataset
# uses uppercase ".ASC" on disk while MaxEnt conventionally writes ".asc".
ASC_SUFFIXES = {".asc"}

# Rasters that MaxEnt writes even for partially failed runs -- these must NOT
# count as evidence that a model succeeded.
NON_PREDICTION_ASC = re.compile(r"_clamping|_novel", re.IGNORECASE)

# Log lines indicating MaxEnt complained, mirroring the R grep() call.
ERROR_PATTERNS = re.compile(r"error|exception|failed|cannot|missing", re.IGNORECASE)

# maxentResults.csv header fragments that only exist when the jackknife ran.
JACKKNIFE_RESULT_COL = re.compile(r"without |with only ", re.IGNORECASE)

# Lambdas lines that are MaxEnt metadata, not fitted features. Mirrors the
# metadata_keywords vector in RFunctions$read_lambdas.
LAMBDAS_METADATA = {"linearpredictornormalizer", "densitynormalizer",
                    "numbackgroundpoints", "entropy"}

# Fingerprints of a JVM that died before maxent.jar was ever loaded. These are
# fatal for the WHOLE sweep (not just one beta), so they get special handling.
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
    """Raised when Java fails to boot -- aborts the sweep instead of looping."""


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
    MaxEnt/Java knobs that stay constant across every BETA in the sweep.

    `betamultiplier` is deliberately NOT stored here -- it is the one thing that
    varies, so it is passed per-run into run_maxent().

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
    jackknife: bool = DEFAULT_JACKKNIFE        # jackknife= (2n+1 models per run)
    pictures: bool = True                      # pictures= (write PNG maps)
    plots: bool = DEFAULT_PLOTS                # plots= (jackknife bar charts)
    write_plot_data: bool = False              # writeplotdata=

    # --- Feature classes (held constant so beta is the only variable) --------
    linear: bool = True
    quadratic: bool = True
    product: bool = True
    threshold: bool = True
    hinge: bool = True

    # --- Replication / test split -------------------------------------------
    replicates: int = 1                        # 1 = single model, no replication
    replicate_type: str = "crossvalidate"      # crossvalidate | bootstrap | subsample
    random_test_percentage: int = DEFAULT_TEST_PCT   # randomtestpoints=; 0 = none

    # --- Misc ----------------------------------------------------------------
    random_seed: bool = False                  # randomseed=; False = reproducible
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

    def model_args(self, betamultiplier: float) -> list[str]:
        """
        Translate the settings into MaxEnt `key=value` tokens for ONE beta.

        `betamultiplier` is the only argument that differs between runs in this
        sweep -- everything else is byte-identical, which is what makes the
        resulting AICc/gain comparison valid.
        """
        def tf(flag: bool) -> str:
            return "true" if flag else "false"

        args = [
            f"betamultiplier={betamultiplier}",     # <<< THE SWEEP VARIABLE
            f"outputformat={self.output_format}",   # logistic response surface, etc.
            f"responsecurves={tf(self.response_curves)}",
            f"jackknife={tf(self.jackknife)}",
            f"pictures={tf(self.pictures)}",
            f"plots={tf(self.plots)}",              # jackknife bar charts live here
            f"writeplotdata={tf(self.write_plot_data)}",
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
        if self.random_test_percentage and self.random_test_percentage > 0:
            # Withhold points so jackknife can report Test gain / AUC. MaxEnt
            # ignores this when replicates > 1 (folds supply the test data).
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

    Requested names and filenames can disagree on case ("BIO11" vs "bio11.asc"),
    so every toggle argument is emitted using the REAL stem.
    """
    return {name.strip().upper(): name for name in all_layers}


def resolve_fixed_variables(requested: Sequence[str],
                            lookup: dict[str, str]) -> list[str]:
    """
    Resolve the fixed variable set against the rasters actually on disk.

    Unlike the combination runner (which skipped bad folders), a missing layer
    here is FATAL: silently running a 4-variable model across a beta sweep would
    invalidate every comparison, so we stop and say which raster is absent.
    """
    resolved: list[str] = []
    missing: list[str] = []
    for tok in requested:
        hit = lookup.get(tok.strip().upper())
        if hit is None:
            missing.append(tok)
        elif hit not in resolved:                  # de-dupe, keep requested order
            resolved.append(hit)
    if missing:
        raise FileNotFoundError(
            "No raster found for requested variable(s): " + ", ".join(missing)
            + ". Available: " + ", ".join(sorted(lookup.values())))
    if not resolved:
        raise ValueError("No variables requested -- pass --variables")
    return resolved


def build_layer_args(selected: Sequence[str],
                     all_layers: Sequence[str],
                     toggle_flag: str = "togglelayerselected") -> list[str]:
    """
    Switch OFF every layer that is not part of the fixed variable set.

    MaxEnt is always handed the whole `environmentallayers` directory, so the
    only way to restrict the variable set is to deselect the unwanted layers
    (here: BIO4, which lost the combination step).
    `togglelayerselected=<name>` removes a layer from the model;
    `togglelayertype=<name>` (used in MaxentCaller.R) merely flips
    continuous<->categorical, which does NOT reduce the variable set.
    """
    chosen = set(selected)                                   # O(1) membership tests
    return [f"{toggle_flag}={name}" for name in all_layers if name not in chosen]


# =============================================================================
# BETA LADDER
# =============================================================================
def beta_ladder(min_beta: float, max_beta: float, step: float) -> list[float]:
    """
    Build the list of beta multipliers to test, inclusive of both endpoints.

    Stepping is done in INTEGER units of (step/1) scaled by 1000 to dodge
    floating-point drift: naive `while b <= max: b += 0.5` accumulates error and
    can yield 2.4999999999, which would then format into a folder name that
    doesn't match the existing "Beta2.5" directory.
    """
    if step <= 0:
        raise ValueError("--beta-step must be positive")
    if max_beta < min_beta:
        raise ValueError("--max-beta must be >= --min-beta")

    scale = 1000                                     # work in thousandths
    lo = int(round(min_beta * scale))
    hi = int(round(max_beta * scale))
    inc = int(round(step * scale))

    values: list[float] = []
    cur = lo
    while cur <= hi:
        values.append(cur / scale)                   # exact-enough float, e.g. 2.5
        cur += inc
    return values


def beta_folder(betas_dir: Path, beta: float) -> Path:
    """
    Map a beta value onto its output folder: 1.0 -> "<betas_dir>/Beta1.0".

    Formatting is pinned to one decimal place (BETA_FOLDER_FMT) so the names
    match the existing tree exactly -- "Beta1" or "Beta1.00" would create
    duplicate folders alongside the real ones.
    """
    return betas_dir / BETA_FOLDER_FMT.format(beta)


# =============================================================================
# RESULT HARVESTING (for the beta comparison table)
# =============================================================================
def jackknife_in_results(output_dir: Path) -> bool:
    """
    True if maxentResults.csv contains jackknife columns ("... without BIO11",
    "... with only BIO11").

    Authoritative check that the jackknife actually RAN, rather than merely that
    jackknife=true was requested.
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


def count_lambdas_parameters(output_dir: Path) -> tuple[int | None, str]:
    """
    Count non-zero fitted features in the .lambdas file -> (count, filename).

    Mirrors RFunctions$read_lambdas: drop the four MaxEnt metadata lines
    (linearPredictorNormalizer, densityNormalizer, numBackgroundPoints,
    entropy) and any feature whose weight is exactly zero. This is the `k` that
    AIC/AICc/BIC needs, and it is the headline number for a beta sweep: higher
    beta = stronger regularization = fewer retained parameters.
    """
    lambdas = sorted(p for p in output_dir.glob("*.lambdas"))
    if not lambdas:
        return None, ""
    path = lambdas[0]
    count = 0
    try:
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = raw.split(",")
            if len(parts) < 2:
                continue                              # not a feature line
            if parts[0].strip().lower() in LAMBDAS_METADATA:
                continue                              # metadata, not a parameter
            try:
                weight = float(parts[1].strip())
            except ValueError:
                continue                              # unparseable weight
            if weight != 0.0:
                count += 1
    except OSError:
        return None, path.name
    return count, path.name


def harvest_maxent_results(output_dir: Path) -> dict[str, str]:
    """
    Pull the headline metrics out of maxentResults.csv for the comparison table.

    MaxEnt's column names vary slightly by version and by whether test data
    exists, so each metric is matched against several candidate headers and
    missing ones simply come back blank rather than raising.
    """
    results_csv = output_dir / "maxentResults.csv"
    out = {"samples": "", "train_gain": "", "train_gain_unreg": "",
           "train_auc": "", "test_auc": "", "features_in_model": ""}
    if not results_csv.is_file():
        return out
    try:
        with results_csv.open(newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return out
    if not rows:
        return out
    # With replicates>1 MaxEnt appends an "(average)" summary row; take the last
    # row so we read that average, otherwise the single model's row.
    row = rows[-1]

    def pick(*candidates: str) -> str:
        """First matching column (case-insensitive, substring-tolerant)."""
        for cand in candidates:
            for key, val in row.items():
                if key and key.strip().lower() == cand.lower():
                    return (val or "").strip()
        for cand in candidates:
            for key, val in row.items():
                if key and cand.lower() in key.strip().lower():
                    return (val or "").strip()
        return ""

    out["samples"]           = pick("#Training samples", "X.Training.samples")
    out["train_gain"]        = pick("Regularized training gain")
    out["train_gain_unreg"]  = pick("Unregularized training gain")
    out["train_auc"]         = pick("Training AUC")
    out["test_auc"]          = pick("Test AUC")
    out["features_in_model"] = pick("#Features", "X.Features")
    return out


# =============================================================================
# SINGLE MAXENT RUN
# =============================================================================
def run_maxent(settings: MaxentSettings,
               samples: Path,
               layers_dir: Path,
               output_dir: Path,
               betamultiplier: float,
               selected: Sequence[str],
               all_layers: Sequence[str],
               toggle_flag: str,
               dry_run: bool = False,
               log_tail_lines: int = 25) -> dict:
    """
    Execute ONE MaxEnt model at ONE beta multiplier and return a result record.

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
        f"outputdirectory={output_dir}",             # results land in Beta<N.N>/
    ]
    cmd += settings.base_args()                                  # visible/autorun/etc.
    cmd += build_layer_args(selected, all_layers, toggle_flag)    # fix the variable set
    cmd += settings.model_args(betamultiplier)                   # incl. betamultiplier

    # Persist the command + the two things that define this run, so any folder
    # is self-describing. Grep maxent_command.txt for "betamultiplier=".
    (output_dir / "maxent_command.txt").write_text("\n".join(cmd), encoding="utf-8")
    (output_dir / "_variables.txt").write_text("\n".join(selected) + "\n",
                                               encoding="utf-8")
    (output_dir / "_beta.txt").write_text(f"{betamultiplier}\n", encoding="utf-8")

    record = {
        "beta": f"{betamultiplier:.1f}",
        "folder": output_dir.name,
        "variables": ";".join(selected),
        "n_variables": len(selected),
        "status": "dry-run",
        "returncode": None,
        "seconds": 0.0,
        "prediction_asc": "",
        "jackknife_ok": False,
        "parameters": "",                            # non-zero lambdas (the AIC `k`)
        "lambdas_file": "",
        "samples": "",
        "train_gain": "",
        "train_gain_unreg": "",
        "train_auc": "",
        "test_auc": "",
        "features_in_model": "",
    }
    if dry_run:
        print(" ".join(cmd))                         # show, don't launch
        return record

    jk_note = " +jackknife" if settings.jackknife else ""
    print(f"[run] {output_dir.name}: beta={betamultiplier:.1f}, "
          f"{len(selected)} vars ({', '.join(selected)}){jk_note}", flush=True)
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

    # ---- JVM-level failure: fatal for the whole sweep, not just this beta ----
    # The JVM prints these BEFORE maxent.jar loads, so retrying other betas
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

        # Harvest the numbers that make the beta sweep interpretable.
        nparams, lambdas_name = count_lambdas_parameters(output_dir)
        record["parameters"] = "" if nparams is None else str(nparams)
        record["lambdas_file"] = lambdas_name
        record.update(harvest_maxent_results(output_dir))
        record["jackknife_ok"] = jackknife_in_results(output_dir)

        print(f"    ok ({record['seconds']}s) -> {record['prediction_asc']}; "
              f"params={record['parameters'] or '?'}, "
              f"gain={record['train_gain'] or '?'}", flush=True)
        if settings.jackknife and not record["jackknife_ok"]:
            print("    !! jackknife=true was sent but maxentResults.csv has no "
                  "'without'/'with only' columns -- check the command file",
                  flush=True)
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
    """Resume support: treat a beta as complete if it has a prediction raster."""
    if not output_dir.is_dir():
        return False
    return any(p.is_file() and p.suffix.lower() in ASC_SUFFIXES
               and not NON_PREDICTION_ASC.search(p.name)
               for p in output_dir.iterdir())


# =============================================================================
# SWEEP DRIVER
# =============================================================================
SUMMARY_FIELDS = ["beta", "folder", "variables", "n_variables", "status",
                  "returncode", "seconds", "prediction_asc", "jackknife_ok",
                  "parameters", "lambdas_file", "samples", "train_gain",
                  "train_gain_unreg", "train_auc", "test_auc",
                  "features_in_model"]


def write_summary(betas_dir: Path, results: Sequence[dict]) -> Path:
    """Write run_summary.csv -- one row per beta, ordered by beta value."""
    summary_path = betas_dir / "run_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in sorted(results, key=lambda r: float(r["beta"])):
            writer.writerow({k: row.get(k, "") for k in SUMMARY_FIELDS})
    print(f"[info] summary written: {summary_path}", flush=True)
    return summary_path


def write_comparison(betas_dir: Path, results: Sequence[dict]) -> Path:
    """
    Write beta_comparison.csv -- the trimmed, decision-oriented table.

    This is what you actually read to choose a beta: parameter count and gain
    per beta, side by side. Feed `parameters` + a log-likelihood into your
    AIC/AICc/BIC step to complete Step 12.
    """
    comparison_path = betas_dir / "beta_comparison.csv"
    fields = ["beta", "folder", "status", "parameters", "samples",
              "train_gain", "train_gain_unreg", "train_auc", "test_auc",
              "features_in_model", "seconds", "prediction_asc"]
    with comparison_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in sorted(results, key=lambda r: float(r["beta"])):
            writer.writerow({k: row.get(k, "") for k in fields})
    print(f"[info] comparison written: {comparison_path}", flush=True)
    return comparison_path


def print_comparison_table(results: Sequence[dict]) -> None:
    """
    Echo a compact beta/params/gain table to the console.

    Parameter count should DECREASE monotonically-ish as beta rises. If it is
    flat across all betas, `betamultiplier` is not reaching MaxEnt and the whole
    sweep is nine copies of the same model.
    """
    ok = [r for r in results if r["status"] == "ok"]
    if not ok:
        return
    print("\n  beta   params   train_gain   train_AUC   test_AUC", flush=True)
    print("  ----   ------   ----------   ---------   --------", flush=True)
    for r in sorted(ok, key=lambda r: float(r["beta"])):
        print(f"  {r['beta']:>4}   {r['parameters'] or '?':>6}   "
              f"{r['train_gain'] or '?':>10}   {r['train_auc'] or '?':>9}   "
              f"{r['test_auc'] or '-':>8}", flush=True)
    distinct = {r["parameters"] for r in ok if r["parameters"]}
    if len(distinct) == 1:
        print("\n  !! Every beta produced the SAME parameter count. Verify that "
              "betamultiplier= differs between the maxent_command.txt files.",
              flush=True)
    print("", flush=True)


def run_beta_sweep(settings: MaxentSettings,
                   samples: Path,
                   layers_dir: Path,
                   betas_dir: Path,
                   betas: Sequence[float],
                   selected: Sequence[str],
                   all_layers: Sequence[str],
                   toggle_flag: str,
                   jobs: int = 1,
                   skip_existing: bool = False,
                   dry_run: bool = False,
                   continue_on_jvm_error: bool = False) -> list[dict]:
    """Loop (or thread-pool) over every beta multiplier and collect results."""
    betas_dir.mkdir(parents=True, exist_ok=True)

    # Record the sweep definition up front so it survives a crashed batch.
    (betas_dir / "_variables.txt").write_text("\n".join(selected) + "\n",
                                              encoding="utf-8")
    with (betas_dir / "beta_runs.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["beta", "folder", "variables"])
        for beta in betas:
            writer.writerow([f"{beta:.1f}", beta_folder(betas_dir, beta).name,
                             ";".join(selected)])

    pending: list[tuple[float, Path]] = []
    for beta in betas:
        out_dir = beta_folder(betas_dir, beta)
        out_dir.mkdir(parents=True, exist_ok=True)   # reuse the existing tree
        if skip_existing and already_done(out_dir):
            print(f"[skip] {out_dir.name} already has a prediction raster", flush=True)
            continue
        pending.append((beta, out_dir))

    heap_label = (f"-Xmx{settings.java_memory_mb}m" if settings.java_memory_mb
                  else "JVM default heap")
    jk_label = (f"jackknife=ON (test={settings.random_test_percentage}%)"
                if settings.jackknife else "jackknife=off")
    print(f"[info] {len(betas)} beta value(s): "
          f"{', '.join(f'{b:.1f}' for b in betas)}", flush=True)
    print(f"[info] {len(pending)} to run; jobs={jobs}; visible={settings.visible}; "
          f"{heap_label}; {jk_label}", flush=True)
    if settings.jackknife:
        # 2n+1 models per beta: warn so the runtime is not a surprise.
        est = (2 * len(selected) + 1) * len(pending)
        print(f"[info] jackknife trains 2n+1 models per beta "
              f"(~{est} model fits total)", flush=True)

    results: list[dict] = []
    try:
        if jobs <= 1:
            # Sequential path: one MaxEnt window at a time, easiest to watch.
            for beta, out_dir in pending:
                try:
                    results.append(run_maxent(settings, samples, layers_dir, out_dir,
                                              beta, selected, all_layers,
                                              toggle_flag, dry_run))
                except JvmStartupError:
                    results.append({"beta": f"{beta:.1f}", "folder": out_dir.name,
                                    "variables": ";".join(selected),
                                    "n_variables": len(selected),
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
                    pool.submit(run_maxent, settings, samples, layers_dir, out_dir,
                                beta, selected, all_layers, toggle_flag,
                                dry_run): (beta, out_dir)
                    for beta, out_dir in pending
                }
                for fut in as_completed(futures):
                    beta, out_dir = futures[fut]
                    try:
                        results.append(fut.result())
                    except JvmStartupError:
                        results.append({"beta": f"{beta:.1f}", "folder": out_dir.name,
                                        "variables": ";".join(selected),
                                        "n_variables": len(selected),
                                        "status": "jvm-error", "returncode": None,
                                        "seconds": 0.0, "prediction_asc": "",
                                        "jackknife_ok": False})
                        if not continue_on_jvm_error:
                            raise
    finally:
        # Always leave the tables behind, even on an aborted sweep.
        write_summary(betas_dir, results)
        write_comparison(betas_dir, results)
        if not dry_run:
            print_comparison_table(results)

    return results


# =============================================================================
# CLI
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run MaxEnt repeatedly on a FIXED variable set, varying only "
                    "the beta multiplier, into Beta<N.N>/ folders.")

    # --- I/O (all defaulted so the script can run with no arguments) ---------
    p.add_argument("--jar", type=Path, default=DEFAULT_JAR,
                   help=f"Path to maxent.jar (default: {DEFAULT_JAR})")
    p.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES,
                   help=f"Occurrence CSV: species,latitude,longitude "
                        f"(default: {DEFAULT_SAMPLES})")
    p.add_argument("--layers", type=Path, default=DEFAULT_LAYERS,
                   help=f"Folder of .asc environmental layers (default: {DEFAULT_LAYERS})")
    p.add_argument("--betas-dir", type=Path, default=DEFAULT_BETAS_DIR,
                   help=f"Root for the Beta<N.N> output folders "
                        f"(default: {DEFAULT_BETAS_DIR})")

    # --- Fixed variable set --------------------------------------------------
    p.add_argument("--variables", nargs="+", default=DEFAULT_VARIABLES,
                   help=f"The winning variable set, used for EVERY beta "
                        f"(default: {' '.join(DEFAULT_VARIABLES)})")
    p.add_argument("--toggle-flag", default="togglelayerselected",
                   choices=["togglelayerselected", "togglelayertype"],
                   help="How non-selected layers are switched off (default "
                        "correctly deselects; togglelayertype replicates "
                        "MaxentCaller.R)")

    # --- Beta ladder ---------------------------------------------------------
    p.add_argument("--min-beta", type=float, default=DEFAULT_MIN_BETA,
                   help=f"First beta multiplier (default: {DEFAULT_MIN_BETA})")
    p.add_argument("--max-beta", type=float, default=DEFAULT_MAX_BETA,
                   help=f"Last beta multiplier, inclusive (default: {DEFAULT_MAX_BETA})")
    p.add_argument("--beta-step", type=float, default=DEFAULT_BETA_STEP,
                   help=f"Beta increment (default: {DEFAULT_BETA_STEP})")

    # --- Model parameters (held constant across the sweep) -------------------
    p.add_argument("--output-format", default="logistic",
                   choices=["logistic", "cloglog", "cumulative", "raw"],
                   help="Response/output scale (default: logistic)")
    p.add_argument("--no-response-curves", action="store_true",
                   help="Disable response curves (on by default)")
    p.add_argument("--jackknife", dest="jackknife", action="store_true",
                   default=DEFAULT_JACKKNIFE,
                   help=f"jackknife=true: measure variable importance "
                        f"(trains 2n+1 models per beta). Default: {DEFAULT_JACKKNIFE}")
    p.add_argument("--no-jackknife", dest="jackknife", action="store_false",
                   help="Disable the jackknife variable-importance runs")
    p.add_argument("--random-test-percentage", type=int, default=DEFAULT_TEST_PCT,
                   help=f"randomtestpoints=N. Keep 0 for AICc-based beta "
                        f"selection; use 25 for AUC diagnostics "
                        f"(default: {DEFAULT_TEST_PCT})")
    p.add_argument("--no-plots", dest="plots", action="store_false",
                   default=DEFAULT_PLOTS,
                   help="Disable plots= (also removes the jackknife bar charts)")
    p.add_argument("--no-pictures", action="store_true",
                   help="Disable PNG map output (on by default)")
    p.add_argument("--write-plot-data", action="store_true", help="writeplotdata=true")
    p.add_argument("--no-linear", action="store_true", help="Disable linear features")
    p.add_argument("--no-quadratic", action="store_true")
    p.add_argument("--no-product", action="store_true")
    p.add_argument("--no-threshold", action="store_true")
    p.add_argument("--no-hinge", action="store_true")
    p.add_argument("--replicates", type=int, default=1, help="1 = no replication")
    p.add_argument("--replicate-type", default="crossvalidate",
                   choices=["crossvalidate", "bootstrap", "subsample"])
    p.add_argument("--random-seed", action="store_true",
                   help="randomseed=true. Leave OFF so betas stay comparable.")
    p.add_argument("--maximum-iterations", type=int)
    p.add_argument("--nodata", type=int, default=-9999,
                   help="Must match the NAflag of your .asc layers")
    p.add_argument("--projection-layers", type=Path,
                   help="Optional folder to project the model onto (mask/future)")

    # --- Runtime -------------------------------------------------------------
    p.add_argument("--hidden", action="store_true",
                   help="Run headless (visible=false); default is a VISIBLE window")
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
                   help="Skip betas that already produced a prediction .asc")
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

    # Assemble the settings shared by every beta. NOTE: every field is supplied
    # explicitly here, which is why the dataclass defaults never take effect at
    # runtime -- change the DEFAULT_* constants instead.
    settings = MaxentSettings(
        jar=args.jar,
        java_memory_mb=requested_heap,
        java_exe=args.java,
        visible=not args.hidden,                     # VISIBLE unless --hidden given
        output_format=args.output_format,
        response_curves=not args.no_response_curves,
        jackknife=args.jackknife,
        pictures=not args.no_pictures,
        plots=args.plots,
        write_plot_data=args.write_plot_data,
        linear=not args.no_linear,
        quadratic=not args.no_quadratic,
        product=not args.no_product,
        threshold=not args.no_threshold,
        hinge=not args.no_hinge,
        replicates=args.replicates,
        replicate_type=args.replicate_type,
        random_test_percentage=args.random_test_percentage,
        random_seed=args.random_seed,
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
            # instead of nine times inside the sweep loop.
            if not args.skip_java_check:
                settings.java_memory_mb = negotiate_heap(
                    settings.java_exe, requested_heap,
                    allow_fallback=not args.no_heap_fallback)

        # Flag settings that quietly undermine a beta comparison.
        if args.random_seed:
            print("[warn] --random-seed makes each beta use a different background "
                  "sample; gains/AICc will not be strictly comparable",
                  file=sys.stderr)
        if settings.random_test_percentage > 0 and settings.replicates <= 1:
            print(f"[warn] randomtestpoints={settings.random_test_percentage} "
                  "withholds points, which changes the likelihood AND the "
                  "parameter count -- use 0 if these runs feed AICc selection",
                  file=sys.stderr)
        if settings.jackknife and not (settings.plots and settings.pictures):
            print("[warn] jackknife bar-chart PNGs need both plots=true and "
                  "pictures=true; importance columns will still be written",
                  file=sys.stderr)
        if settings.visible and args.jobs > 1:
            print(f"[warn] --jobs {args.jobs} with a visible MaxEnt window will open "
                  f"{args.jobs} GUIs at once; consider --hidden for batch runs",
                  file=sys.stderr)

        # ---- Resolve the FIXED variable set once, up front ------------------
        all_layers = discover_layers(args.layers)     # real on-disk stems
        lookup = build_layer_lookup(all_layers)       # case-insensitive resolver
        print(f"[info] {len(all_layers)} layer(s) in {args.layers}: "
              f"{', '.join(all_layers)}", flush=True)

        selected = resolve_fixed_variables(args.variables, lookup)
        excluded = [n for n in all_layers if n not in set(selected)]
        print(f"[info] fixed variable set ({len(selected)}): {', '.join(selected)}",
              flush=True)
        print(f"[info] toggled OFF ({len(excluded)}): "
              f"{', '.join(excluded) if excluded else '(none)'}", flush=True)

        # ---- Build the beta ladder -----------------------------------------
        betas = beta_ladder(args.min_beta, args.max_beta, args.beta_step)
        print(f"[info] beta folders -> {args.betas_dir}"
              f"\\{BETA_FOLDER_FMT.format(betas[0])} ... "
              f"{BETA_FOLDER_FMT.format(betas[-1])}", flush=True)

        results = run_beta_sweep(settings, args.samples, args.layers, args.betas_dir,
                                 betas, selected, all_layers, args.toggle_flag,
                                 jobs=max(1, args.jobs),
                                 skip_existing=args.skip_existing,
                                 dry_run=args.dry_run,
                                 continue_on_jvm_error=args.continue_on_jvm_error)

        failures = [r for r in results if r["status"] not in ("ok", "dry-run")]
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