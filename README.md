# FrogProjectGEOG392





\# /PythonR/ — MaxEnt Species Distribution Modeling Pipeline



A mixed Python + R toolkit for preparing environmental raster layers and

occurrence records, screening predictors for collinearity, and batch-driving

MaxEnt (`maxent.jar`) over variable combinations and regularization settings.



The scripts were written around a specific project tree on Windows

(`D:/Frogs/...`), with hard-coded paths in a `SETTINGS` / `DEFAULT\_\*` block at

the top of each file. They are meant to be edited and run, not imported as a

library.



\---



\## Pipeline Overview



| Stage | Script | Language |

|-------|--------|----------|

| 1. Clip rasters to study extent, convert GeoTIFF → ESRI ASCII | `trim\_rasters\_in\_polygon.py` | Python |

| 1b. Diagnose/rescue a single problem raster | `trim\_rasters\_in\_polygon\_exception.py` | Python |

| 2. Split ASCII grids into continental subsets | `trimASCtoContinents.R` | R |

| 3. Clip occurrence points to the same extent | `trim\_points\_in\_polygon.py` | Python |

| 4. Pearson correlation matrix among candidate layers | `correlation.R`, `Generating correlations.R` | R |

| 5. Enumerate non-collinear variable combinations | `combos.R` | R |

| 6. Run MaxEnt once per variable combination | `combosMaxentRunner.py` | Python |

| 7. Run MaxEnt across a regularization (beta) ladder | `betasMaxentRunner.py` | Python |



\---



\## Dependencies



\*\*Python\*\* (3.9+; the MaxEnt runners use `from \_\_future\_\_ import annotations`

for modern type-hint syntax)



\- `rasterio` — raster I/O, `WarpedVRT` reprojection, windowed reads

\- `numpy` — array math, NoData masking, `savetxt` for ASCII grids

\- `pandas` — occurrence CSV handling

\- Standard library: `pathlib`, `argparse`, `csv`, `re`, `subprocess`, `shutil`,

&#x20; `itertools`, `dataclasses`, `concurrent.futures`, `time`, `sys`, `os`,

&#x20; `traceback`



\*\*R\*\*



\- `terra` — modern raster cropping/writing (`rast`, `ext`, `crop`, `writeRaster`)

\- `raster` + `sp` — legacy stack/brick workflow and `layerStats()` correlations

\- Base R: `combn`, `file.choose`, `choose.dir`, `tools::file\_ext`



\*\*External\*\*



\- Java (64-bit strongly recommended — see JVM notes below)

\- `maxent.jar` (MaxEnt 3.4.x command-line interface)



\---



\## Stage 1 — Raster Preparation



\### `trim\_rasters\_in\_polygon.py`



Walks a directory (optionally recursively) for GeoTIFFs and reprojects/crops

every one onto a \*\*single shared grid\*\*, writing ESRI ASCII (`.asc`) output so

the stack is MaxEnt-ready.



Key behaviors:



\- \*\*Fixed target extent\*\*: lon `\[-100, -55]`, lat `\[9, 37]`, EPSG:4326.

\- \*\*Cell size\*\* is either set explicitly or inferred as the smallest native

&#x20; resolution among readable inputs; projected rasters get their resolution

&#x20; converted to approximate degrees via `transform\_bounds`.

\- Reprojection/resampling is done lazily through `WarpedVRT` (bilinear by

&#x20; default, with a note that `nearest` is correct for categorical layers).

\- Rasters with \*\*no CRS\*\* get an assumed one; \*\*0–360 longitude grids\*\* are

&#x20; detected and remapped with a `+lon\_wrap=180` PROJ tweak.

\- The ASCII header is written \*\*by hand\*\* rather than through a GDAL driver, to

&#x20; avoid driver quirks — `ncols`, `nrows`, `xllcorner`, `yllcorner`, `cellsize`,

&#x20; `NODATA\_value` (−9999), then `np.savetxt` for the body.

\- Multi-band inputs can be exploded into `<stem>\_band<N>.asc`.

\- Reports per-file status (`OK` / `EMPTY` / `FAIL`) with data-coverage

&#x20; percentages, then a summary listing all-NoData outputs and unreadable files

&#x20; with reasons.



\### `trim\_rasters\_in\_polygon\_exception.py`



A single-file troubleshooting harness, written after one WorldClim layer

(`wc2.1\_2.5m\_bio\_1.tif`) refused to convert. It runs four staged diagnostics

and then four escalating conversion attempts:



\*\*Diagnostics\*\*

1\. \*Filesystem\* — existence, size sanity check, readability, TIFF magic bytes

&#x20;  (`II`/`MM`).

2\. \*Metadata\* — driver, dimensions, dtype, CRS, resolution vs. the expected

&#x20;  1/24°, bounds, NoData, tiling, compression, and whether the raster actually

&#x20;  overlaps the target extent.

3\. \*Integrity probe\* — reads the \*\*last block\*\* (catches truncated downloads)

&#x20;  plus a mid-raster 64×64 sample to confirm plausible values.



\*\*Conversion strategies\*\*, tried in order until one yields non-empty output:

\- \*\*A\*\* — direct boundless windowed read (fastest, assumes matching CRS)

\- \*\*B\*\* — `WarpedVRT` onto the exact target grid (bilinear)

\- \*\*C\*\* — mask-aware VRT read that ignores a wrong or missing NoData header

\- \*\*D\*\* — row-by-row streaming via the inverse affine transform (lowest memory,

&#x20; survives flaky I/O)



All outputs pass through a `clean()` step that collapses NaN, infinities, and

WorldClim's huge-negative sentinel values (≤ `-1e30`) to −9999. On success it

sanity-checks that values fall in a plausible °C range for bio\_1 and echoes the

new `.asc` header for comparison against the rest of the folder.



\### `trimASCtoContinents.R`



Batch-crops every `.asc` in a folder to five hard-coded rectangular regions —

\*\*SA, NA, Africa, Asia, AusNZ\*\* — producing `<name>\_<region>.asc` per

combination (`terra::crop` with `snap = "out"` to stay aligned to the input

grid).



Notable details:



\- Sets `GDAL\_PAM\_ENABLED = "NO"` to suppress `.aux.xml` sidecar files.

\- Validates that every bounding box has `west < east` and `south < north`.

\- Guards against the classic R gotcha where the region label `"NA"` is parsed as

&#x20; a missing value.

\- Assumes EPSG:4326 for bare `.asc` files that carry no CRS.

\- Skips region/raster pairs whose extents don't intersect, wraps each write in

&#x20; `tryCatch`, logs rows/cols/extent per output, and writes a `trim\_log.csv`

&#x20; alongside the results plus a written/skipped/failed tally.



\---



\## Stage 2 — Occurrence Points



\### `trim\_points\_in\_polygon.py`



Clips an occurrence CSV (`species, latitude, longitude`) to the same rectangular

extent using `pandas`.



\- Flexible column detection: matches common aliases (`decimalLatitude`, `lat`,

&#x20; `y`, `scientificName`, …) case-insensitively, falling back to the first three

&#x20; columns positionally with a warning.

\- Coerces coordinates with `to\_numeric(errors="coerce")` and optionally drops

&#x20; unparseable rows.

\- Warns if any `latitude` exceeds ±90, a signal that lat/lon were swapped.

\- Optional exact-duplicate removal on species + lat + lon.

\- Prints a read/invalid/kept/dropped summary and the top 20 species by record

&#x20; count.



\---



\## Stage 3 — Collinearity Screening



\### `Generating correlations.R`



Interactive script: prompts the user to prepare a folder of `.asc` layers (with

a reminder that MaxEnt-era tooling prefers names ≤ 5 characters), stacks them

with `raster::stack`, converts to a `brick`, sets values `< -100` to `NA` to

strip NoData sentinels, computes `layerStats(..., 'pearson', na.rm = TRUE)`, and

saves the matrix to a user-chosen CSV.



\### `correlation.R`



The same computation refactored into a reusable function, `run\_correlation()`,

for an organized pipeline. It depends on `get\_top\_permutation\_importance()`

(sourced from `permutation\_analyzer.R`) to pick the top predictors from a global

MaxEnt run's `maxentResults.csv`, builds `.asc` paths from those names, errors

out if any file is missing, then stacks → bricks → masks → correlates → writes

to a path supplied in a `paths` config list.



\### `combos.R`



Reads a Pearson matrix CSV, strips the

`pearson.correlation.coefficient.` prefix that R prepends to column names, and

prompts for a correlation threshold. It then enumerates all variable subsets of

size 2 through 7 with `combn()`, rejecting any subset containing a pair whose

`|r|` meets or exceeds the threshold, and creates one directory per surviving

combination named `VAR1\_VAR2\_...` — plus `combinations\_<size>.csv` indexes.

These folders are exactly what `combosMaxentRunner.py` consumes.



> Note: the script is a loose block of top-level code (originally a function

> body), and the default threshold applied on empty input (0.9) differs from the

> prompt text (0.8).



\---



\## Stage 4 — MaxEnt Drivers



Both Python runners are the heart of the repo and share most of their

infrastructure. They replace an earlier R implementation (`MaxentCaller.R`),

invoking `java -jar maxent.jar` with `key=value` arguments via `subprocess`.



\### Shared architecture



\*\*`MaxentSettings` dataclass\*\* — holds every Java/MaxEnt knob that stays

constant across runs: jar path, heap size, GUI visibility, `autorun`,

`redoifexists`, output format (`logistic`/`cloglog`/`cumulative`/`raw`),

response curves, jackknife, pictures/plots, feature classes (linear, quadratic,

product, threshold, hinge), replication, test split, NoData value, projection

layers, thread count, and a raw passthrough list. It exposes `validate()`,

`java\_args()` (JVM flags that must precede `-jar`), `base\_args()`, and

`model\_args()`.



> Both files document a deliberate trap: `main()` reconstructs

> `MaxentSettings` entirely from parsed CLI arguments, so the dataclass field

> defaults are decorative. Permanent changes belong in the `DEFAULT\_\*` module

> constants that feed `argparse`.



\*\*JVM preflight\*\* — a notable chunk of defensive engineering around the

infamous `Error occurred during initialization of VM`:



\- `java\_version\_banner()` captures `java -version` (which prints to stderr).

\- `jvm\_is\_32bit()` heuristically detects a 32-bit VM, which caps heap near

&#x20; \~1.2–1.5 GB regardless of installed RAM.

\- `heap\_is\_reservable()` cheaply probes whether `java -Xmx<N>m -version` starts.

\- `negotiate\_heap()` walks a fallback ladder (1536 → 1024 → 768 → 512 → no

&#x20; `-Xmx` flag) and raises a custom `JvmStartupError` if nothing boots.

\- `JVM\_STARTUP\_PATTERNS` regexes distinguish \*JVM died before MaxEnt loaded\*

&#x20; (fatal for the whole batch) from \*MaxEnt complained about this one model\*.

&#x20; Uses the standard `-Xmx` flag rather than the legacy `-mx` alias the R

&#x20; original used.



\*\*Layer discovery and toggling\*\* — MaxEnt is always handed the whole

`environmentallayers` directory, so restricting the variable set means

\*deselecting\* everything else. `discover\_layers()` collects `.asc` stems

case-insensitively (the dataset uses uppercase `.ASC` on disk),

`build\_layer\_lookup()` builds a case-insensitive token → real-stem map, and

`build\_layer\_args()` emits `togglelayerselected=<name>` for every unwanted

layer. The code explicitly documents that `togglelayertype=` — used in the old R

version — only flips continuous↔categorical and does \*\*not\*\* reduce the variable

set.



\*\*Per-run bookkeeping\*\* — each output folder receives `maxent\_command.txt` (the

exact replayable command line), `maxent\_stdout.log` (combined stdout+stderr),

and manifest files. Success is judged not by exit code but by the existence of a

genuine prediction raster, filtering out `\_clamping` and `\_novel` grids that

MaxEnt writes even for partial failures. Failures dump the head of the log so the

cause is immediately visible. `already\_done()` supports `--skip-existing` resume

behavior.



\*\*Execution\*\* — sequential by default (one visible MaxEnt window at a time, easy

to watch); `--jobs N` switches to a `ThreadPoolExecutor`, with a warning that

each job is a separate JVM \*and\* a separate GUI, so RAM scales with

`jobs × heap`. Summary CSVs are written in a `finally` block so an aborted batch

still leaves a record. Exit codes: `0` success, `1` generic error, `2` some runs

failed, `3` JVM startup failure.



\### `combosMaxentRunner.py`



Runs \*\*one model per combination folder\*\*. Variables come from

`\_variables.txt` inside the folder if present (underscore-safe and

authoritative), otherwise from splitting the folder name on `\_` — with an

explicit caveat that this is only reversible while layer names contain no

underscores. Folders referencing an unknown layer are reported and skipped

rather than silently run wrong.



Two modes:

\- `--mode folders` (default) — execute every existing subfolder of `--combos`.

\- `--mode generate` — a Python port of `combos.R`: enumerate subsets with

&#x20; `itertools.combinations` between `--min-size` and `--max-size`, optionally

&#x20; force `--required` variables into every set, optionally screen pairs against a

&#x20; Pearson matrix CSV loaded by `load\_correlation\_matrix()`, create the folders,

&#x20; then run them.



Heavy emphasis on \*\*jackknife verification\*\* (marked throughout with `<<<

JACKKNIFE` comments, clearly the product of a debugging session where the flag

silently never reached MaxEnt). `jackknife\_in\_results()` inspects

`maxentResults.csv` for `"... without <VAR>"` / `"... with only <VAR>"` columns

as proof the jackknife actually ran, rather than trusting that `jackknife=true`

was requested. The script warns about combinations that quietly produce fewer

outputs: jackknife with fewer than 2 variables, `replicates > 1` making MaxEnt

ignore `randomtestpoints`, no test split meaning training-gain charts only, and

bar-chart PNGs needing both `plots=true` and `pictures=true`. It also warns that

jackknife trains \*2n+1\* models per combo (\~3× slower) and estimates the total

fit count up front. Output: `combinations.csv` and `run\_summary.csv`.



\### `betasMaxentRunner.py`



Regularization optimization. Holds the \*\*winning variable set fixed\*\* and varies

only `betamultiplier`, writing each run into `Beta1.0`, `Beta1.5`, … `Beta5.0`.



\- `beta\_ladder()` builds the ladder in integer thousandths rather than

&#x20; accumulating `+= 0.5` in floating point, so values never drift to

&#x20; `2.4999999999` and miss the existing `Beta2.5` folder. `BETA\_FOLDER\_FMT` pins

&#x20; names to one decimal place.

\- `resolve\_fixed\_variables()` treats a missing layer as \*\*fatal\*\* (unlike the

&#x20; combo runner's skip-and-report), because silently fitting a 4-variable model

&#x20; somewhere in the sweep would invalidate every comparison.

\- `count\_lambdas\_parameters()` parses the `.lambdas` file, dropping the four

&#x20; MaxEnt metadata lines (`linearPredictorNormalizer`, `densityNormalizer`,

&#x20; `numBackgroundPoints`, `entropy`) and any zero-weight feature, to recover the

&#x20; non-zero parameter count — the `k` that AIC/AICc/BIC requires.

\- `harvest\_maxent\_results()` pulls sample size, regularized/unregularized

&#x20; training gain, training and test AUC, and feature count from

&#x20; `maxentResults.csv`, matching each metric against several candidate header

&#x20; names (they vary by MaxEnt version and test-split presence) and returning

&#x20; blanks instead of raising.

\- Defaults `--random-test-percentage` to \*\*0\*\*, documenting that withholding

&#x20; points changes both the likelihood and the parameter count, making AICc scores

&#x20; incomparable across betas. Also warns that `--random-seed` would give each

&#x20; beta a different background sample.

\- Outputs `beta\_runs.csv`, `run\_summary.csv`, and the decision-oriented

&#x20; `beta\_comparison.csv`, plus a console table of beta / params / gain / AUC.

&#x20; `print\_comparison\_table()` includes a built-in sanity check: if every beta

&#x20; yields an identical parameter count, `betamultiplier=` is not reaching MaxEnt

&#x20; and the sweep is just N copies of the same model.



\---



\## Conventions Across the Repo



\- \*\*NoData is −9999 everywhere\*\* — rasters, `--nodata`, and the R `NAflag` must

&#x20; agree or MaxEnt will treat sentinel values as real data.

\- \*\*Geographic CRS assumed\*\*: EPSG:4326, extent lon `\[-100, -55]`, lat

&#x20; `\[9, 37]`.

\- Configuration lives in a clearly delimited settings block at the top of each

&#x20; file; the Python runners additionally expose full `argparse` interfaces so

&#x20; they can run with zero arguments or be fully overridden.

\- Heavy inline commenting, progress logging to stdout, and a summary CSV or

&#x20; tally at the end of every batch operation.

\- Errors are caught per item so one bad file never kills a run, with the

&#x20; deliberate exception of JVM startup failures and missing fixed variables.



\---



\## Known Rough Edges



\- `trim\_rasters\_in\_polygon.py` sets `IN\_DIR` to a `.tif` \*\*file\*\* rather than a

&#x20; directory, so `collect\_rasters()` will find nothing as shipped.

\- `combos.R` is an un-wrapped function body, mixes its stated (0.8) and applied

&#x20; (0.9) default thresholds, and relies on Windows-only `choose.dir()`.

\- `Generating correlations.R` prompts the user to pick a folder but then uses a

&#x20; hard-coded path, and its `list.files` pattern `"\*.ASC"` is case-sensitive and

&#x20; not a proper regex (`"\\\\.asc$"` with `ignore.case = TRUE` would be correct).

\- `correlation.R` requires `permutation\_analyzer.R` to be sourced first and uses

&#x20; the older `raster`/`sp` stack rather than `terra`.

\- The two MaxEnt runners duplicate \~400 lines of shared infrastructure (JVM

&#x20; preflight, settings dataclass, layer toggling, run execution) that could be

&#x20; factored into a common module.

