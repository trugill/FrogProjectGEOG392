#!/usr/bin/env python3
"""
Crop/warp every GeoTIFF in a directory to a fixed extent and write ESRI ASCII.

Extent: 100W 37N to 55W 9N  ->  lon [-100, -55], lat [9, 37]

Every readable raster produces an .asc on an identical grid.
Files are only skipped if they cannot be opened at all, and every
skip is reported with a reason at the end.

Requires: pip install rasterio numpy
"""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

# ---- SETTINGS ----------------------------------------------
IN_DIR  = Path("D:/Frogs/TIF/wc2.1_2.5m_bio/wc2.1_2.5m_bio_1.tif")
OUT_DIR = Path("D:/Frogs/ASC/")

XMIN, XMAX = -100.0, -55.0
YMIN, YMAX =    9.0,  37.0
DST_CRS    = CRS.from_epsg(4326)

# Output cell size in degrees. None = infer from the first readable raster.
# Common values: 0.0083333333 (30 arc-sec), 0.0416666667 (2.5 min), 0.5
CELLSIZE = None

NA_VALUE   = -9999
DECIMALS   = 3              # None to keep full precision
RESAMPLING = Resampling.bilinear   # use .nearest for categorical/integer layers
ASSUME_CRS = CRS.from_epsg(4326)   # applied when a raster has no CRS
RECURSIVE  = True
EXTENSIONS = {".tif", ".tiff", ".gtif", ".gtiff"}
ALL_BANDS  = True           # False = first band only
# ------------------------------------------------------------


def collect_rasters(in_dir: Path, recursive: bool):
    """Case-insensitive, duplicate-free file discovery."""
    it = in_dir.rglob("*") if recursive else in_dir.iterdir()
    out = {}
    for p in it:
        if p.is_file() and p.suffix.lower() in EXTENSIONS:
            out[p.resolve()] = p          # resolve() dedupes case variants
    return sorted(out.values())


def infer_cellsize(paths):
    """Smallest native cell size among readable rasters, in degrees."""
    best = None
    for p in paths:
        try:
            with rasterio.open(p) as src:
                xres, yres = abs(src.transform.a), abs(src.transform.e)
                if src.crs and not src.crs.is_geographic:
                    # Convert a projected resolution to approximate degrees
                    l, b, r, t = transform_bounds(src.crs, DST_CRS, *src.bounds)
                    xres = abs(r - l) / src.width
                    yres = abs(t - b) / src.height
                cs = min(xres, yres)
                if cs > 0 and (best is None or cs < best):
                    best = cs
        except Exception:
            continue
    return best


def build_target_grid(cellsize):
    width  = int(np.ceil((XMAX - XMIN) / cellsize))
    height = int(np.ceil((YMAX - YMIN) / cellsize))
    transform = from_origin(XMIN, YMAX, cellsize, cellsize)
    return transform, width, height, cellsize


def normalise_longitude(src):
    """
    Detect 0-360 longitude rasters and return a shifted copy of the target
    bounds, or None if no shift is needed.
    """
    if src.crs is None or not src.crs.is_geographic:
        return False
    return src.bounds.left >= -1e-6 and src.bounds.right > 180.0


def write_asc(path: Path, arr: np.ndarray, cellsize: float):
    """Write an ESRI ASCII grid directly — no driver capability surprises."""
    nrows, ncols = arr.shape
    dec = 6 if DECIMALS is None else DECIMALS
    with open(path, "w", newline="\n") as fh:
        fh.write(f"ncols         {ncols}\n")
        fh.write(f"nrows         {nrows}\n")
        fh.write(f"xllcorner     {XMIN!r}\n")
        fh.write(f"yllcorner     {YMIN!r}\n")
        fh.write(f"cellsize      {cellsize!r}\n")
        fh.write(f"NODATA_value  {NA_VALUE}\n")
        np.savetxt(fh, arr, fmt=f"%.{dec}f", delimiter=" ")


def process(src_path: Path, grid, out_dir: Path):
    transform, width, height, cellsize = grid

    with rasterio.open(src_path) as src:
        src_crs = src.crs or ASSUME_CRS
        note = None
        if src.crs is None:
            note = f"no CRS; assumed {ASSUME_CRS.to_string()}"

        # Handle 0-360 longitude grids by shifting the source CRS definition
        if normalise_longitude(src):
            src_crs = CRS.from_proj4(
                src_crs.to_proj4() + " +lon_wrap=180"
                if "lon_wrap" not in src_crs.to_proj4() else src_crs.to_proj4()
            )
            note = (note + "; " if note else "") + "0-360 longitude remapped"

        vrt_opts = dict(
            src_crs=src_crs,
            crs=DST_CRS,
            transform=transform,
            width=width,
            height=height,
            resampling=RESAMPLING,
            nodata=NA_VALUE,
        )
        if src.nodata is not None:
            vrt_opts["src_nodata"] = src.nodata

        with WarpedVRT(src, **vrt_opts) as vrt:
            bands = range(1, src.count + 1) if ALL_BANDS else [1]
            written, coverage = [], 0.0

            for bi in bands:
                arr = vrt.read(bi).astype("float32")
                arr = np.where(np.isnan(arr), NA_VALUE, arr)
                if DECIMALS is not None:
                    arr = np.round(arr, DECIMALS)

                valid = int((arr != NA_VALUE).sum())
                coverage = max(coverage, valid / arr.size)

                suffix = f"_band{bi}" if (src.count > 1 and ALL_BANDS) else ""
                out_path = out_dir / f"{src_path.stem}{suffix}.asc"
                write_asc(out_path, arr, cellsize)
                written.append(out_path.name)

    return written, coverage, note


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = collect_rasters(IN_DIR, RECURSIVE)
    if not paths:
        raise SystemExit(f"No raster files found in: {IN_DIR}")

    cellsize = CELLSIZE or infer_cellsize(paths)
    if not cellsize:
        raise SystemExit("Could not determine a cell size — set CELLSIZE manually.")

    grid = build_target_grid(cellsize)
    transform, width, height, cellsize = grid

    print(f"Input:    {IN_DIR}  ({len(paths)} candidate file(s))")
    print(f"Extent:   lon [{XMIN}, {XMAX}], lat [{YMIN}, {YMAX}]")
    print(f"Grid:     {width} x {height} cells @ {cellsize} deg")
    print(f"Output:   {OUT_DIR}\n")

    ok, empty, failed, total_written = [], [], [], 0

    for p in paths:
        try:
            written, coverage, note = process(p, grid, OUT_DIR)
            total_written += len(written)
            tag = f"  [{note}]" if note else ""
            if coverage == 0.0:
                empty.append(p.name)
                print(f"  EMPTY  {p.name} -> written, but 0% data in extent{tag}")
            else:
                ok.append(p.name)
                print(f"  OK     {p.name} -> {len(written)} file(s), "
                      f"{coverage:.1%} data coverage{tag}")
        except Exception as e:
            failed.append((p.name, f"{type(e).__name__}: {e}"))
            print(f"  FAIL   {p.name} -> {type(e).__name__}: {e}")

    print("\n" + "-" * 60)
    print(f"Candidates : {len(paths)}")
    print(f"Converted  : {len(ok) + len(empty)}  ({total_written} .asc written)")
    print(f"  with data: {len(ok)}")
    print(f"  all NoData: {len(empty)}")
    print(f"Unreadable : {len(failed)}")

    if empty:
        print("\nAll-NoData outputs (check their native extent/CRS):")
        for n in empty:
            print(f"  - {n}")
    if failed:
        print("\nFailures:")
        for n, reason in failed:
            print(f"  - {n}: {reason}")

    print("\nAll outputs share one grid, so they are MaxEnt-ready.")


if __name__ == "__main__":
    main()