#!/usr/bin/env python3
"""
Isolate and convert a single problem raster: wc2.1_2.5m_bio_1.tif

Runs staged diagnostics, then tries four independent conversion strategies
in order of increasing robustness, stopping at the first that works.

Requires: pip install rasterio numpy
"""

from pathlib import Path
import os
import traceback
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, from_bounds

# ---- SETTINGS ----------------------------------------------
SRC      = Path("D:/Frogs/TIF/wc2.1_2.5m_bio/wc2.1_2.5m_bio_1.tif")
OUT_DIR  = Path("D:/Frogs/ASC/")
OUT_NAME = "wc2.1_2.5m_bio_1.asc"

XMIN, XMAX = -100.0, -55.0
YMIN, YMAX =    9.0,  37.0
CELLSIZE   = 1.0 / 24.0        # 2.5 arc-minutes, exact
NA_VALUE   = -9999
DECIMALS   = 3

# WorldClim stores NoData as a huge negative float; anything at or below
# this threshold is treated as NoData even if the header disagrees.
SENTINEL_FLOOR = -1e30
# ------------------------------------------------------------

NCOLS = int(round((XMAX - XMIN) / CELLSIZE))
NROWS = int(round((YMAX - YMIN) / CELLSIZE))
DST_TRANSFORM = from_origin(XMIN, YMAX, CELLSIZE, CELLSIZE)


def rule(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def clean(arr):
    """Normalise NaN, inf and sentinel values to NA_VALUE."""
    arr = np.asarray(arr, dtype="float32")
    bad = ~np.isfinite(arr) | (arr <= SENTINEL_FLOOR)
    arr = np.where(bad, NA_VALUE, arr)
    if DECIMALS is not None:
        arr = np.round(arr, DECIMALS)
    return arr


def write_asc(path, arr):
    nrows, ncols = arr.shape
    dec = 6 if DECIMALS is None else DECIMALS
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n") as fh:
        fh.write(f"ncols         {ncols}\n")
        fh.write(f"nrows         {nrows}\n")
        fh.write(f"xllcorner     {XMIN}\n")
        fh.write(f"yllcorner     {YMIN}\n")
        fh.write(f"cellsize      {CELLSIZE!r}\n")
        fh.write(f"NODATA_value  {NA_VALUE}\n")
        np.savetxt(fh, arr, fmt=f"%.{dec}f", delimiter=" ")
    return path


# ============================================================
# STAGE 1 — filesystem
# ============================================================
def stage1_filesystem():
    rule("STAGE 1 — Filesystem")
    if not SRC.exists():
        print(f"  FATAL: path does not exist: {SRC}")
        siblings = sorted(SRC.parent.glob("wc2.1*"))
        if siblings:
            print("  Similar names in that folder:")
            for s in siblings:
                print(f"    {s.name}")
        return False

    size = SRC.stat().st_size
    print(f"  Path     : {SRC}")
    print(f"  Size     : {size:,} bytes ({size / 1e6:.1f} MB)")
    print(f"  Readable : {os.access(SRC, os.R_OK)}")

    if size < 1_000_000:
        print("  WARNING: suspiciously small for a global 2.5m layer —")
        print("           likely a truncated or failed extraction.")
    try:
        with open(SRC, "rb") as fh:
            magic = fh.read(4)
        print(f"  Magic    : {magic!r} "
              f"({'valid TIFF' if magic[:2] in (b'II', b'MM') else 'NOT a TIFF'})")
    except Exception as e:
        print(f"  Could not read header bytes: {e}")
        return False
    return True


# ============================================================
# STAGE 2 — metadata
# ============================================================
def stage2_metadata():
    rule("STAGE 2 — Raster metadata")
    try:
        with rasterio.open(SRC) as src:
            xres, yres = abs(src.transform.a), abs(src.transform.e)
            print(f"  Driver      : {src.driver}")
            print(f"  Size        : {src.width} x {src.height}, {src.count} band(s)")
            print(f"  Dtype       : {src.dtypes[0]}")
            print(f"  CRS         : {src.crs}")
            print(f"  Resolution  : {xres:.10g} x {yres:.10g}")
            print(f"  Bounds      : {tuple(round(b, 5) for b in src.bounds)}")
            print(f"  NoData      : {src.nodata}")
            print(f"  Tiled       : {src.is_tiled}  blocks={src.block_shapes}")
            print(f"  Compression : {src.compression}")
            print(f"  Flipped     : {src.transform.e > 0}")

            exp = 1.0 / 24.0
            if not np.isclose(xres, exp, rtol=1e-4):
                print(f"  NOTE: resolution differs from expected {exp:.10g};"
                      f" adjust CELLSIZE to {xres:.10g}")
            sb = src.bounds
            overlaps = not (XMIN >= sb.right or XMAX <= sb.left or
                            YMIN >= sb.top or YMAX <= sb.bottom)
            print(f"  Overlaps target extent: {overlaps}")
            if not overlaps and sb.left >= 0 and sb.right > 180:
                print("  -> Looks like a 0-360 longitude grid.")
            return True
    except Exception as e:
        print(f"  FATAL: cannot open: {type(e).__name__}: {e}")
        print("\n" + traceback.format_exc())
        return False


# ============================================================
# STAGE 3 — integrity probe
# ============================================================
def stage3_integrity():
    rule("STAGE 3 — Integrity probe")
    try:
        with rasterio.open(SRC) as src:
            # Reading the final block catches most truncated downloads
            bh, bw = src.block_shapes[0]
            win = Window(max(0, src.width - bw), max(0, src.height - bh),
                         min(bw, src.width), min(bh, src.height))
            _ = src.read(1, window=win)
            print(f"  Last block read OK ({win.width} x {win.height})")

            mid = src.read(1, window=Window(src.width // 2, src.height // 2,
                                            min(64, src.width), min(64, src.height)))
            finite = mid[np.isfinite(mid) & (mid > SENTINEL_FLOOR)]
            if finite.size:
                print(f"  Sample values: min={finite.min():.3f} "
                      f"max={finite.max():.3f} mean={finite.mean():.3f}")
            else:
                print("  Sample block contained no valid values (may be ocean)")
            return True
    except Exception as e:
        print(f"  Integrity read FAILED: {type(e).__name__}: {e}")
        print("  -> Strongly suggests a truncated/corrupt file. Re-download.")
        return False


# ============================================================
# STAGE 4 — conversion strategies
# ============================================================
def strategy_a():
    """Direct boundless window read — fastest, assumes matching CRS."""
    with rasterio.open(SRC) as src:
        win = from_bounds(XMIN, YMIN, XMAX, YMAX, transform=src.transform)
        win = win.round_offsets(op="floor").round_lengths(op="ceil")
        arr = src.read(1, window=win, boundless=True,
                       fill_value=NA_VALUE,
                       out_shape=(NROWS, NCOLS),
                       resampling=Resampling.nearest)
    return clean(arr)


def strategy_b():
    """WarpedVRT onto the exact target grid — handles CRS and alignment."""
    with rasterio.open(SRC) as src:
        opts = dict(crs=CRS.from_epsg(4326), transform=DST_TRANSFORM,
                    width=NCOLS, height=NROWS,
                    resampling=Resampling.bilinear, nodata=NA_VALUE)
        if src.crs is None:
            opts["src_crs"] = CRS.from_epsg(4326)
        if src.nodata is not None:
            opts["src_nodata"] = src.nodata
        with WarpedVRT(src, **opts) as vrt:
            arr = vrt.read(1)
    return clean(arr)


def strategy_c():
    """Mask-aware VRT read: ignores a wrong/absent nodata header."""
    with rasterio.open(SRC) as src:
        opts = dict(crs=CRS.from_epsg(4326), transform=DST_TRANSFORM,
                    width=NCOLS, height=NROWS,
                    resampling=Resampling.nearest,
                    src_nodata=None, nodata=None)
        if src.crs is None:
            opts["src_crs"] = CRS.from_epsg(4326)
        with WarpedVRT(src, **opts) as vrt:
            arr = vrt.read(1)
    return clean(arr)


def strategy_d():
    """Row-by-row streaming — lowest memory, survives flaky I/O."""
    out = np.full((NROWS, NCOLS), NA_VALUE, dtype="float32")
    with rasterio.open(SRC) as src:
        inv = ~src.transform
        for r in range(NROWS):
            y = YMAX - (r + 0.5) * CELLSIZE
            xs = XMIN + (np.arange(NCOLS) + 0.5) * CELLSIZE
            cols, rows = inv * (xs, np.full(NCOLS, y))
            cols = np.floor(cols).astype(int)
            rows = np.floor(rows).astype(int)
            keep = ((cols >= 0) & (cols < src.width) &
                    (rows >= 0) & (rows < src.height))
            if not keep.any():
                continue
            r0, r1 = rows[keep].min(), rows[keep].max() + 1
            c0, c1 = cols[keep].min(), cols[keep].max() + 1
            block = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
            out[r, keep] = block[rows[keep] - r0, cols[keep] - c0]
    return clean(out)


def stage4_convert():
    rule("STAGE 4 — Conversion")
    print(f"  Target grid: {NCOLS} x {NROWS} @ {CELLSIZE!r} deg\n")

    for name, fn in (("A  direct window read",      strategy_a),
                     ("B  WarpedVRT (bilinear)",    strategy_b),
                     ("C  mask-aware VRT",          strategy_c),
                     ("D  streaming row-by-row",    strategy_d)):
        print(f"  Trying {name} ...")
        try:
            arr = fn()
            if arr.shape != (NROWS, NCOLS):
                print(f"    shape mismatch {arr.shape}; discarded")
                continue
            valid = int((arr != NA_VALUE).sum())
            pct = valid / arr.size
            print(f"    read OK — {valid:,} valid cells ({pct:.1%})")
            if valid == 0:
                print("    all NoData; trying next strategy")
                continue
            good = arr[arr != NA_VALUE]
            print(f"    range: {good.min():.3f} to {good.max():.3f} "
                  f"(mean {good.mean():.3f})")
            if not (-40 < good.min() and good.max() < 60):
                print("    NOTE: values outside a plausible °C range for bio_1 —")
                print("          check whether this layer is scaled by 10.")
            out = write_asc(OUT_DIR / OUT_NAME, arr)
            print(f"\n  SUCCESS via strategy {name[0]}")
            print(f"  Written: {out}  ({out.stat().st_size / 1e6:.1f} MB)")
            return True
        except Exception as e:
            print(f"    failed: {type(e).__name__}: {e}")

    print("\n  All four strategies failed.")
    return False


# ============================================================
def main():
    print(f"Isolating: {SRC.name}")
    if not stage1_filesystem():
        return
    if not stage2_metadata():
        print("\nFile cannot be opened — re-download from WorldClim and retry.")
        return
    stage3_integrity()
    if stage4_convert():
        rule("VERIFY")
        p = OUT_DIR / OUT_NAME
        with open(p) as fh:
            for _ in range(6):
                print("  " + fh.readline().rstrip())
        print("\n  Header should match the other .asc files in the folder.")


if __name__ == "__main__":
    main()