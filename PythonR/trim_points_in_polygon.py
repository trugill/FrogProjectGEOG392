#!/usr/bin/env python3
"""
Trim species occurrence points to a rectangular extent.

Input CSV format: species, latitude, longitude
Extent: 100W 37N (NW corner) to 55W 9N (SE corner)
        -> longitude in [-100, -55], latitude in [9, 37]

Requires: pip install pandas
"""

from pathlib import Path
import pandas as pd

# ---- SETTINGS ----------------------------------------------
IN_CSV  = Path("D:/Frogs/CSV/points.trimmed.csv")
OUT_CSV = Path("D:/Frogs/CSV/points.refined.csv")

XMIN, XMAX = -100.0, -55.0   # longitude (west, east)
YMIN, YMAX =    9.0,  37.0   # latitude  (south, north)

HAS_HEADER   = True          # False if the CSV has no header row
DROP_INVALID = True          # drop rows with missing/unparseable coords
DROP_DUPES   = False         # True to drop exact species+lat+lon duplicates
# ------------------------------------------------------------


def load_points(path: Path, has_header: bool) -> pd.DataFrame:
    if has_header:
        df = pd.read_csv(path)
        cols = {c.strip().lower(): c for c in df.columns}

        def pick(*candidates):
            for cand in candidates:
                if cand in cols:
                    return cols[cand]
            return None

        sp  = pick("species", "scientificname", "scientific_name", "taxon", "name")
        lat = pick("latitude", "lat", "decimallatitude", "decimal_latitude", "y")
        lon = pick("longitude", "lon", "long", "decimallongitude",
                   "decimal_longitude", "x")

        # Fall back to positional order: species, latitude, longitude
        if sp is None or lat is None or lon is None:
            if df.shape[1] < 3:
                raise ValueError("CSV needs at least 3 columns.")
            sp, lat, lon = df.columns[0], df.columns[1], df.columns[2]
            print("  ! Column names not recognised; using first three columns "
                  "as species, latitude, longitude.")

        df = df.rename(columns={sp: "species", lat: "latitude", lon: "longitude"})
    else:
        df = pd.read_csv(path, header=None,
                         names=["species", "latitude", "longitude"])
    return df


def main():
    df = load_points(IN_CSV, HAS_HEADER)
    n_start = len(df)

    # Coerce coordinates to numeric
    df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    n_bad = int(df[["latitude", "longitude"]].isna().any(axis=1).sum())
    if DROP_INVALID and n_bad:
        df = df.dropna(subset=["latitude", "longitude"])

    # Sanity check: flag possible lat/lon swap
    if (df["latitude"].abs() > 90).any():
        print("  ! Some 'latitude' values exceed ±90 — are lat/lon swapped?")

    inside = (
        df["longitude"].between(XMIN, XMAX) &
        df["latitude"].between(YMIN, YMAX)
    )
    kept = df[inside].copy()

    if DROP_DUPES:
        before = len(kept)
        kept = kept.drop_duplicates(subset=["species", "latitude", "longitude"])
        print(f"  Removed {before - len(kept)} duplicate record(s).")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(OUT_CSV, index=False)

    # ---- Report ----
    print(f"\nExtent: lon [{XMIN}, {XMAX}], lat [{YMIN}, {YMAX}]")
    print(f"Read:    {n_start:,} records")
    if n_bad:
        print(f"Invalid: {n_bad:,} records with bad/missing coordinates")
    print(f"Kept:    {len(kept):,} inside extent")
    print(f"Dropped: {n_start - n_bad - len(kept):,} outside extent")
    print(f"Written: {OUT_CSV}")

    if len(kept):
        print("\nRecords per species (top 20):")
        print(kept["species"].value_counts().head(20).to_string())


if __name__ == "__main__":
    main()