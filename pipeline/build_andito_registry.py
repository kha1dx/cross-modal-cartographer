#!/usr/bin/env python3
"""
build_andito_registry.py
------------------------
Builds dataset_andito/landmarks_registry.csv from the images and JSONL records
produced by fetch_andito_egypt.py.

Schema mirrors dataset/landmarks_registry.csv exactly so phase_andito_build.ipynb
can reuse the same metadata-join logic as phase 2 of the original pipeline.

Run from V0/:
    python build_andito_registry.py
"""

import os
import csv
import json
from collections import Counter

from build_registry import (
    classify,
    derive_style,
    TYPE_RULES,
    ERA_RULES,
    CITY_TO_REGION,
    CITY_COORDS,
)

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(HERE, "dataset_andito", "images")
RECORDS_JSONL = os.path.join(HERE, "dataset_andito", "andito_raw_records.jsonl")
OUTPUT_CSV = os.path.join(HERE, "dataset_andito", "landmarks_registry.csv")


def load_records_by_landmark(jsonl_path: str) -> dict:
    """Group JSONL records by sanitized landmark folder name."""
    grouped: dict[str, list[dict]] = {}
    if not os.path.exists(jsonl_path):
        print(f"WARNING: {jsonl_path} not found. City lookup will fall back to Cairo.")
        return grouped
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            folder = rec.get("folder") or rec.get("landmark_folder")
            if not folder:
                continue
            grouped.setdefault(folder, []).append(rec)
    return grouped


def most_common(values, fallback):
    values = [v for v in values if v]
    if not values:
        return fallback
    return Counter(values).most_common(1)[0][0]


def main():
    if not os.path.isdir(IMAGES_DIR):
        raise SystemExit(f"images dir not found: {IMAGES_DIR}\nRun fetch_andito_egypt.py first.")

    records = load_records_by_landmark(RECORDS_JSONL)

    folders = sorted(
        f for f in os.listdir(IMAGES_DIR)
        if os.path.isdir(os.path.join(IMAGES_DIR, f))
    )

    rows = []
    missing_city_coords: list[str] = []

    for i, folder in enumerate(folders, start=1):
        # Type / era / style — rule-based on canonical folder name (same logic as build_registry.py)
        landmark_type = classify(folder, TYPE_RULES) or "Historical Site"
        era           = classify(folder, ERA_RULES)  or "Islamic"
        style         = derive_style(landmark_type, era)

        # City / region — pulled from the parsed Q&A text in the JSONL records
        recs = records.get(folder, [])
        city_text   = most_common([r.get("city_text")   for r in recs], None)
        region_text = most_common([r.get("region_text") for r in recs], None)

        # Use the parsed city if we have CITY_COORDS for it; else fall back to Cairo.
        if city_text and city_text in CITY_COORDS:
            city = city_text
        elif city_text:
            # Try a relaxed match against known city names (case-insensitive substring)
            relaxed = next((c for c in CITY_COORDS if c.lower() == city_text.lower()), None)
            if relaxed:
                city = relaxed
            else:
                missing_city_coords.append(f"{folder}  ->  {city_text!r}")
                city = "Cairo"
        else:
            city = "Cairo"

        region = CITY_TO_REGION.get(city) or region_text or "Greater Cairo"
        lat, lon = CITY_COORDS.get(city, (30.0444, 31.2357))

        rows.append({
            "landmark_id":         i,
            "canonical_name":      folder,
            "landmark_type":       landmark_type,
            "historical_era":      era,
            "geographic_region":   region,
            "city":                city,
            "architectural_style": style,
            "coordinates_lat":     lat,
            "coordinates_lon":     lon,
        })

    fieldnames = [
        "landmark_id", "canonical_name", "landmark_type", "historical_era",
        "geographic_region", "city", "architectural_style",
        "coordinates_lat", "coordinates_lon",
    ]
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written {len(rows)} landmarks -> {OUTPUT_CSV}")

    type_dist = Counter(r["landmark_type"] for r in rows)
    era_dist  = Counter(r["historical_era"] for r in rows)
    city_dist = Counter(r["city"] for r in rows)
    print("\nType distribution:")
    for t, n in type_dist.most_common():
        print(f"  {t:25s} {n:3d}")
    print("\nEra distribution:")
    for e, n in era_dist.most_common():
        print(f"  {e:25s} {n:3d}")
    print("\nTop cities:")
    for c, n in city_dist.most_common(15):
        print(f"  {c:25s} {n:3d}")

    if missing_city_coords:
        print(f"\nWARNING: {len(missing_city_coords)} landmarks had a parsed city not in CITY_COORDS.")
        print("Falling back to Cairo coords. Consider adding to CITY_COORDS in build_registry.py:")
        for line in missing_city_coords[:30]:
            print(f"  {line}")
        if len(missing_city_coords) > 30:
            print(f"  ... and {len(missing_city_coords) - 30} more")


if __name__ == "__main__":
    main()
