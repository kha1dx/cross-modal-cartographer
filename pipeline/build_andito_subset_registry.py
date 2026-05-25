#!/usr/bin/env python3
"""
build_andito_subset_registry.py
-------------------------------
Build dataset/andito_landmarks_registry.csv from the andito/google-landmarks
Egyptian subset that was streamed into a flat directory by Colab.

Inputs
    egyptian_landmarks_data/checkpoints/auto_checkpoint_1257991.json
        list of {name, image_path, metadata: [{user, assistant}, ...]}
    egyptian_landmarks_data/images/<rownum>_<sanitized_name>.jpg
        flat directory of 4,224 JPGs

Output
    ../dataset/andito_landmarks_registry.csv
        same 9-column schema as dataset/landmarks_registry.csv
        one row per unique landmark name (canonical_name = sanitized name)

Run from V0/:
    python build_andito_subset_registry.py
"""

import csv
import json
import os
import re
from collections import Counter, defaultdict

from build_registry import (
    classify_type,
    classify_era,
    derive_style,
    CITY_LOOKUP,
    CITY_TO_REGION,
    CITY_COORDS,
    CITY_ALIASES,
    NOT_IN_EGYPT,
)

HERE        = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT  = os.path.join(HERE, "egyptian_landmarks_data", "checkpoints",
                           "auto_checkpoint_1257991.json")
IMAGES_DIR  = os.path.join(HERE, "egyptian_landmarks_data", "images")
OUTPUT_CSV  = os.path.join(HERE, "..", "dataset", "andito_landmarks_registry.csv")

LOCATION_RE = re.compile(r"located in\s+(.+?)\.?\s*$", re.IGNORECASE)


def sanitize(name: str) -> str:
    """Match the same sanitization the Colab loop used when saving image filenames:
       ''.join([c if c.isalnum() else '_' for c in name])
    """
    return "".join(c if c.isalnum() else "_" for c in name)


def parse_location_parts(qa_metadata: list) -> list:
    """Pull the location parts list from a Q&A 'It is located in X, Y, Z.' turn."""
    for turn in qa_metadata or []:
        ans = (turn.get("assistant") or "").strip()
        if "located in" not in ans.lower():
            continue
        m = LOCATION_RE.search(ans)
        if not m:
            continue
        return [p.strip() for p in m.group(1).rstrip(".").split(",") if p.strip()]
    return []


def resolve_city(canonical_name: str, qa_entries: list) -> str:
    """Pick a city name that's in CITY_COORDS when possible.

    Priority:
      1. Exact match in CITY_LOOKUP by canonical_name
      2. Q&A location parts — try parts[0], then parts[1], match against CITY_COORDS keys
      3. Most common parts[0] across this landmark's Q&A entries (verbatim)
      4. Fallback: 'Cairo'
    """
    # 1. CITY_LOOKUP exact match
    if canonical_name in CITY_LOOKUP:
        return CITY_LOOKUP[canonical_name]

    # Collect location parts from every Q&A entry for this landmark
    all_parts = [parse_location_parts(qa) for qa in qa_entries]
    all_parts = [p for p in all_parts if p]

    if all_parts:
        # 2. Try matching parts[0] then parts[1] against known CITY_COORDS or CITY_ALIASES
        coord_keys_lower = {k.lower(): k for k in CITY_COORDS}
        for parts in all_parts:
            for candidate in parts[:2]:
                cand_lower = candidate.lower()
                hit = coord_keys_lower.get(cand_lower)
                if hit:
                    return hit
                # Check aliases
                alias_hit = CITY_ALIASES.get(cand_lower)
                if alias_hit:
                    return alias_hit

        # 3. Most common parts[0] verbatim — even if not in CITY_COORDS
        first_parts = [parts[0] for parts in all_parts if parts]
        if first_parts:
            return Counter(first_parts).most_common(1)[0][0]

    # 4. Default
    return "Cairo"


def main() -> None:
    if not os.path.exists(CHECKPOINT):
        raise SystemExit(f"checkpoint not found: {CHECKPOINT}")
    if not os.path.isdir(IMAGES_DIR):
        raise SystemExit(f"images dir not found: {IMAGES_DIR}")

    with open(CHECKPOINT, encoding="utf-8") as f:
        entries = json.load(f)

    # Group entries by canonical landmark name (sanitized, matches filename pattern)
    by_canonical: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        canonical = sanitize(e["name"])
        by_canonical[canonical].append(e)

    print(f"Loaded {len(entries)} image-records across {len(by_canonical)} unique landmarks.")

    rows = []
    missing_city_in_coords: list[str] = []

    for i, (canonical, qa_entries_for_landmark) in enumerate(
            sorted(by_canonical.items()), start=1):

        original_name = qa_entries_for_landmark[0]["name"]

        # Skip landmarks that are not in Egypt
        if canonical in NOT_IN_EGYPT:
            print(f"  SKIPPED (not in Egypt): {canonical}")
            continue

        # classify_type/classify_era check explicit overrides first,
        # then fall back to substring rules.
        landmark_type = classify_type(canonical)
        if landmark_type == "Historical Site":
            # Try with original (unsanitized) name as fallback
            alt = classify_type(original_name)
            if alt != "Historical Site":
                landmark_type = alt

        era = classify_era(canonical)
        if era == "Islamic":
            alt = classify_era(original_name)
            if alt != "Islamic":
                era = alt

        qa_lists = [e.get("metadata", []) for e in qa_entries_for_landmark]
        city = resolve_city(canonical, qa_lists)

        if city not in CITY_COORDS:
            missing_city_in_coords.append(f"{canonical} -> {city!r}")

        region   = CITY_TO_REGION.get(city, "Greater Cairo")
        lat, lon = CITY_COORDS.get(city, (30.0444, 31.2357))
        style    = derive_style(landmark_type, era)

        rows.append({
            "landmark_id":         i,
            "canonical_name":      canonical,
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
        print(f"  {t:25s} {n:4d}")
    print("\nEra distribution:")
    for e, n in era_dist.most_common():
        print(f"  {e:25s} {n:4d}")
    print("\nTop 15 cities:")
    for c, n in city_dist.most_common(15):
        print(f"  {c:25s} {n:4d}")

    if missing_city_in_coords:
        print(f"\nWARNING: {len(missing_city_in_coords)} landmarks resolved to a city "
              f"not in CITY_COORDS — coordinates default to Cairo.")
        for line in missing_city_in_coords[:20]:
            print(f"  {line}")
        if len(missing_city_in_coords) > 20:
            print(f"  ... and {len(missing_city_in_coords) - 20} more")


if __name__ == "__main__":
    main()
