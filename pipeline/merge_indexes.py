#!/usr/bin/env python3
"""
merge_indexes.py
----------------
Merges the main FAISS index and the andito (Google Landmarks Egypt) FAISS index
into a single unified index with corrected metadata.

Steps:
  1. Load both indexes and metadata pickles
  2. Re-classify ALL metadata using updated build_registry functions
  3. Filter out NOT_IN_EGYPT landmarks from the andito set
  4. Deduplicate using GLD_ALIASES (andito names → main canonical names)
  5. Build new unified IndexFlatIP(512) + merged metadata pickle

Run from V0/:
    python merge_indexes.py
"""

import os
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np

# Add V0 to path for build_registry imports
sys.path.insert(0, str(Path(__file__).parent))

from build_registry import (
    CITY_LOOKUP, CITY_COORDS, CITY_TO_REGION,
    NOT_IN_EGYPT, GLD_ALIASES,
    classify_type, classify_era, derive_style,
)

EMB_DIR = Path(__file__).parent / "embeddings"

# Input files — use the _clean variants (Phase 0 photo filter applied, diagrams removed)
MAIN_INDEX   = EMB_DIR / "faiss_index_clean.bin"
MAIN_META    = EMB_DIR / "faiss_metadata_clean.pkl"
ANDITO_INDEX = EMB_DIR / "andito_faiss_index_clean.bin"
ANDITO_META  = EMB_DIR / "andito_faiss_metadata_clean.pkl"

# Output files
OUT_INDEX = EMB_DIR / "faiss_index_unified.bin"
OUT_META  = EMB_DIR / "faiss_metadata_unified.pkl"


def reclassify_entry(m: dict) -> dict:
    """Re-classify a single metadata entry using updated build_registry functions.
    Overwrites type, era, city, region, style, and coordinates field-by-field."""
    name = m["landmark_name"]
    ltype = classify_type(name)
    era = classify_era(name)
    city = CITY_LOOKUP.get(name, m.get("city", "Cairo"))
    region = CITY_TO_REGION.get(city, "Greater Cairo")
    lat, lon = CITY_COORDS.get(city, (30.0444, 31.2357))
    style = derive_style(ltype, era)

    return {
        "image_path":         m["image_path"],       # keep original
        "landmark_name":      name,                  # keep original
        "landmark_type":      ltype,                 # re-classified
        "historical_era":     era,                   # re-classified
        "geographic_region":  region,                # re-derived
        "city":               city,                  # re-looked-up
        "architectural_style": style,                # re-derived
        "coordinates_lat":    str(lat),              # re-derived
        "coordinates_lon":    str(lon),              # re-derived
    }


def main():
    # ── Load main index ──────────────────────────────────────────────────────
    print("Loading main index...")
    main_index = faiss.read_index(str(MAIN_INDEX))
    with open(MAIN_META, "rb") as f:
        main_meta = pickle.load(f)

    print(f"  Main: {main_index.ntotal} vectors, {len(main_meta)} metadata entries")
    assert main_index.ntotal == len(main_meta), "Main index/metadata count mismatch!"

    # ── Load andito index ────────────────────────────────────────────────────
    print("Loading andito index...")
    andito_index = faiss.read_index(str(ANDITO_INDEX))
    with open(ANDITO_META, "rb") as f:
        andito_meta = pickle.load(f)

    print(f"  Andito: {andito_index.ntotal} vectors, {len(andito_meta)} metadata entries")
    assert andito_index.ntotal == len(andito_meta), "Andito index/metadata count mismatch!"

    # ── Reconstruct vectors ──────────────────────────────────────────────────
    print("Reconstructing vectors...")
    main_vecs = np.zeros((main_index.ntotal, 512), dtype=np.float32)
    for i in range(main_index.ntotal):
        main_vecs[i] = main_index.reconstruct(i)

    andito_vecs = np.zeros((andito_index.ntotal, 512), dtype=np.float32)
    for i in range(andito_index.ntotal):
        andito_vecs[i] = andito_index.reconstruct(i)

    # ── Re-classify main metadata ────────────────────────────────────────────
    print("Re-classifying main metadata...")
    main_meta_new = [reclassify_entry(m) for m in main_meta]

    # Collect set of main landmark names for dedup
    main_names = {m["landmark_name"] for m in main_meta_new}

    # ── Filter and deduplicate andito ────────────────────────────────────────
    print("Filtering and deduplicating andito entries...")
    keep_indices = []
    andito_meta_new = []
    n_filtered_egypt = 0
    n_filtered_dedup = 0

    for i, m in enumerate(andito_meta):
        name = m["landmark_name"]

        # Filter: NOT_IN_EGYPT
        if name in NOT_IN_EGYPT:
            n_filtered_egypt += 1
            continue

        # Resolve GLD alias to canonical name for dedup check
        canonical = GLD_ALIASES.get(name, name)

        # Dedup: skip if canonical name already in main set
        if canonical in main_names:
            n_filtered_dedup += 1
            continue

        # Re-classify (use the original GLD name for image_path, but canonical for lookups)
        entry = reclassify_entry(m)
        # If it was aliased, also add the canonical name to seen set
        main_names.add(name)
        if canonical != name:
            main_names.add(canonical)

        keep_indices.append(i)
        andito_meta_new.append(entry)

    print(f"  Removed {n_filtered_egypt} NOT_IN_EGYPT entries")
    print(f"  Removed {n_filtered_dedup} duplicate entries (via GLD_ALIASES)")
    print(f"  Keeping {len(keep_indices)} andito entries")

    # Filter andito vectors to match
    andito_vecs_keep = andito_vecs[keep_indices] if keep_indices else np.zeros((0, 512), dtype=np.float32)

    # ── Merge ────────────────────────────────────────────────────────────────
    print("Merging indexes...")
    all_vecs = np.vstack([main_vecs, andito_vecs_keep])
    all_meta = main_meta_new + andito_meta_new

    assert all_vecs.shape[0] == len(all_meta), "Vector/metadata count mismatch after merge!"

    # Build new FAISS index
    unified_index = faiss.IndexFlatIP(512)
    unified_index.add(all_vecs)

    # ── Save ─────────────────────────────────────────────────────────────────
    print(f"Saving unified index: {unified_index.ntotal} vectors...")
    faiss.write_index(unified_index, str(OUT_INDEX))

    with open(OUT_META, "wb") as f:
        pickle.dump(all_meta, f)

    print(f"\nDone!")
    print(f"  Output index:    {OUT_INDEX}")
    print(f"  Output metadata: {OUT_META}")
    print(f"  Total vectors:   {unified_index.ntotal}")
    print(f"  Unique landmarks: {len({m['landmark_name'] for m in all_meta})}")

    # QA: show some re-classified examples
    print("\nSample re-classifications (previously misclassified):")
    check_names = ["Bayt_Al-Suhaymi", "Colossi_of_Memnon", "Aswan_High_Dam", "Montaza_Palace"]
    for name in check_names:
        entries = [m for m in all_meta if m["landmark_name"] == name]
        if entries:
            e = entries[0]
            print(f"  {name}: type={e['landmark_type']}, era={e['historical_era']}, city={e['city']}")


if __name__ == "__main__":
    main()
