#!/usr/bin/env python3
"""
fetch_andito_egypt.py
---------------------
Stream the `andito/google-landmarks` Hugging Face dataset and save only the
Egyptian landmarks to disk.

Dataset: 252 shards × ~1.74 GB compressed = ~438 GB total.
Egyptian rows: ~0.07 % → expect ~840 images total.
Wall time at HF CDN speed (~1.6 MB/s): ~77 hours unattended.
Shard-level checkpointing lets you pause and resume across sessions.

Strategy
--------
Direct tarfile streaming over HTTP — no library decoding overhead.
For each tar member we buffer the raw image bytes and flush them only if the
matching `.texts.json` member says the landmark is Egyptian.  Non-Egyptian
JPEGs are discarded without touching the filesystem.

Run from V0/ with the venv active. V0/.env must contain one of:
    HF_TOKEN   HF_Token   HUGGINGFACE_TOKEN

Usage:
    python fetch_andito_egypt.py                  # full run, resume-aware
    python fetch_andito_egypt.py --shards 3       # first 3 shards only
    python fetch_andito_egypt.py --dry-run        # count matches, no saves
    python fetch_andito_egypt.py --status         # show checkpoint status
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import re
import sys
import tarfile
import threading
import time
from typing import Optional

import requests
from dotenv import load_dotenv
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))

OUT_DIR        = os.path.join(HERE, "dataset_andito")
IMAGES_DIR     = os.path.join(OUT_DIR, "images")
JSONL_PATH     = os.path.join(OUT_DIR, "andito_raw_records.jsonl")
CHECKPOINT_FILE = os.path.join(OUT_DIR, "fetch_checkpoint.json")
HF_BASE        = "https://huggingface.co/datasets/andito/google-landmarks/resolve/main"

EGYPT_RE      = re.compile(r"\bEgypt\b", re.IGNORECASE)
LOCATION_RE   = re.compile(r"located in\s+(.+?)\.?\s*$", re.IGNORECASE)
PARTS_RE      = re.compile(r"^([^,]+),\s*([^,]+),\s*([^,]+)$")
NAME_RE       = re.compile(r"^\s*This is\s+(.+?)\s*\.?\s*$", re.IGNORECASE)
SAFE_RE       = re.compile(r"[^A-Za-z0-9._()\-]+")


# ── Auth ──────────────────────────────────────────────────────────────────────

def get_token() -> Optional[str]:
    for k in ("HF_TOKEN", "HF_Token", "HUGGINGFACE_TOKEN"):
        v = os.getenv(k)
        if v:
            return v
    return None


# ── Q&A parsing ───────────────────────────────────────────────────────────────

def parse_qa(texts: list) -> dict:
    name = city = region = country = category = None
    for turn in texts or []:
        ans = (turn.get("assistant") or "").strip()
        if not ans:
            continue
        m = NAME_RE.match(ans)
        if m and name is None:
            name = m.group(1).strip().rstrip(".")
            continue
        if "located in" in ans.lower() and country is None:
            loc_m = LOCATION_RE.search(ans)
            if loc_m:
                loc_str = loc_m.group(1).strip()
                pm = PARTS_RE.match(loc_str)
                if pm:
                    city    = pm.group(1).strip() or None
                    region  = pm.group(2).strip() or None
                    country = pm.group(3).strip() or None
                else:
                    pieces = [p.strip() for p in loc_str.split(",") if p.strip()]
                    country = pieces[-1] if pieces else None
                    region  = pieces[-2] if len(pieces) >= 2 else None
                    city    = pieces[-3] if len(pieces) >= 3 else (pieces[0] if pieces else None)
            continue
        if category is None:
            category = ans
    return {"landmark_name": name, "city_text": city, "region_text": region,
            "country": country, "category_text": category}


def is_egyptian(texts: list) -> bool:
    for turn in texts or []:
        if EGYPT_RE.search(turn.get("assistant") or ""):
            return True
    return False


def sanitize(name: str) -> str:
    name = SAFE_RE.sub("_", name.strip().rstrip(".")).strip("_")
    return name[:120] or "Unknown"


# ── Checkpoint ────────────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"done_shards": [], "total_saved": 0, "total_scanned": 0}


def save_checkpoint(cp: dict):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(cp, f, indent=2)


# ── Per-landmark image counter ────────────────────────────────────────────────

def load_counter() -> dict[str, int]:
    counter: dict[str, int] = {}
    if os.path.isdir(IMAGES_DIR):
        for folder in os.listdir(IMAGES_DIR):
            sub = os.path.join(IMAGES_DIR, folder)
            if os.path.isdir(sub):
                counter[folder] = len([f for f in os.listdir(sub) if f.endswith(".jpg")])
    return counter


# ── Shard processor ───────────────────────────────────────────────────────────

def process_shard(shard_name: str, token: str, counter: dict, lock: threading.Lock,
                  jsonl_fh, dry_run: bool, verbose: bool) -> tuple[int, int]:
    """Download and parse one shard. Returns (scanned, saved)."""
    url = f"{HF_BASE}/{shard_name}"
    scanned = saved = 0

    try:
        resp = requests.get(url, headers={"Authorization": f"Bearer {token}"},
                            stream=True, timeout=120)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [ERROR] {shard_name}: {e!r}", flush=True)
        return 0, 0

    # Buffer: while iterating members we might see .jpg before .texts.json
    # Keep a per-key buffer of raw JPEG bytes.
    jpg_buf: dict[str, bytes] = {}
    texts_buf: dict[str, list] = {}

    def flush_key(key: str):
        """Check if we have both jpg + texts for this key; if Egyptian, save."""
        nonlocal saved
        if key not in jpg_buf or key not in texts_buf:
            return
        texts = texts_buf.pop(key)
        jpg_bytes = jpg_buf.pop(key)

        if not is_egyptian(texts):
            return

        parsed = parse_qa(texts)
        name   = parsed.get("landmark_name") or "Unknown"
        folder = sanitize(name)

        if not dry_run:
            sub_dir = os.path.join(IMAGES_DIR, folder)
            os.makedirs(sub_dir, exist_ok=True)
            with lock:
                idx = counter.get(folder, 0)
                counter[folder] = idx + 1
            out_path = os.path.join(sub_dir, f"{idx}.jpg")
            try:
                # Write raw bytes (already JPEG from source); validate via PIL.
                with open(out_path, "wb") as f:
                    f.write(jpg_bytes)
                Image.open(out_path).verify()   # raises if corrupt
            except Exception:
                # Re-encode via PIL if raw write was invalid
                try:
                    Image.open(io.BytesIO(jpg_bytes)).convert("RGB").save(
                        out_path, "JPEG", quality=92)
                except Exception:
                    os.remove(out_path)
                    return

            rec = {
                "landmark_name": parsed["landmark_name"],
                "folder":        folder,
                "city_text":     parsed["city_text"],
                "region_text":   parsed["region_text"],
                "country":       parsed["country"],
                "category_text": parsed["category_text"],
                "image_path":    os.path.relpath(out_path, HERE),
                "image_filename": f"{idx}.jpg",
                "source_key":    key,
                "shard":         shard_name,
            }
            with lock:
                jsonl_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                jsonl_fh.flush()

        saved += 1
        if verbose:
            print(f"    SAVED {folder}/{counter.get(folder,0)-1}.jpg  [{key}]", flush=True)

    try:
        # Stream the gzipped tar without writing it to disk
        gz_stream = resp.raw

        with tarfile.open(fileobj=gzip.GzipFile(fileobj=gz_stream), mode="r|") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                name_in_tar = member.name          # e.g. "images/30880_20.jpg"
                parts       = name_in_tar.rsplit(".", 1)
                if len(parts) != 2:
                    continue
                key_raw, ext = parts[0], parts[1].lower()
                # Strip leading path component to normalise key
                key = key_raw  # e.g. "images/30880_20"

                f = tf.extractfile(member)
                if f is None:
                    continue
                raw = f.read()

                if ext in ("jpg", "jpeg"):
                    jpg_buf[key] = raw
                    scanned += 1
                elif ext in ("json",) or name_in_tar.endswith("texts.json"):
                    try:
                        texts_buf[key] = json.loads(raw)
                    except Exception:
                        texts_buf[key] = []
                # else: source.txt etc — ignore

                flush_key(key)

    except Exception as e:
        print(f"  [ERROR] parsing {shard_name}: {e!r}", flush=True)

    return scanned, saved


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards",   type=int, default=252,
                    help="number of shards to process (default: all 252)")
    ap.add_argument("--dry-run",  action="store_true",
                    help="filter without saving (just count)")
    ap.add_argument("--verbose",  action="store_true",
                    help="print each saved image")
    ap.add_argument("--status",   action="store_true",
                    help="show checkpoint status and exit")
    args = ap.parse_args()

    cp = load_checkpoint()

    if args.status:
        done = len(cp["done_shards"])
        print(f"Checkpoint status:")
        print(f"  Shards done:    {done} / 252")
        print(f"  Rows scanned:   {cp['total_scanned']:,}")
        print(f"  Images saved:   {cp['total_saved']:,}")
        pct = done / 252 * 100
        remaining_h = (252 - done) * 1.74e9 / 1.58e6 / 3600
        print(f"  Progress:       {pct:.1f}%")
        print(f"  Est. remaining: {remaining_h:.0f} h at 1.6 MB/s")
        return

    token = get_token()
    if not token:
        print("ERROR: no HF token in V0/.env (HF_TOKEN / HF_Token / HUGGINGFACE_TOKEN)",
              file=sys.stderr)
        sys.exit(2)

    os.makedirs(OUT_DIR, exist_ok=True)
    if not args.dry_run:
        os.makedirs(IMAGES_DIR, exist_ok=True)

    counter = load_counter()
    already_images = sum(counter.values())
    done_shards    = set(cp.get("done_shards", []))

    print(f"{'='*72}")
    print(f" fetch_andito_egypt — Egyptian subset of andito/google-landmarks")
    print(f"{'='*72}")
    print(f" Shards total:       252  (~438 GB compressed)")
    print(f" Expected Egyptian:  ~840 images (~0.07% of 1.26M rows)")
    print(f" Estimated time:     ~77 hours at HF CDN speed (1.6 MB/s)")
    print(f"                     → leave running overnight / over a weekend")
    print(f" Checkpoint:         {CHECKPOINT_FILE}")
    print(f"   Already done:     {len(done_shards)} shards, {already_images} images")
    if args.dry_run:
        print(f" Mode:               DRY RUN (no files saved)")
    print(f"{'='*72}")

    lock = threading.Lock()
    jsonl_fh = None if args.dry_run else open(JSONL_PATH, "a", encoding="utf-8")

    total_scanned = cp.get("total_scanned", 0)
    total_saved   = cp.get("total_saved",   0) + already_images

    t0 = time.time()

    try:
        n_shards = min(args.shards, 252)
        for shard_idx in range(n_shards):
            shard_name = f"shard-{shard_idx:06d}.tar.gz"

            if shard_name in done_shards:
                continue

            remaining = n_shards - shard_idx - 1
            eta_h       = remaining * 1.74e9 / 1.58e6 / 3600 if remaining > 0 else 0
            print(f"\n[shard {shard_idx+1}/{n_shards}] {shard_name}  "
                  f"saved_so_far={total_saved}  eta≈{eta_h:.0f}h", flush=True)

            sc, sv = process_shard(
                shard_name, token, counter, lock,
                jsonl_fh, args.dry_run, args.verbose
            )
            total_scanned += sc
            total_saved   += sv

            with lock:
                done_shards.add(shard_name)
                cp["done_shards"]    = sorted(done_shards)
                cp["total_scanned"]  = total_scanned
                cp["total_saved"]    = total_saved
                save_checkpoint(cp)

            print(f"  → scanned={sc}  saved={sv}  "
                  f"cumulative: scanned={total_scanned:,}  saved={total_saved}", flush=True)

    except KeyboardInterrupt:
        print("\n[interrupted] checkpoint saved — rerun to resume", flush=True)
    finally:
        if jsonl_fh:
            jsonl_fh.close()

    elapsed = time.time() - t0
    print(f"\n{'='*72}")
    print(f"FINISHED in {elapsed/3600:.1f}h")
    print(f"  Total scanned:  {total_scanned:,} rows")
    print(f"  Total saved:    {total_saved} Egyptian images")
    print(f"  Unique landmarks: {len(counter)}")
    print(f"  Images dir:     {IMAGES_DIR}")
    print(f"  JSONL:          {JSONL_PATH}")


if __name__ == "__main__":
    main()
