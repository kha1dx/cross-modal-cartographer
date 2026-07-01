"""
Cross-Modal Cartographer — FastAPI Backend
==========================================
POST /search     — sketch + text → top-k landmark results (FAISS/KG toggleable)
GET  /health     — liveness probe
GET  /api/cities — city list for frontend autocomplete
GET  /nearby     — geo-nearest landmarks
GET  /           — serves the web frontend

Run from the repo root (cross-modal-cartographer/):
    ./start_server.sh
or manually, with the venv active:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

# ── OpenMP / MPS guards MUST be first ────────────────────────────────────────
import os
os.environ["OMP_NUM_THREADS"]             = "1"
os.environ["MKL_NUM_THREADS"]             = "1"
os.environ["VECLIB_MAXIMUM_THREADS"]      = "1"
os.environ["NUMEXPR_NUM_THREADS"]         = "1"
os.environ["KMP_DUPLICATE_LIB_OK"]        = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"]      = "false"
os.environ["KMP_INIT_AT_FORK"]            = "FALSE"

import math
import pickle
import sys
from io import BytesIO
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import torch
import clip  # clip-anytorch
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
# The backend is self-contained: index/metadata ship in backend/artifacts/ and
# the pipeline modules live in ../pipeline. A legacy fallback to a sibling V0/
# folder is kept so the original development layout still runs unchanged.
BACKEND_DIR  = Path(__file__).parent
REPO_ROOT    = BACKEND_DIR.parent                   # cross-modal-cartographer/
ARTIFACT_DIR = BACKEND_DIR / "artifacts"            # committed index + metadata
PIPELINE_DIR = REPO_ROOT / "pipeline"               # kg_local.py, build_registry.py
STATIC_DIR   = BACKEND_DIR / "static"

# Legacy dev layout (Project/V0/…): used only if the in-repo copies are absent.
_V0_DIR      = REPO_ROOT.parent / "V0"

# Embeddings/index: prefer the committed backend/artifacts, else fall back to V0.
EMB_DIR = ARTIFACT_DIR if (ARTIFACT_DIR / "faiss_index_unified.bin").exists() \
    or (ARTIFACT_DIR / "faiss_index.bin").exists() else _V0_DIR / "embeddings"

# Optional image directories (1.2 GB — not shipped; thumbnails only).
# Set IMAGES_DIR / ANDITO_IMAGES_DIR env vars to point at a local copy.
DATA_DIR   = Path(os.getenv("IMAGES_DIR", _V0_DIR / "dataset"))
ANDITO_IMG = Path(os.getenv("ANDITO_IMAGES_DIR", _V0_DIR / "egyptian_landmarks_data" / "images"))

# .env is OPTIONAL — the KG is built locally from metadata, no Neo4j at runtime.
ENV_FILE = REPO_ROOT / ".env"
load_dotenv(ENV_FILE)

# Make the in-repo pipeline modules importable (fall back to V0 dev layout).
sys.path.insert(0, str(PIPELINE_DIR if PIPELINE_DIR.exists() else _V0_DIR))

from kg_local import LocalKGClient                # noqa: E402
from build_registry import CITY_COORDS, CITY_ALIASES as _REGISTRY_ALIASES  # noqa: E402

# ── Location aliases (extends build_registry's CITY_ALIASES) ─────────────────
# Maps user-friendly names → canonical city name from CITY_COORDS
LOCATION_ALIASES: dict[str, str] = {}
# Self-map all canonical city names (case-insensitive)
for _city in CITY_COORDS:
    LOCATION_ALIASES[_city.lower()] = _city
# Add build_registry aliases
for _alias, _city in _REGISTRY_ALIASES.items():
    LOCATION_ALIASES[_alias.lower()] = _city
# Add common user-friendly aliases
_EXTRA_ALIASES = {
    "pyramids": "Giza", "pyramid": "Giza", "giza pyramids": "Giza",
    "alex": "Alexandria", "iskandaria": "Alexandria",
    "sharm": "Sharm el-Sheikh", "sharm el sheikh": "Sharm el-Sheikh",
    "valley of the kings": "Luxor", "karnak": "Luxor",
    "abu simbel": "Nubia", "philae": "Aswan",
    "saqqara": "Saqqara", "dahshur": "Dahshur",
    "marsa alam": "Marsa Alam", "el gouna": "Hurghada",
    "matrouh": "Marsa Matrouh", "port said": "Port Said",
    "suez": "Suez", "ismailia": "Ismailia",
    "wadi natrun": "Wadi Natrun", "fayoum": "Faiyum", "fayum": "Faiyum",
    "minya": "Minya", "minia": "Minya", "sohag": "Sohag",
    "qena": "Qena", "edfu": "Edfu", "esna": "Esna",
    "kom ombo": "Kom Ombo", "abydos": "Abydos",
}
for _alias, _city in _EXTRA_ALIASES.items():
    LOCATION_ALIASES[_alias.lower()] = _city


# ── Globals (loaded once at startup) ──────────────────────────────────────────
device: str = "cpu"
clip_model = None
clip_preprocess = None
faiss_index = None
metadata: list = []
kg_client: Optional[LocalKGClient] = None

TAU = 0.60  # similarity acceptance threshold

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Cross-Modal Cartographer",
    description="Sketch + text search engine for Egyptian landmarks",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve dataset images (optional — 1.2 GB, not shipped in the repo).
# Search, similarities and KG facts work without these; only thumbnails need them.
_MAIN_IMG = DATA_DIR / "images"
if _MAIN_IMG.exists():
    app.mount("/images", StaticFiles(directory=str(_MAIN_IMG)), name="images")

# Serve andito (Google Landmarks) images — flat directory (also optional).
if ANDITO_IMG.exists():
    app.mount("/andito-images", StaticFiles(directory=str(ANDITO_IMG)), name="andito-images")


# ── Startup / Shutdown ────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global device, clip_model, clip_preprocess, faiss_index, metadata, kg_client

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()

    # Load unified FAISS index (merged main + andito)
    unified_path = EMB_DIR / "faiss_index_unified.bin"
    fallback_path = EMB_DIR / "faiss_index.bin"
    index_path = unified_path if unified_path.exists() else fallback_path
    faiss_index = faiss.read_index(str(index_path))

    meta_path = EMB_DIR / "faiss_metadata_unified.pkl" if unified_path.exists() else EMB_DIR / "faiss_metadata.pkl"
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    # Build local KG from metadata (no Neo4j needed)
    kg_client = LocalKGClient(metadata)

    print(f"[startup] CLIP on {device} | FAISS {faiss_index.ntotal} vectors | "
          f"LocalKG ready | {len(CITY_COORDS)} cities | index={index_path.name}")


@app.on_event("shutdown")
async def shutdown_event():
    global kg_client
    if kg_client:
        kg_client.close()
    print("[shutdown] cleanup done")


# ── Response schemas ──────────────────────────────────────────────────────────
class LandmarkResult(BaseModel):
    id: str
    name: str
    location: str
    distance_km: Optional[float] = None
    score: float
    image_url: str
    category: str
    is_nearby: bool = False
    historical_era: Optional[str] = None
    architectural_style: Optional[str] = None
    geographic_region: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    schema_pass: bool = True
    verdict: str = "OK"
    nearby_landmarks: list[str] = []
    # Analysis fields
    analysis: Optional[str] = None           # human-readable explanation
    expected_type: Optional[str] = None      # what the query predicted
    tau_threshold: Optional[float] = None    # the tau value used
    score_gap: Optional[float] = None        # how far below tau (for rejected)


class SearchResponse(BaseModel):
    results: list[LandmarkResult]
    rejected_tau: list[LandmarkResult] = []   # results below threshold
    schema_violations: list[LandmarkResult] = []  # results with S-violations
    n_accepted: int
    n_rejected_tau: int
    n_s_violations: int
    query_summary: str
    predicted_type: Optional[str] = None      # inferred query type
    tau: float = 0.60
    use_faiss: bool = True
    use_kg: bool = True


class CityInfo(BaseModel):
    name: str
    lat: float
    lon: float


# ── Helpers ────────────────────────────────────────────────────────────────────
def _encode_image_bytes(image_bytes: bytes) -> np.ndarray:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    tensor = clip_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = clip_model.encode_image(tensor)
        feat /= feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype(np.float32)


def _encode_text(text: str) -> np.ndarray:
    tokens = clip.tokenize([text], truncate=True).to(device)
    with torch.no_grad():
        feat = clip_model.encode_text(tokens)
        feat /= feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype(np.float32)


def _fused_query(sketch_vec: np.ndarray, text: str, alpha: float) -> np.ndarray:
    if text:
        t = _encode_text(text)
        q = alpha * sketch_vec + (1.0 - alpha) * t
        q /= np.linalg.norm(q, axis=-1, keepdims=True) + 1e-9
        return q.astype(np.float32)
    return sketch_vec


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(d_lon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


_TYPE_KW = {
    "pyramid": "Pyramid", "mosque": "Mosque", "temple": "Temple",
    "palace": "Palace", "church": "Church", "museum": "Museum",
    "bridge": "Bridge", "tomb": "Tomb", "obelisk": "Obelisk",
    "fortress": "Fortress", "citadel": "Fortress", "beach": "Natural Site",
    "monastery": "Monastery", "sphinx": "Monument", "dam": "Dam",
}


def _type_hint(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.lower()
    for kw, lt in _TYPE_KW.items():
        if kw in t:
            return lt
    return None


def _resolve_location(location: str) -> tuple[Optional[str], Optional[float], Optional[float]]:
    """Resolve a location string to (city_name, lat, lon).
    Accepts either 'lat,lon' coordinates or a city name / alias.
    Returns (None, None, None) if unresolvable.
    """
    if not location or not location.strip():
        return None, None, None

    location = location.strip()

    # Try lat,lon format
    parts = location.split(",")
    if len(parts) == 2:
        try:
            lat, lon = float(parts[0].strip()), float(parts[1].strip())
            # Find nearest city
            best_city = None
            best_dist = float("inf")
            for city, (clat, clon) in CITY_COORDS.items():
                d = _haversine_km(lat, lon, clat, clon)
                if d < best_dist:
                    best_dist = d
                    best_city = city
            if best_dist <= 80.0:
                return best_city, lat, lon
            return None, lat, lon
        except ValueError:
            pass

    # Try city name / alias lookup
    key = location.lower().strip()
    city = LOCATION_ALIASES.get(key)
    if city and city in CITY_COORDS:
        lat, lon = CITY_COORDS[city]
        return city, lat, lon

    # Partial match: find first city whose name contains the query
    for city_name in CITY_COORDS:
        if key in city_name.lower():
            lat, lon = CITY_COORDS[city_name]
            return city_name, lat, lon

    return None, None, None


def _image_url_for(image_path: str, host: str) -> str:
    """Convert metadata image_path to a URL served by this server.
    Main dataset:  'dataset/images/Mohandessin/8.jpg'   → /images/Mohandessin/8.jpg
    Andito dataset: 'egyptian_landmarks_data/images/1000075_Djoser_Pyramid_complex_in_Sakkara.jpg'
                    → /andito-images/1000075_Djoser_Pyramid_complex_in_Sakkara.jpg
    """
    if image_path.startswith("egyptian_landmarks_data/images/"):
        filename = image_path.replace("egyptian_landmarks_data/images/", "", 1)
        return f"{host}/andito-images/{filename}"
    relative = image_path.replace("dataset/images/", "", 1)
    return f"{host}/images/{relative}"


# ── Endpoints ─────────────────────────────────────────────────────────────────

# Serve frontend
@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/cities", response_model=list[CityInfo])
async def get_cities():
    """Return all cities with coordinates for frontend autocomplete + geolocation."""
    return sorted(
        [CityInfo(name=name, lat=lat, lon=lon) for name, (lat, lon) in CITY_COORDS.items()],
        key=lambda c: c.name,
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "faiss_vectors": faiss_index.ntotal if faiss_index else 0,
        "device": device,
        "kg_landmarks": len(kg_client._landmarks) if kg_client else 0,
    }


@app.post("/search", response_model=SearchResponse)
async def search(
    sketch: UploadFile = File(...),
    text_query: str = Form(""),
    location: str = Form(""),
    alpha: float = Form(0.6),
    k: int = Form(10),
    use_faiss: bool = Form(True),
    use_kg: bool = Form(True),
    host: str = Form("http://localhost:8000"),
):
    if not sketch.content_type or not sketch.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="sketch must be an image file")

    # ── Resolve location ─────────────────────────────────────────────────────
    user_city, user_lat, user_lon = _resolve_location(location)

    # Enhance text query with location context
    effective_text = text_query.strip()
    if user_city and effective_text:
        effective_text = f"{effective_text} in {user_city} Egypt"
    elif user_city:
        effective_text = f"landmark in {user_city} Egypt"

    # ── Encode sketch + fuse ─────────────────────────────────────────────────
    sketch_bytes = await sketch.read()
    sketch_vec = _encode_image_bytes(sketch_bytes)
    query_vec = _fused_query(sketch_vec, effective_text, alpha)

    # ── FAISS search (togglable) ─────────────────────────────────────────────
    if not use_faiss:
        return SearchResponse(
            results=[],
            n_accepted=0,
            n_rejected_tau=0,
            n_s_violations=0,
            query_summary=_query_summary(sketch.filename, text_query, user_city),
            use_faiss=False,
            use_kg=use_kg,
        )

    k_fetch = min(k * 3, faiss_index.ntotal)
    scores, indices = faiss_index.search(query_vec, k_fetch)

    raw = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        entry = dict(metadata[idx])
        entry["_score"] = float(score)
        raw.append(entry)

    # τ filter
    accepted_raw = [r for r in raw if r["_score"] >= TAU]
    rejected_raw = [r for r in raw if r["_score"] < TAU]

    # Schema consistency (from text query, not the location-enhanced version)
    expected_type = _type_hint(text_query)
    n_s_violations = 0

    # Helper to build a LandmarkResult from a raw entry
    def _build_result(r: dict, *, s_pass: bool = True, verdict: str = "OK",
                      analysis: str = None) -> LandmarkResult:
        name = r["landmark_name"]
        kg_data: dict = {}
        nearby_names: list[str] = []

        if use_kg and kg_client:
            try:
                enriched = kg_client.enrich([name])
                if enriched and enriched[0].get("found"):
                    kg_data = enriched[0]
                    nearby_names = [nb["name"].replace("_", " ")
                                    for nb in kg_data.get("nearby_landmarks", [])[:5]]
            except Exception:
                pass

        dist_km: Optional[float] = None
        try:
            lat = float(kg_data.get("lat") or r.get("coordinates_lat") or 0)
            lon = float(kg_data.get("lon") or r.get("coordinates_lon") or 0)
            if user_lat is not None and lat and lon:
                dist_km = round(_haversine_km(user_lat, user_lon, lat, lon), 1)
        except (TypeError, ValueError):
            pass

        city = kg_data.get("city") or r.get("city") or ""
        location_str = f"{city}, Egypt" if city else "Egypt"
        actual_type = kg_data.get("landmark_type") or r.get("landmark_type", "")

        return LandmarkResult(
            id=name,
            name=name.replace("_", " "),
            location=location_str,
            distance_km=dist_km,
            score=round(r["_score"], 4),
            image_url=_image_url_for(r["image_path"], host),
            category=actual_type,
            historical_era=kg_data.get("historical_era") or r.get("historical_era") if use_kg else None,
            architectural_style=kg_data.get("architectural_style") or r.get("architectural_style") if use_kg else None,
            geographic_region=kg_data.get("geographic_region") or r.get("geographic_region") if use_kg else None,
            lat=float(kg_data.get("lat") or r.get("coordinates_lat") or 0) or None,
            lon=float(kg_data.get("lon") or r.get("coordinates_lon") or 0) or None,
            is_nearby=(user_city is not None and city == user_city),
            schema_pass=s_pass,
            verdict=verdict,
            nearby_landmarks=nearby_names,
            analysis=analysis,
            expected_type=expected_type,
            tau_threshold=TAU,
            score_gap=round(TAU - r["_score"], 4) if r["_score"] < TAU else None,
        )

    # ── Deduplicate accepted by landmark_name, keep best score ───────────────
    seen_landmarks: dict[str, dict] = {}
    for r in accepted_raw:
        name = r["landmark_name"]
        if name not in seen_landmarks or r["_score"] > seen_landmarks[name]["_score"]:
            seen_landmarks[name] = r

    # ── Build accepted results (with schema check) ──────────────────────────
    results: list[LandmarkResult] = []
    schema_violations: list[LandmarkResult] = []

    for name, r in list(seen_landmarks.items())[:k]:
        s_pass = True
        verdict = "OK"
        analysis = None
        actual_type = r.get("landmark_type", "")

        if use_kg and kg_client and expected_type:
            if actual_type != expected_type:
                s_pass = False
                n_s_violations += 1
                verdict = "S"
                analysis = (
                    f"Schema violation: the query suggests type \"{expected_type}\" "
                    f"but this landmark is classified as \"{actual_type}\" in the Knowledge Graph. "
                    f"The CLIP similarity score ({r['_score']:.2%}) was above the acceptance "
                    f"threshold (tau={TAU:.0%}), so the visual/textual features matched, but the "
                    f"semantic type from the KG disagrees with the query intent."
                )

        if s_pass:
            analysis = (
                f"Accepted: similarity score {r['_score']:.2%} exceeds the threshold "
                f"(tau={TAU:.0%}) by {r['_score'] - TAU:.2%}."
            )
            if use_kg and expected_type:
                analysis += f" Type \"{actual_type}\" matches the predicted query type."
            results.append(_build_result(r, s_pass=True, verdict="OK", analysis=analysis))
        else:
            result = _build_result(r, s_pass=False, verdict="S", analysis=analysis)
            schema_violations.append(result)
            results.append(result)

    # ── Build rejected (below threshold) results ────────────────────────────
    seen_rejected: dict[str, dict] = {}
    for r in rejected_raw:
        name = r["landmark_name"]
        if name not in seen_rejected or r["_score"] > seen_rejected[name]["_score"]:
            seen_rejected[name] = r

    rejected_results: list[LandmarkResult] = []
    for name, r in list(seen_rejected.items())[:k]:
        gap = TAU - r["_score"]
        actual_type = r.get("landmark_type", "")
        analysis = (
            f"Rejected: similarity score {r['_score']:.2%} is below the acceptance "
            f"threshold (tau={TAU:.0%}) by {gap:.2%}. "
        )
        if gap < 0.05:
            analysis += "This is a near-miss — the result is close to the threshold. "
        elif gap < 0.15:
            analysis += "Moderate gap — the visual/textual features partially match but not strongly enough. "
        else:
            analysis += "Large gap — the CLIP embeddings show weak similarity to the query. "

        if expected_type and actual_type != expected_type:
            analysis += (
                f"Additionally, the landmark type \"{actual_type}\" differs from the "
                f"predicted query type \"{expected_type}\", suggesting both visual and "
                f"semantic mismatch."
            )
        elif expected_type:
            analysis += (
                f"The type \"{actual_type}\" does match the query intent, so the low score "
                f"is likely due to visual dissimilarity between the sketch and this landmark's images."
            )

        rejected_results.append(_build_result(
            r, s_pass=True, verdict="TAU", analysis=analysis,
        ))

    return SearchResponse(
        results=results,
        rejected_tau=rejected_results,
        schema_violations=schema_violations,
        n_accepted=len(accepted_raw),
        n_rejected_tau=len(rejected_raw),
        n_s_violations=n_s_violations,
        query_summary=_query_summary(sketch.filename, text_query, user_city),
        predicted_type=expected_type,
        tau=TAU,
        use_faiss=use_faiss,
        use_kg=use_kg,
    )


def _query_summary(filename: str, text_query: str, city: Optional[str]) -> str:
    parts = []
    if filename:
        parts.append("sketch")
    if text_query:
        parts.append(f'"{text_query}"')
    if city:
        parts.append(f"@ {city}")
    return " + ".join(parts) if parts else "sketch"


# ── /nearby ───────────────────────────────────────────────────────────────────
class NearbyLandmark(BaseModel):
    name: str
    city: str
    category: str
    image_url: str
    distance_km: Optional[float] = None


@app.get("/nearby", response_model=list[NearbyLandmark])
async def nearby(
    lat: float = 0.0,
    lon: float = 0.0,
    k: int = 8,
    host: str = "http://localhost:8000",
):
    seen: dict[str, dict] = {}
    for m in metadata:
        name = m.get("landmark_name", "")
        if name and name not in seen:
            seen[name] = m
    unique = list(seen.values())

    user_city, _, _ = _resolve_location(f"{lat},{lon}") if lat and lon else (None, None, None)

    if user_city and lat and lon:
        def _dist(m: dict) -> float:
            try:
                return _haversine_km(lat, lon,
                                     float(m.get("coordinates_lat") or 0),
                                     float(m.get("coordinates_lon") or 0))
            except (TypeError, ValueError):
                return float("inf")

        sorted_lms = sorted(unique, key=_dist)[:k]
        result_meta = [(m, _dist(m)) for m in sorted_lms]
    else:
        FALLBACK_NAMES = [
            "Great_Pyramid_of_Giza", "Luxor_Temple", "Karnak_precinct_of_Amun-Ra",
            "Pyramid_of_Djoser", "Citadel_of_Qaitbay", "Edfu_Temple",
            "Muhammad_Ali_Mosque", "Mortuary_Temple_of_Hatshepsut",
        ]
        fallback = [m for m in unique if m.get("landmark_name") in FALLBACK_NAMES]
        if len(fallback) < k:
            extra_names = {m["landmark_name"] for m in fallback}
            for m in unique:
                if len(fallback) >= k:
                    break
                if m.get("landmark_name") not in extra_names:
                    fallback.append(m)
                    extra_names.add(m["landmark_name"])
        result_meta = [(m, None) for m in fallback[:k]]

    output = []
    for m, dist in result_meta:
        output.append(NearbyLandmark(
            name=m.get("landmark_name", "").replace("_", " "),
            city=m.get("city") or "",
            category=m.get("landmark_type") or "",
            image_url=_image_url_for(m.get("image_path", ""), host),
            distance_km=round(dist, 1) if dist is not None and dist != float("inf") else None,
        ))
    return output


# ── Static files (must be LAST — catches all remaining paths) ────────────────
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
