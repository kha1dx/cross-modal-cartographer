# The Cross-Modal Cartographer

**A Neuro-Symbolic Retrieval System for Egyptian Cultural Landmarks**

*Bachelor Thesis -- Media Engineering and Technology, German University in Cairo, SS26*

*Author: Khaled Mohamed Salleh*
*Supervisor: Dr. Ahmed M. H. Abdelfattah*

---

## Overview

The Cross-Modal Cartographer is a multimodal search engine that lets users find Egyptian landmarks by **drawing a sketch** and optionally typing a text description. The system fuses both inputs into a shared CLIP embedding space, retrieves candidates from a FAISS vector index of **4,923 photographs** spanning **279 landmark classes**, enriches each result with structured facts from a **Neo4j Knowledge Graph** (48 cities, 11,698 proximity relationships), and verifies every result through a formal **audit function** that classifies outcomes into named failure categories.

The system addresses the "elusive name" problem in cultural tourism: a visitor may remember the silhouette of a monument but lack the vocabulary to search for it. By combining visual intuition with semantic grounding, the Cross-Modal Cartographer bridges two research threads that have remained entirely separate in the published literature -- sketch-based image retrieval and knowledge-graph-augmented verification.

<!-- Add screenshots here -->
<!-- ![Web App](docs/screenshots/webapp-overview.png) -->
<!-- ![Verification Panel](docs/screenshots/verification-panel.png) -->

---

## Formal Framework

The system is formalized through the knowledge tuple **K = (A, D, S)**:

| Symbol | Name | What It Represents |
|--------|------|--------------------|
| **A** | Artifact Space | 4,923 indexed photographic landmark instances |
| **D** | Evidence Space | Embedding + provenance + schema-metadata records per image |
| **S** | Schema Space | Landmark type, historical era, city, region, architectural style |
| **E** | Embedding Space | Shared CLIP ViT-B/32 latent space, E &sub; R<sup>512</sup> |
| **&Phi;** | Perception Mapping | &Phi;: Q &rarr; A &cup; {&empty;} -- encodes sketch/text via CLIP |
| **R** | Retrieval Mapping | R: (A &times; S) &rarr; P(D) -- FAISS search + schema filtering |
| **&vDash;** | Verification Relation | q &vDash; p &hArr; sim(E<sub>q</sub>, E(p)) &ge; &tau; AND schema-consistent |
| **A(q)** | Audit Function | Classifies every outcome into: OK, S-violation, &tau;-rejection, &Phi;-failure, D-failure |
| **&tau;** | Verification Threshold | 0.60 -- modality-calibrated boundary between text and image cones |
| **&alpha;** | Fusion Weight | Controls sketch vs. text contribution. &alpha;=0.3 is accuracy-optimal |

### Pipeline Architecture

```
  Sketch (required)          Text (optional, may include city name)
        |                              |
   CLIP Vision Encoder            CLIP Text Encoder
        |                              |
        Es(x)                        Et(t)
         \                           /
          \                         /
     Weighted Linear Fusion: Eq = norm( alpha * Es + (1-alpha) * Et )
                       |
              FAISS Top-k Retrieval  (IndexFlatIP, cosine similarity)
                       |
              Neo4j KG Enrichment  (type, era, city, style, nearby)
                       |
              Verification Layer
              |-- tau-filter:    sim >= 0.60?
              |-- schema check:  KG type matches query intent?
              |-- audit function A(q): classify verdict
                       |
              Verified Output Yq  +  Rejected candidates with explanations
```

### Key Results

| Finding | Metric |
|---------|--------|
| Sketch+text fusion vs. vague text-only | **25% relative improvement** in P@1 (0.833 vs. 0.667) |
| Sketch contribution to similarity | Always positive (&delta; > 0); +0.18 at &alpha;=0.3, +0.37 at &alpha;=0.7 |
| Sabotage detection (type-contradictory) | **80% correctly rejected** by schema-consistency check |
| Geographic context (city token in text) | Lifts temple P@5 from 0.20 to **1.00** |
| Modality gap | Text-only: all below &tau;. Sketch-fused (&alpha;=0.7): 25/30 above &tau; |

---

## Repository Structure

```
cross-modal-cartographer/
|
|-- backend/                       # FastAPI web server + frontend
|   |-- main.py                    # All endpoints, CLIP/FAISS/KG integration
|   |-- requirements.txt           # Python dependencies
|   |-- static/
|       |-- index.html             # Web UI (sketch canvas + results)
|       |-- app.js                 # Canvas logic (Fabric.js)
|       |-- style.css
|
|-- pipeline/                      # ML pipeline scripts and notebooks
|   |-- build_registry.py          # Landmark metadata classifier (279 landmarks)
|   |-- kg_local.py                # In-memory KG client (no Neo4j needed at runtime)
|   |-- kg_query.py                # Neo4j AuraDB client (cloud, optional)
|   |-- populate_neo4j.py          # Push registry into Neo4j (run once)
|   |-- merge_indexes.py           # Merge main + GLD FAISS indexes
|   |-- fetch_andito_egypt.py      # Download GLD Egypt subset from HuggingFace
|   |-- build_andito_registry.py   # Build GLD subset metadata
|   |-- build_andito_subset_registry.py
|   |-- generate_thesis_figures.py # Generate result visualizations for thesis
|   |-- generate_main_results.py
|   |-- generate_zeroshot_temple.py
|   |-- landmarks_registry.csv     # Master registry: 279 landmarks with metadata
|   |-- gldv2_egypt.csv            # GLD Egypt subset metadata
|   |-- .env.example               # Credential template (Neo4j, HuggingFace)
|   |-- phase0_andito.ipynb        # Phase 0: CLIP encoding + photo filter
|   |-- phase2_andito.ipynb        # Phase 2: FAISS index construction
|   |-- phase2_gld.ipynb           # Phase 2: GLD subset index (cross-dataset)
|   |-- phase3_neo4j.ipynb         # Phase 3: KG enrichment experiments
|   |-- phase4_verification.ipynb  # Phase 4: Audit function A(q) implementation
|   |-- phase5_evaluation.ipynb    # Phase 5: Controlled experiments + results
|   |-- phase_test_andito.ipynb    # Development test notebook
|   |-- _legacy/                   # Earlier notebook versions (archived)
|
|-- sketches/                      # Test sketch inputs
|   |-- pyramid-mid.jpg
|   |-- sketch_mosque.png
|   |-- sketch_temple.png
|
|-- docs/screenshots/              # Add your screenshots here
|-- start_server.sh                # Quick-start script
|-- .gitignore
```

---

## How to Run the Web Application

### Prerequisites

- **Python 3.11+**
- **~4 GB free disk** (CLIP model downloads automatically on first run)
- The **Egyptian Landmarks image dataset** (not included due to size -- see Step 2)
- Precomputed **FAISS index + metadata** (not included -- see Step 2)

### Step 1: Clone and Install Dependencies

```bash
git clone https://github.com/<your-username>/cross-modal-cartographer.git
cd cross-modal-cartographer

python3 -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows

pip install -r backend/requirements.txt
```

### Step 2: Prepare the Data

The image dataset and FAISS artifacts are too large for Git. Create the following directory structure and populate it:

```
cross-modal-cartographer/
|-- V0/
    |-- embeddings/
    |   |-- faiss_index_unified.bin       # FAISS index (or faiss_index.bin)
    |   |-- faiss_metadata_unified.pkl    # Aligned metadata (or faiss_metadata.pkl)
    |-- dataset/
    |   |-- images/                       # 279 landmark subdirectories with JPGs
    |       |-- Great_Pyramid_of_Giza/
    |       |-- Al-Azhar_Mosque/
    |       |-- Karnak_Temple/
    |       |-- ...
    |-- .env                              # (Optional) Neo4j credentials
```

**Option A -- Build from scratch** (if you have the raw image dataset):

```bash
mkdir -p V0/embeddings V0/dataset

# 1. Place images in V0/dataset/images/ (279 subdirectories)

# 2. Copy pipeline scripts into V0 so notebooks can find them:
cp pipeline/build_registry.py V0/
cp pipeline/kg_local.py V0/
cp pipeline/kg_query.py V0/

# 3. Generate the landmark registry:
cd V0
python build_registry.py

# 4. Run the pipeline notebooks in order:
jupyter notebook
# Open and run: phase0_andito.ipynb  (CLIP encoding + photo filter)
# Open and run: phase2_andito.ipynb  (FAISS index construction)
```

**Option B -- Use prebuilt artifacts** (if you have them from a previous run):

```bash
mkdir -p V0/embeddings V0/dataset
cp /path/to/faiss_index_unified.bin V0/embeddings/
cp /path/to/faiss_metadata_unified.pkl V0/embeddings/
# Copy or symlink your images directory:
ln -s /path/to/landmark-images V0/dataset/images
```

### Step 3: (Optional) Set Up Neo4j

The app works **without Neo4j** -- it uses an in-memory Knowledge Graph built from the FAISS metadata at startup via `kg_local.py` (NetworkX). For the full cloud KG:

```bash
cp pipeline/.env.example V0/.env
# Edit V0/.env with your Neo4j AuraDB credentials

cd V0 && python populate_neo4j.py
```

### Step 4: Copy Pipeline Modules for Backend Import

The backend imports `kg_local.py` and `build_registry.py` from `V0/`. If not already there:

```bash
cp pipeline/kg_local.py V0/
cp pipeline/build_registry.py V0/
```

### Step 5: Start the Server

```bash
source .venv/bin/activate
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Or use the convenience script (expects `.venv` in `V0/`):

```bash
./start_server.sh
```

### Step 6: Open the App

Open **[http://localhost:8000](http://localhost:8000)** in your browser.

**What you can do:**

1. **Draw a sketch** on the canvas (pyramid, mosque, temple, etc.)
2. **Type a text description** (e.g., "a stone structure", "ancient temple Luxor")
3. **Adjust &alpha;** with the slider (0.0 = text only, 1.0 = sketch only, **0.3 recommended**)
4. **Toggle FAISS/KG** independently to see the contribution of each component
5. **View per-result verification diagnostics**: cosine similarity score, &tau; threshold, gap, KG-derived attributes (type, era, city, style, nearby landmarks), schema verdict, and a natural-language explanation of the acceptance or rejection

**Useful URLs:**

| URL | What It Shows |
|-----|---------------|
| [localhost:8000](http://localhost:8000) | Web application |
| [localhost:8000/docs](http://localhost:8000/docs) | Interactive API docs (Swagger UI) |
| [localhost:8000/health](http://localhost:8000/health) | Health check (model, index size, device) |

---

## API Reference

### `POST /search`

Main search endpoint. Accepts a sketch image + optional text, returns verified landmark results with full diagnostics.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `sketch` | file | **Yes** | -- | Hand-drawn sketch image (PNG/JPG) |
| `text_query` | string | No | `""` | Text description; may include a city name for geographic context |
| `alpha` | float | No | `0.3` | Fusion weight. 0 = text only, 1 = sketch only |
| `k` | int | No | `10` | Number of candidates to retrieve |
| `use_faiss` | bool | No | `true` | Enable FAISS vector retrieval |
| `use_kg` | bool | No | `true` | Enable Knowledge Graph enrichment |
| `user_lat` | float | No | -- | User latitude (for distance calculation) |
| `user_lon` | float | No | -- | User longitude (for distance calculation) |

**Response:** `SearchResponse` with three result lists:

| Field | Description |
|-------|-------------|
| `results` | Accepted landmarks (passed &tau; + schema) |
| `rejected_tau` | Candidates below the similarity threshold, with explanations |
| `schema_violations` | Candidates with KG type mismatches, with explanations |

Each result includes: `score`, `verdict`, `analysis` (natural-language explanation), `expected_type`, `tau_threshold`, `score_gap`, `category`, `historical_era`, `geographic_region`, `architectural_style`, `nearby_landmarks`, `lat`, `lon`, `image_url`.

### `GET /nearby?lat={lat}&lon={lon}&k={k}`

Returns the *k* nearest landmarks to the given GPS coordinates.

### `GET /api/cities`

Returns all 48 cities with coordinates. Used for frontend autocomplete.

### `GET /health`

Returns model device, FAISS index size, KG status, and server uptime.

---

## Running the ML Pipeline

The pipeline consists of 6 phases executed in order. Each notebook reads artifacts from the previous phase.

| Phase | Notebook | What It Does |
|-------|----------|-------------|
| 0 | `phase0_andito.ipynb` | CLIP ViT-B/32 encoding; photo filter removes diagrams/renders from the index |
| 2 | `phase2_andito.ipynb` | Builds FAISS IndexFlatIP on clean embeddings; attaches registry metadata |
| 2b | `phase2_gld.ipynb` | Same pipeline on Google Landmarks Egypt subset (cross-dataset experiment) |
| 3 | `phase3_neo4j.ipynb` | Queries Neo4j AuraDB to enrich results with structured graph facts |
| 4 | `phase4_verification.ipynb` | Implements &tau;-filter, schema check, semantic drift auditor, failure taxonomy |
| 5 | `phase5_evaluation.ipynb` | Controlled ablation (E1--E4), sabotage benchmark, cross-dataset temple probe |

**Key supporting scripts:**

| Script | Purpose |
|--------|---------|
| `build_registry.py` | Classifies 279 landmark folders into type/era/city/region/style/coordinates |
| `populate_neo4j.py` | Pushes the registry into Neo4j AuraDB (MERGE-based, re-runnable) |
| `kg_local.py` | In-memory KG using NetworkX -- drop-in replacement for Neo4j at runtime |
| `merge_indexes.py` | Merges curated + GLD FAISS indexes into a unified index |
| `fetch_andito_egypt.py` | Downloads Egyptian subset from Google Landmarks v2 via HuggingFace |

---

## Technical Details

### CLIP Encoding and Fusion

All modalities are projected into a shared 512-dimensional space using CLIP ViT-B/32. The fusion rule:

```
Eq = normalize( alpha * Es(sketch) + (1 - alpha) * Et(text) )
```

The sketch is the **primary modality** -- the system cannot operate without it. At &alpha;=0.3 the text carries more fusion weight because natural language provides richer semantic content, but without the sketch every query is &tau;-rejected: text-only cosine similarities cluster at ~0.30, well below &tau;=0.60, due to the CLIP modality gap (Liang et al., NeurIPS 2022).

### FAISS Index

`IndexFlatIP` on L2-normalized vectors = **exact cosine similarity search**. The equivalence (inner product on unit vectors = cosine similarity) was verified against brute-force NumPy computation.

### Knowledge Graph

| Nodes | Relationships |
|-------|---------------|
| 279 Landmarks | LOCATED_IN (Landmark &rarr; City) |
| 48 Cities | IN_REGION (City &rarr; Region) |
| 9 Regions | BUILT_DURING (Landmark &rarr; HistoricalPeriod) |
| 6 Historical Periods | HAS_STYLE (Landmark &rarr; ArchitecturalStyle) |
| 6 Architectural Styles | NEAR (Landmark &harr; Landmark, &le; 5 km Haversine) |

11,698 NEAR edges enable proximity-based exploration.

### Verification Layer

**&tau; = 0.60** is empirically calibrated:
1. Correct-class sketch retrievals score 0.72--0.85; incorrect matches score 0.45--0.60
2. Sits at the boundary between the text cone (~0.30) and image cone (~0.70) of CLIP's joint space
3. 80% sabotage detection while most valid sketch-fused results pass

### Dataset Quality Assurance

A systematic audit found **33% of landmarks** (93/279) were initially misclassified by the keyword-based registry builder (substring collisions, rule-ordering bugs, unsafe defaults). All corrected before evaluation. This validates the schema space S: corrupted schema attributes directly degrade verification verdicts.

---

## Citation

```bibtex
@thesis{Salleh2026CrossModalCartographer,
  author  = {Salleh, Khaled Mohamed},
  title   = {Multi-Sensory Search Engine: The Cross-Modal Cartographer},
  school  = {German University in Cairo},
  year    = {2026},
  type    = {Bachelor Thesis},
  note    = {Supervisor: Dr. Ahmed M. H. Abdelfattah}
}
```

---

## License

This project was developed as a Bachelor Thesis at the German University in Cairo, Media Engineering and Technology Faculty. The code is provided for academic and research purposes.

The Egyptian Landmarks Dataset and Google Landmarks v2 are subject to their respective licenses.
