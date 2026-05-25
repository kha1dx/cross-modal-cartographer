"""
Generate all thesis figures from pre-computed embeddings.
Run from the V0/ directory:  python generate_thesis_figures.py
Saves PNGs to ../SS26_MyThesisTemplate_Ashry_NeSy4/img/
"""

import os
import pickle
import torch
import clip
import matplotlib
matplotlib.use("Agg")          # no display needed — saves files only
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

OUT_DIR = "../SS26_MyThesisTemplate_Ashry_NeSy4/img"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load model ────────────────────────────────────────────────────────────────
device = "mps" if torch.backends.mps.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print(f"Using device: {device}")

# ── Load pre-computed raw embeddings ──────────────────────────────────────────
image_features = torch.load("embeddings/image_features.pt",
                             weights_only=True).to(device)
with open("embeddings/image_paths.pkl", "rb") as f:
    image_paths = pickle.load(f)
print(f"Loaded {len(image_paths)} images")

# ── Build photo filter mask (shared across all queries) ───────────────────────
filter_texts = clip.tokenize([
    "a real photograph of a landmark or building",
    "an architectural diagram, floor plan, or technical drawing",
    "a 3D render or computer generated illustration",
]).to(device)

with torch.no_grad():
    filter_feats = model.encode_text(filter_texts)
    filter_feats /= filter_feats.norm(dim=-1, keepdim=True)
    scores = image_features @ filter_feats.T

keep_mask = scores[:, 0] > scores[:, 1:].max(dim=1).values
filtered_features = image_features[keep_mask]
filtered_paths   = [image_paths[i] for i in keep_mask.nonzero(as_tuple=True)[0].tolist()]
removed = len(image_paths) - len(filtered_paths)
print(f"Filter removed {removed} non-photo images → {len(filtered_paths)} remain")


def _label(path):
    """Return a clean display name from folder name."""
    raw = path.split("/")[-2]
    return raw.replace("-", " ").replace("_", " ").title()


def save_retrieval_grid(sketch_path, query_tag, query_title, top_k=5):
    """Save before/after filter retrieval grids for one sketch query."""
    sketch = preprocess(Image.open(sketch_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        sketch_feat = model.encode_image(sketch)
        sketch_feat /= sketch_feat.norm(dim=-1, keepdim=True)

    # ── BEFORE ────────────────────────────────────────────────────────────────
    sims_raw = (sketch_feat @ image_features.T).cpu().numpy()
    top_raw  = sims_raw.argsort()[0][-top_k:][::-1]

    fig, axes = plt.subplots(1, top_k, figsize=(18, 4))
    for i, idx in enumerate(top_raw):
        path = image_paths[idx]
        try:
            axes[i].imshow(mpimg.imread(path))
        except Exception:
            pass
        axes[i].set_title(
            f"Score: {sims_raw[0][idx]:.3f}\n{_label(path)[:22]}",
            fontsize=8
        )
        axes[i].axis("off")
    fig.suptitle(
        f"Top-{top_k} results — {query_title} sketch  |  BEFORE photo filter",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    out_before = os.path.join(OUT_DIR, f"before_filter_{query_tag}.png")
    plt.savefig(out_before, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_before}")

    # ── AFTER ─────────────────────────────────────────────────────────────────
    sims_clean = (sketch_feat @ filtered_features.T).cpu().numpy()
    top_clean  = sims_clean.argsort()[0][-top_k:][::-1]

    fig, axes = plt.subplots(1, top_k, figsize=(18, 4))
    for i, idx in enumerate(top_clean):
        path = filtered_paths[idx]
        try:
            axes[i].imshow(mpimg.imread(path))
        except Exception:
            pass
        axes[i].set_title(
            f"Score: {sims_clean[0][idx]:.3f}\n{_label(path)[:22]}",
            fontsize=8
        )
        axes[i].axis("off")
    fig.suptitle(
        f"Top-{top_k} results — {query_title} sketch  |  AFTER photo filter",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    out_after = os.path.join(OUT_DIR, f"after_filter_{query_tag}.png")
    plt.savefig(out_after, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_after}")


def save_zero_shot_bar(sketch_path, query_tag, query_title):
    """Save a horizontal bar chart of zero-shot CLIP classification scores."""
    text_labels = [
        "a pyramid", "a mosque", "a temple", "a palace",
        "a bridge", "a museum", "a beach", "a stadium",
    ]
    image_input = preprocess(Image.open(sketch_path)).unsqueeze(0).to(device)
    text_tokens = clip.tokenize(text_labels).to(device)
    with torch.no_grad():
        logits, _ = model(image_input, text_tokens)
        probs = logits.softmax(dim=-1).cpu().numpy()[0]

    colors = ["#2196F3" if p < 0.5 else "#4CAF50" for p in probs]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(text_labels[::-1], probs[::-1], color=colors[::-1])
    ax.set_xlabel("CLIP confidence score", fontsize=10)
    ax.set_title(f"Zero-shot classification — {query_title} sketch", fontsize=11)
    for bar, prob in zip(bars, probs[::-1]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{prob:.4f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.1)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"zeroshot_{query_tag}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ── Generate figures for each query sketch ────────────────────────────────────
QUERIES = [
    ("pyramid-mid.jpg", "pyramid", "Pyramid"),
    ("mosque.jpg",      "mosque",  "Mosque"),
]

for sketch_path, tag, title in QUERIES:
    if not os.path.exists(sketch_path):
        print(f"  Skipping {sketch_path} — file not found")
        continue
    print(f"\n{'─'*50}")
    print(f"Query: {title}  ({sketch_path})")
    save_retrieval_grid(sketch_path, tag, title)
    save_zero_shot_bar(sketch_path, tag, title)

# ── Also save canonical names for the thesis (aliases) ────────────────────────
# 05Results.tex references before_filter.png / after_filter.png (pyramid)
import shutil
for suffix in ("before_filter", "after_filter"):
    src = os.path.join(OUT_DIR, f"{suffix}_pyramid.png")
    dst = os.path.join(OUT_DIR, f"{suffix}.png")
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  Copied {src} → {dst}")

print("\nDone. All figures saved to", OUT_DIR)
