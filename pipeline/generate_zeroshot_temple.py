"""Generate zero-shot classification bar chart for the temple sketch."""
import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

device = "mps" if torch.backends.mps.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print(f"Using device: {device}")

sketch_path = "sketch_temple.png"
image_input = preprocess(Image.open(sketch_path)).unsqueeze(0).to(device)

text_labels = [
    "a pyramid",
    "a mosque",
    "a temple",
    "a palace",
    "a bridge",
    "a museum",
    "a beach",
    "a stadium",
]
text_tokens = clip.tokenize(text_labels).to(device)

with torch.no_grad():
    logits_per_image, _ = model(image_input, text_tokens)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

for label, prob in zip(text_labels, probs):
    print(f"  {label}: {prob:.4f}")

# Bar chart (same style as pyramid/mosque versions)
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#2196F3' if p < max(probs) else '#F44336' for p in probs]
bars = ax.bar(text_labels, probs, color=colors, edgecolor='black', linewidth=0.5)
ax.set_ylabel("Probability", fontsize=12)
ax.set_title("Zero-Shot CLIP Classification — Temple Sketch", fontsize=13)
ax.set_ylim(0, 1.05)

for bar, prob in zip(bars, probs):
    if prob > 0.01:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{prob:.3f}", ha='center', va='bottom', fontsize=9)

plt.xticks(rotation=30, ha='right')
plt.tight_layout()

out_path = "../SS26_MyThesisTemplate_Ashry_NeSy4/img/zeroshot_temple.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")

# Also save locally
fig.savefig("embeddings/zeroshot_temple.png", dpi=150, bbox_inches="tight")
print("Saved to embeddings/zeroshot_temple.png")
plt.close()
