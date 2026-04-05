import matplotlib.pyplot as plt

text = """=== AFTER preprocessing: mosei_bottleneck.pkl (processed data) ===

File structure:
├── train   → 13843 segments
├── val     → 2957 segments
├── test    → 2965 segments

Example of one sample 'auTpzciNgls[0]':
    audio shape:    (17, 74)    → real-length audio features
    visual shape:   (17, 713)   → real-length visual features
    label:          int         → 6 emotion classes
    text:           string      → cleaned sentence

Format:  Pickle file (training-ready)
Result:  normalized, split, no global padding, ready for training
"""

fig = plt.figure(figsize=(14, 6), facecolor="#2b2b2b")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_facecolor("#2b2b2b")
ax.axis("off")

ax.text(
    0.03, 0.97, text,
    va="top", ha="left",
    fontsize=18,
    family="monospace",
    color="#e6e6e6"
)

plt.savefig("after_preprocessing.png", dpi=200, bbox_inches="tight", facecolor="#2b2b2b")
plt.close()
print("Saved: after_preprocessing.png")