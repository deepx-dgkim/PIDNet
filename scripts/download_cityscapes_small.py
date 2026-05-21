# download_cityscapes_small.py

import os
from datasets import load_dataset
from tqdm import tqdm

OUT_DIR = "cityscapes_small"
SPLIT = "validation"
NUM_SAMPLES = 500 # MAX 500 samples

img_dir = os.path.join(OUT_DIR, "images")
mask_dir = os.path.join(OUT_DIR, "masks")

os.makedirs(img_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)

ds = load_dataset(
    "Chris1/cityscapes",
    split=SPLIT,
    streaming=True,
)

for idx, sample in enumerate(tqdm(ds.take(NUM_SAMPLES), total=NUM_SAMPLES)):
    image = sample["image"]
    mask = sample["semantic_segmentation"]

    image.save(os.path.join(img_dir, f"{idx:06d}.png"))
    mask.save(os.path.join(mask_dir, f"{idx:06d}_mask.png"))

print(f"Saved {NUM_SAMPLES} samples to {OUT_DIR}")
