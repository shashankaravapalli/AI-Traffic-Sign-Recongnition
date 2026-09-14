import os
from torchvision.datasets import GTSRB

CLASS_MAP = {
    14: "stop",
    13: "yield",
    17: "no_entry",
    33: "turn_right",
    1: "speed_limit"
}

MAX_IMAGES = 300

for folder in CLASS_MAP.values():
    os.makedirs(f"dataset/{folder}", exist_ok=True)

print("Downloading GTSRB dataset...")

dataset = GTSRB(
    root="gtsrb_data",
    split="train",
    download=True
)

counts = {name: 0 for name in CLASS_MAP.values()}

for image, label in dataset:
    if label not in CLASS_MAP:
        continue

    folder = CLASS_MAP[label]

    if counts[folder] >= MAX_IMAGES:
        continue

    filename = f"dataset/{folder}/{folder}_{counts[folder]:04d}.jpg"
    image.convert("RGB").save(filename)

    counts[folder] += 1

    if all(count >= MAX_IMAGES for count in counts.values()):
        break

print("Dataset preparation complete!")
print(counts)