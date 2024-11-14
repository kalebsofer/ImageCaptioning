import os
from PIL import Image
import numpy as np
from torchvision import transforms

image_path = "data/flickr30k-images/"

mean = np.zeros(3)
std = np.zeros(3)
nb_samples = 0

transform = transforms.ToTensor()

for filename in os.listdir(image_path):
    if filename.endswith(".jpg"):
        img = Image.open(os.path.join(image_path, filename)).convert("RGB")
        img_tensor = transform(img)

        mean += img_tensor.mean(dim=(1, 2)).numpy()
        std += img_tensor.std(dim=(1, 2)).numpy()
        nb_samples += 1

mean /= nb_samples
std /= nb_samples

print(f"Mean: {mean}")
print(f"Std: {std}")

# Mean: [0.44408225 0.42113259 0.38472826]
# Std: [0.25182596 0.2414833  0.24269642]
