# %%
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json

# %%


class FlattenedFlickr30kDataset(Dataset):
    def __init__(self, image_dir, metadata_file, transform=None):
        # Load and flatten metadata
        self.metadata = pd.read_csv(metadata_file)
        self.image_dir = image_dir
        self.transform = (
            transform
            if transform
            else transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        )

        # Parse JSON in the 'raw' column
        self.metadata["raw"] = self.metadata["raw"].apply(json.loads)

        # Filter for train split if needed and flatten
        self.flattened_data = []
        for _, row in self.metadata[self.metadata["split"] == "train"].iterrows():
            image_id = row["img_id"]
            for caption in row["raw"]:  # Use 'raw' for captions
                self.flattened_data.append({"image_id": image_id, "caption": caption})

    def __len__(self):
        return len(self.flattened_data)

    def __getitem__(self, idx):
        data = self.flattened_data[idx]
        image_id = data["image_id"]
        caption = data["caption"]

        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        print(f"Trying to open image: {image_path}")
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return {"image": image, "caption": caption, "image_id": image_id}

# %%
image_dir = "data/flickr30k_images/"
metadata_file = "data/flickr_annotations_30k.csv"

dataset = FlattenedFlickr30kDataset(image_dir, metadata_file)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# %%
# test if all images have corresponding metadata
for data in dataset:
    if not os.path.exists(data["image_path"]):
        print(f"Image does not exist: {data['image_path']}")

# %%
for batch in dataloader:
    images = batch["image"]
    captions = batch["caption"]
    image_ids = batch["image_id"]
    print(images.shape, captions, image_ids)
    break  # Break after one batch for demonstration

# %%
