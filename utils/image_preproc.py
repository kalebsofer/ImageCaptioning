# %%
import ast
import os
import torch
import pandas as pd
import sentencepiece as spm
import matplotlib.pyplot as plt
from PIL import Image
from textwrap import wrap
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

# %%
data = pd.read_csv("data/flickr_annotations_30k.csv")
image_path = "data/flickr30k-images"

# %%
sp = spm.SentencePieceProcessor()
sp.load("spm.model")

# Define special tokens
START_TOKEN = sp.piece_to_id("<s>")
END_TOKEN = sp.piece_to_id("</s>")


# %%
def read_image(path, img_size=224):
    # Define the transformation pipeline
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )

    img = Image.open(path).convert("RGB")

    img = transform(img)

    return img


# %%
def display_images(temp_df):
    temp_df = temp_df.reset_index(drop=True)
    plt.figure(figsize=(20, 20))
    n = 0
    for i in range(len(temp_df)):
        image = read_image(f"{image_path}/{temp_df.filename[i]}")

        captions = ast.literal_eval(temp_df.raw[i])

        for caption in captions:
            n += 1
            plt.subplot(5, 5, n)
            plt.subplots_adjust(hspace=0.7, wspace=0.3)
            plt.imshow(image.permute(1, 2, 0))
            plt.title("\n".join(wrap(caption, 20)))
            plt.axis("off")

            if n >= 15:
                break
        if n >= 15:
            break
    plt.show()


# %%

display_images(data.sample(3))


# %%
class ImageCaptionDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = (
            transform
            if transform
            else transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        # from get_norm_metrics.py
                        mean=[0.44408225, 0.42113259, 0.38472826],
                        std=[0.25182596, 0.2414833, 0.24269642],
                    ),
                ]
            )
        )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = os.path.join(self.image_dir, row["filename"])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        captions = ast.literal_eval(row["raw"])
        processed_captions = [sp.encode(caption, out_type=int) for caption in captions]

        # Generate (caption-input, caption-label) pairs
        caption_pairs = []
        for caption in processed_captions:
            caption_input = [START_TOKEN] + caption
            caption_label = caption + [END_TOKEN]
            caption_pairs.append((caption_input, caption_label))

        return {"image": image, "caption_pairs": caption_pairs}


# %%
def custom_collate_fn(batch):
    images = [item["image"] for item in batch]
    images = torch.stack(images, dim=0)

    caption_inputs = []
    caption_labels = []

    for item in batch:
        for caption_input, caption_label in item["caption_pairs"]:
            caption_inputs.append(torch.tensor(caption_input))
            caption_labels.append(torch.tensor(caption_label))

    padded_caption_inputs = pad_sequence(
        caption_inputs, batch_first=True, padding_value=0
    )
    padded_caption_labels = pad_sequence(
        caption_labels, batch_first=True, padding_value=0
    )

    return {
        "image": images,
        "caption_inputs": padded_caption_inputs,
        "caption_labels": padded_caption_labels,
    }


# %%
dataset = ImageCaptionDataset(data, image_path)

import pickle

with open("data/processed_dataset.pkl", "wb") as f:
    pickle.dump(dataset, f)

# %%

# number of captions per image
print(len(dataset[0]["caption_pairs"]))  # 5

# shape of first caption:
print(torch.tensor(dataset[0]["caption_pairs"][0][0]).shape)  # torch.Size([18])

# shape of first image:
print(dataset[0]["image"].shape)  # torch.Size([3, 256, 256])


# %%
# Example usage

dataloader = DataLoader(
    dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn
)

# for batch in dataloader:
#     images = batch["image"]
#     caption_inputs = batch["caption_inputs"]
#     caption_labels = batch["caption_labels"]
#     print(images.shape, caption_inputs.shape, caption_labels.shape)
#     break  # Break after one batch for demonstration

# %%
