# %%
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from textwrap import wrap
import pandas as pd
import ast
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict

from torch.utils.data import Dataset, DataLoader

# %%
data = pd.read_csv("data/flickr_annotations_30k.csv")
image_path = "data/flickr30k-images"


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
        # Load the image once
        image = read_image(f"{image_path}/{temp_df.filename[i]}")

        # Parse the 'raw' column to get the list of captions
        captions = ast.literal_eval(temp_df.raw[i])

        # Iterate over each caption associated with the image
        for caption in captions:
            n += 1
            plt.subplot(5, 5, n)
            plt.subplots_adjust(hspace=0.7, wspace=0.3)
            plt.imshow(
                image.permute(1, 2, 0)
            )  # Convert from CxHxW to HxWxC for display
            plt.title("\n".join(wrap(caption, 20)))
            plt.axis("off")

            # Break if we reach the desired number of images to display
            if n >= 15:
                break
        if n >= 15:
            break
    plt.show()


# %%

display_images(
    data.sample(3)
)  # Sample 3 images, each with 5 captions, to display 15 images

# %%
nltk.download("punkt")
nltk.download("stopwords")


def preprocess_caption(caption):
    caption = caption.lower()

    caption = caption.translate(str.maketrans("", "", string.punctuation))

    tokens = word_tokenize(caption)

    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]

    return tokens


# %%


def preprocess_captions(temp_df):
    temp_df["processed_captions"] = temp_df["raw"].apply(
        lambda x: [preprocess_caption(caption) for caption in ast.literal_eval(x)]
    )
    return temp_df


# %%
# Example usage
processed_data = preprocess_captions(data)
print(processed_data[["filename", "processed_captions"]].head())
# %%

# %%
# save the processed data
processed_data.to_csv("data/cleaned_captions.csv", index=False)


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
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
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

        # Parse and preprocess captions
        captions = ast.literal_eval(row["raw"])
        processed_captions = [preprocess_caption(caption) for caption in captions]

        return {"image": image, "captions": processed_captions}


# %%
# Example vocabulary (you should build this from your dataset)
vocab = defaultdict(lambda: len(vocab))
vocab["<PAD>"] = 0  # Padding token


def tokenize_caption(caption):
    return [vocab[word] for word in caption]


def custom_collate_fn(batch):
    images = [item["image"] for item in batch]
    captions = [
        torch.tensor(tokenize_caption(item["captions"][0])) for item in batch
    ]  # Use the first caption for simplicity

    # Stack images into a single tensor
    images = torch.stack(images, dim=0)

    # Pad captions to the same length
    padded_captions = pad_sequence(
        captions, batch_first=True, padding_value=vocab["<PAD>"]
    )

    return {"image": images, "captions": padded_captions}


# %%

# Example usage
image_dir = "data/flickr30k-images"
dataset = ImageCaptionDataset(processed_data, image_dir)
dataloader = DataLoader(
    dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn
)

# %%


# %%
for batch in dataloader:
    images = batch["image"]
    captions = batch["captions"]
    print(images.shape, captions.shape)
    break  # Break after one batch for demonstration
# %%
