# %%
import torch
import pandas as pd
import torch
import wandb
from tqdm import tqdm
import datetime
from torch.utils.data import DataLoader

from utils.image_preproc import ImageCaptionDataset
from utils.image_preproc import custom_collate_fn

# %%
data = pd.read_csv("data/flickr_annotations_30k.csv")
image_path = "data/flickr30k-images"

dataset = ImageCaptionDataset(data, image_path)

# %%

train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [0.8, 0.1, 0.1]
)

# %%
# get small sample from train dataset
# train_sample = [train_dataset[i] for i in range(1000)]


# %%
import sentencepiece as spm

# Load the SentencePiece model
sp = spm.SentencePieceProcessor()
sp.load("spm.model")

# Get the vocabulary size
vocab_size = sp.get_piece_size()
print(f"Vocabulary Size: {vocab_size}")

# Instantiate the model with the correct vocabulary size
from transformer.besformer import TransformerB

model = TransformerB(vocab_size=vocab_size)

# %%


# %%
print("Params:", sum(p.numel() for p in model.parameters()))
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
crt = torch.nn.CrossEntropyLoss()

# %%
dataloader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn
)

# %%
wandb.init(project="captions")

# %%

num_epochs = 5

for epoch in range(num_epochs):
    with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
        for batch in dataloader:
            images = batch["image"]
            caption_inputs = batch["caption_inputs"]
            caption_labels = batch["caption_labels"]

            # Check the maximum index in caption_inputs
            max_index = caption_inputs.max().item()
            min_index = caption_inputs.min().item()

            # Clip the indices to be within range
            caption_inputs = torch.clamp(caption_inputs, min=0, max=vocab_size - 1)

            # Verify clamping
            max_index_after_clamp = caption_inputs.max().item()
            # print(f"Max index after clamping: {max_index_after_clamp}")

            # Debugging: Print the shape and some values of caption_inputs
            # print(f"caption_inputs shape: {caption_inputs.shape}")
            # print("caption_inputs max index:", caption_inputs.max().item())
            # print("caption_inputs dtype:", caption_inputs.dtype)  # Should be torch.int64 or torch.long
            # print("caption_inputs device:", caption_inputs.device)

            opt.zero_grad()
            # Use the clamped caption_inputs
            prd = model(images, caption_inputs)
            prd = prd.view(-1, prd.size(-1))
            caption_labels = caption_labels.view(-1)
            loss = crt(prd, caption_labels)
            loss.backward()
            opt.step()

            pbar.set_postfix(loss=loss.item())
            pbar.update(1)

        ts = datetime.datetime.now().strftime("%M_%H_%d")
        torch.save(model.state_dict(), f"weights/m_{ts}.pth")

print("Training complete.")

# %%
