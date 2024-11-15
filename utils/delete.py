import os
import torch
import pandas as pd
import wandb
from tqdm import tqdm
import datetime
from torch.utils.data import DataLoader
import sentencepiece as spm
from ..transformer.transformer import TransformerB
import ast

from PIL import Image
from textwrap import wrap
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


sp = spm.SentencePieceProcessor()
sp.load("spm.model")

# Define special tokens
START_TOKEN = sp.piece_to_id("<s>")
END_TOKEN = sp.piece_to_id("</s>")


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
                    transforms.RandomHorizontalFlip(),  # Added data augmentation
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2
                    ),  # Added data augmentation
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.44408225, 0.42113259, 0.38472826],
                        std=[0.25182596, 0.2414833, 0.24269642],
                    ),
                ]
            )
        )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if isinstance(idx, (list, slice)):
            raise TypeError("Index must be an integer, not a list or slice.")

        row = self.dataframe.iloc[idx]
        image_path = os.path.join(self.image_dir, row["filename"])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        captions = ast.literal_eval(row["raw"])
        processed_captions = [sp.encode(caption, out_type=int) for caption in captions]

        caption_pairs = []
        for caption in processed_captions:
            caption_input = [START_TOKEN] + caption
            caption_label = caption + [END_TOKEN]
            caption_pairs.append((caption_input, caption_label))

        return {"image": image, "caption_pairs": caption_pairs}


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

    attention_mask = (
        padded_caption_inputs != 0
    ).float()  # Added attention mask for padding

    return {
        "image": images,
        "caption_inputs": padded_caption_inputs,
        "caption_labels": padded_caption_labels,
        "attention_mask": attention_mask,
    }


train_dataset = torch.load("../data/train_dataset.pth")
test_dataset = torch.load("../data/test_dataset.pth")
val_dataset = torch.load("../data/val_dataset.pth")


print(len(train_dataset))
print(len(test_dataset))
print(len(val_dataset))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sp = spm.SentencePieceProcessor()
sp.load("../spm.model")
vocab_size = sp.get_piece_size()

model = TransformerB(vocab_size=vocab_size).to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-4)
crt = torch.nn.CrossEntropyLoss()

train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn
)
val_loader = DataLoader(
    val_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn
)

wandb.init(project="captions")


num_epochs = 2
for epoch in range(num_epochs):
    model.train()
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
        for batch in train_loader:
            images = batch["image"].to(device)
            caption_inputs = batch["caption_inputs"].to(device)
            caption_labels = batch["caption_labels"].to(device)

            prd = model(images, caption_inputs)
            prd = prd.view(-1, prd.size(-1))
            caption_labels = caption_labels.view(-1)
            loss = crt(prd, caption_labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            wandb.log({"train_loss": loss.item()})
            pbar.set_postfix(loss=loss.item())
            pbar.update(1)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            caption_inputs = batch["caption_inputs"].to(device)
            caption_labels = batch["caption_labels"].to(device)

            prd = model(images, caption_inputs)
            prd = prd.view(-1, prd.size(-1))
            caption_labels = caption_labels.view(-1)
            loss = crt(prd, caption_labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    wandb.log({"val_loss": avg_val_loss})
    print(f"Validation Loss: {avg_val_loss:.4f}")

    ts = datetime.datetime.now().strftime("%M_%H_%d")
    torch.save(model.state_dict(), f"weights/m_{ts}.pth")

print("Training complete.")


if __name__ == "__main__":
    # run delete.py
    main()
