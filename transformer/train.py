# %%
import torch
import pandas as pd
import wandb
from tqdm import tqdm
import datetime
from torch.utils.data import DataLoader
from utils.image_preproc import ImageCaptionDataset, custom_collate_fn
import sentencepiece as spm
from transformer.besformer import TransformerB


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
data = pd.read_csv("data/flickr_annotations_30k.csv")
image_path = "data/flickr30k-images"
dataset = ImageCaptionDataset(data, image_path)

# Split dataset
train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [0.8, 0.1, 0.1]
)
# %%
sp = spm.SentencePieceProcessor()
sp.load("spm.model")
vocab_size = sp.get_piece_size()

model = TransformerB(vocab_size=vocab_size).to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-4)
crt = torch.nn.CrossEntropyLoss()

# %%
train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn
)
val_loader = DataLoader(
    val_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn
)
# %%
wandb.init(project="captions")

# %%
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

# %%
torch.save(model.state_dict(), f"weights/m_{ts}.pth")

# %%
