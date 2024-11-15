import torch
from torch.utils.data import DataLoader
from transformer.besformer import TransformerB
from utils.image_preproc import custom_collate_fn
import matplotlib.pyplot as plt
import numpy as np
import sentencepiece as spm


vocab_size = 16000
model = TransformerB(vocab_size=vocab_size)
model.load_state_dict(torch.load("weights/m_28_19_14.pth"))
model.eval()

# import test dataset
test_dataset = torch.load("data/test_dataset.pth")

# Prepare the test DataLoader
test_dataloader = DataLoader(
    test_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn
)

crt = torch.nn.CrossEntropyLoss()
sp = spm.SentencePieceProcessor()
sp.load("spm.model")


# Function to decode captions from token IDs to text
def decode_caption(caption_ids, sp):
    return sp.decode_ids(caption_ids.tolist())


# Global variables
total_loss = 0
total_samples = 0
examples_to_inspect = 10
inspected_examples = 0


def evaluate():
    global total_loss, total_samples, inspected_examples

    with torch.no_grad():
        for batch in test_dataloader:
            images = batch["image"]
            caption_inputs = batch["caption_inputs"]
            caption_labels = batch["caption_labels"]

            # Forward pass
            prd = model(images, caption_inputs)
            prd = prd.view(caption_labels.size(0), -1, prd.size(-1))
            predicted_ids = prd.argmax(dim=-1)

            # Compute loss
            loss = crt(prd.view(-1, prd.size(-1)), caption_labels.view(-1))
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            # Human inspection of examples
            if inspected_examples < examples_to_inspect:
                for i in range(
                    min(examples_to_inspect - inspected_examples, images.size(0))
                ):
                    # Decode the generated and reference captions
                    generated_caption = decode_caption(predicted_ids[i], sp)
                    reference_caption = decode_caption(caption_labels[i], sp)

                    # Display the image and captions
                    plt.imshow(np.transpose(images[i].numpy(), (1, 2, 0)))
                    plt.title(
                        f"Generated: {generated_caption}\nReference: {reference_caption}"
                    )
                    plt.axis("off")
                    plt.show()

                inspected_examples += 1
                if inspected_examples >= examples_to_inspect:
                    break

    # Calculate average loss
    average_loss = total_loss / total_samples
    print(f"Average Test Loss: {average_loss:.4f}")
    return


# Run evaluation
evaluate()
if __name__ == "__main__":
    # run file evaluation
    evaluate()
