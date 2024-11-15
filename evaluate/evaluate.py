import torch
from torch.utils.data import DataLoader
from transformer.transformer import TransformerB
from utils.image_preproc import custom_collate_fn
import matplotlib.pyplot as plt
import numpy as np
import sentencepiece as spm

# get current py path
import os

current_path = os.path.dirname(os.path.abspath(__file__))

vocab_size = 16000
model = TransformerB(vocab_size=vocab_size)
model.load_state_dict(torch.load("weights/m_28_19_14.pth"))
model.eval()

sp = spm.SentencePieceProcessor()
sp.load("spm.model")


def decode_caption(caption_ids, sp):
    return sp.decode_ids(caption_ids)


def generate_caption_for_triplet(test_triplet):
    image = test_triplet["image"].unsqueeze(0)  # Add batch dimension
    caption_input = test_triplet["caption_pairs"][0][0]  # Use the first input caption
    caption_input = torch.tensor(caption_input).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        # Forward pass
        output = model(image, caption_input)
        predicted_ids = output.argmax(dim=-1).squeeze().tolist()

    # Decode the predicted caption
    generated_caption = decode_caption(predicted_ids, sp)
    return generated_caption


test_dataset = torch.load("data/test_dataset.pth")

test_triplet = test_dataset[0]
generated_caption = generate_caption_for_triplet(test_triplet)
print("Generated Caption:", generated_caption)


# get shape of input data
example = test_dataset[0]
sample = test_dataset[0]
image = sample["image"]
type(test_dataset)
# Check the type and shape of the image
print(type(image))
print(image.shape)


caption_pairs = sample["caption_pairs"]

# Check the type and length of the caption_pairs
print(type(caption_pairs))
len(caption_pairs)
caption_pairs[0]
if isinstance(caption_pairs, list):
    print(f"Number of caption pairs: {len(caption_pairs)}")
    # Optionally, check the first caption
    print(caption_pairs[0])


# Inspect the image tensor
image_tensor = example["image"]
print("Image Tensor Shape:", image_tensor.shape)

# Inspect the caption pairs
caption_pairs = example["caption_pairs"]
print("Number of Caption Pairs:", len(caption_pairs))

# Print each caption pair
for idx, (input_caption, target_caption) in enumerate(caption_pairs):
    print(f"Caption Pair {idx + 1}:")
    print("  Input Caption Tokens:", input_caption)
    print("  Target Caption Tokens:", target_caption)
    print("  Input Caption Length:", len(input_caption))
    print("  Target Caption Length:", len(target_caption))
# Prepare the test DataLoader
test_dataloader = DataLoader(
    test_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn
)

crt = torch.nn.CrossEntropyLoss()

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
