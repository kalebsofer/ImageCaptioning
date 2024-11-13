import pandas as pd
import ast

# Load your data
data = pd.read_csv("data/flickr_annotations_30k.csv")

# Extract captions
captions = []
for raw_captions in data["raw"]:
    captions.extend(ast.literal_eval(raw_captions))

# Save captions to a text file
with open("captions.txt", "w", encoding="utf-8") as f:
    for caption in captions:
        f.write(caption + "\n")
