# Image Preprocessing Summary


1. **Data Loading:**
   - The dataset is loaded from a CSV file containing image filenames and associated captions.

2. **SentencePiece Tokenization:**
   - A SentencePiece model is used to tokenize captions into sequences of integers, facilitating efficient text processing.

3. **Image Transformation:**
   - Images are resized to a consistent size of 256x256 pixels.
   - Images are converted to PyTorch tensors, scaling pixel values to the range [0, 1].

4. **Normalization:**
   - Images are normalized using the mean `[0.44408225, 0.42113259, 0.38472826]` and standard deviation `[0.25182596, 0.2414833, 0.24269642]` for each RGB channel.
   - **Purpose of Normalization:**
     - **Centering:** Subtracting the mean centers the data around zero, which helps in stabilizing and speeding up the training process.
     - **Scaling:** Dividing by the standard deviation scales the data to have a unit variance, improving convergence during training.

5. **Custom Dataset and DataLoader:**
   - A custom `ImageCaptionDataset` class is implemented to handle image loading and caption tokenization.
   - A `DataLoader` is used to batch the data, applying a custom collate function to pad caption sequences to a uniform length.

### Usage:

- run the `get_norm_metrics.py` script to get the mean and std of the image dataset.
- add the mean and std to the `image_preproc.py` script.
- run the `image_preproc.py` script to preprocess the image dataset.

```python

```