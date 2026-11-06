- **An image with an aspect ratio past 200 is refused on Qwen VL and GLM-OCR models, as Hugging Face's processor refuses it.** The
  resize floored the short side to one patch cell without shrinking the long side to match, so a thin image of a few kilobytes
  resized to many times the model's pixel budget: a 1 x 1,048,576 PNG became about 3 GB of pixel values under the Qwen3.5 config.
  Such a request now fails with an error naming the image's size.
