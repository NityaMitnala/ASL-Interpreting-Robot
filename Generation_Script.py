from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import os
import re

# Load the saved model and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("ASL_Translation_Model")
tokenizer = AutoTokenizer.from_pretrained("ASL_Translation_Tokenizer")
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert(mode="RGB")
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
    return pixel_values

def text_generation():
    image_path = input("Please enter the path to the image of the ASL command you wish to give.\n")
    pixel_values = preprocess_image(image_path)

    # Generate the caption
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)

    # Decode the generated ids to get the caption
    generated_caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("Translation of the image:", generated_caption)
    return generated_caption

# caption = text_generation()
# print("Generated Caption:", caption)