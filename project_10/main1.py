from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests 
import torch

processor =BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model=BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

img_url = 'https://bf.sakarya.edu.tr/sites/bf.sakarya.edu.tr/image/IMG_8534.jpg'
image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

inputs = processor(image, return_tensors="pt")
with torch.no_grad():
    output = model.generate(**inputs)

caption = processor.decode(output[0], skip_special_tokens=True)
print("Generated Caption:", caption)
