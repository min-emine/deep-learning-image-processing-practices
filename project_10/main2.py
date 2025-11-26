from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import requests
import torch

# Modelleri yükle
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Resmi al
img_url = 'https://bf.sakarya.edu.tr/sites/bf.sakarya.edu.tr/image/IMG_8534.jpg'
image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# İşle 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

pixel_values = processor(image, return_tensors="pt").pixel_values
pixel_values = pixel_values.to(device)

# ÇIKTIYI DEĞİŞKENE ATA 
output_ids = model.generate(pixel_values, max_length=32)

# Decode et ve yazdır
caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("ViT-GPT2 Caption:", caption)