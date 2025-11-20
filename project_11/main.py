"""
nueral style transfer



"""
# Temel PyTorch kütüphanelerini import etme
import torch
import torch.nn as nn
import torch.optim as optim
# Görüntü işleme ve model yükleme için torchvision modülleri
from torchvision import models, transforms
# Görüntü dosyalarını açmak ve işlemek için PIL kütüphanesi
from PIL import Image
# Görüntüleri göstermek ve kaydetmek için matplotlib
import matplotlib.pyplot as plt
# Eğitim döngüsünde ilerleme çubuğu göstermek için tqdm
from tqdm import tqdm


def load_image(image_path, max_size=400, shape=None):
    image = Image.open(image_path).convert('RGB')
    
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
    
# Görüntü dönüştürme (Transform) adımlarını tanımla
    in_transform = transforms.Compose([
        # Görüntüyü belirlenen boyuta yeniden boyutlandır (Resize)
        transforms.Resize(size),
        # Görüntüyü PyTorch Tensor'e dönüştür
        transforms.ToTensor(),
        # ImageNet ortalama ve standart sapması ile normalleştirme (Önceden eğitilmiş VGG için gerekli)
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
   # Dönüştürmeyi uygula, ilk 3 kanalı (RGB) al ve BATCH boyutu için bir boyut (unsqueeze) ekle
    # Sonuç: [1, 3, H, W] boyutunda bir Tensor 
    image = in_transform(image)[:3, :, :].unsqueeze(0)
    
    return image


def im_convert(tensor):
    image = tensor.to("cpu").clone().detach().cpu().squeeze(0) 
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)
    image = image.clip(0, 1)
    return image


def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram



class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        self.vgg=models.vgg19(pretrained=True).features[:29].eval()       
        for param in self.vgg.parameters():
            param.requires_grad_(False)
        self.layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',  # content representation
            '28': 'conv5_1'
        }
    def forward(self, x):
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x
        return features
    
def run_style_transfer(content_path, style_path, num_steps=1000, style_weight=1e6, content_weight=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    content = load_image(content_path).to(device)
    style = load_image(style_path, shape=content.shape[-2:]).to(device)
    
    model = VGGFeatures().to(device).eval()
    
    content_features = model(content)
    style_features = model(style)
    
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    
    target = content.clone().requires_grad_(True).to(device)
    
    optimizer = optim.Adam([target], lr=0.003)
    
    for step in tqdm(range(num_steps)):
        target_features = model(target)
        
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
        
        style_loss = 0
        for layer in style_grams:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]
            layer_style_loss = torch.mean((target_gram - style_gram)**2)
            b, d, h, w = target_feature.shape
            style_loss += layer_style_loss / (d * h * w)
        
        total_loss = content_weight * content_loss + style_weight * style_loss
        
        optimizer.zero_grad()
        total_loss.backward()  #geri yayılım target tensorü güncelle
        optimizer.step()
        
    return im_convert(target)


content=load_image("content.jpg")
style=load_image("style.jpg",shape=tuple(content.shape[-2:]))
output=run_style_transfer("content.jpg","style.jpg")
plt.figure(figsize=(10,5))
plt.imshow((output))
plt.title("stil transfer sonucu")
plt.axis('off')
plt.imsave("stylized_output.jpg", output)
plt.show()