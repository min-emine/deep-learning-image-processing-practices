"""
MNIST veri seti:
rakamlama:0-9 toplam 10 sınıf
28x28 boyutunda gri tonlamalı görüntüler
60000 eğitim örneği, 10000 test örneği
gray scale: 0 (siyah) - 255 (beyaz)
amac: el yazısı rakamları tanımak için bir yapay sinir ağı (ANN) modeli oluşturmak


Image processing:
histogram eşitleme: görüntü kontrastını artırmak için kullanılır
gaussion blur: görüntüdeki gürültüyü azaltmak için kullanılır
canny edge detection: görüntüdeki kenarları tespit etmek için kullanılır

ANN ile MNIST veri setini sınıflandırma


libraries:
tensorflow:Keras ile ANN modeli oluşturmak için ve eğitim için
matplotlib: görüntüleri ve eğitim sonuçlarını görselleştirmek için
cv2:opencv image processing

"""


#import libraries
import cv2 #opencv
import numpy as np #sayısal işlemler
import matplotlib.pyplot as plt #görselleştirme

from tensorflow.keras.datasets import mnist #mnist veri seti
from tensorflow.keras.models import Sequential #ann modeli
from tensorflow.keras.layers import Dense, Dropout #ann katmanları
from tensorflow.keras.optimizers import Adam #optimizasyon




#load mnist dataset

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")

"""

x_train shape: (60000, 28, 28)
y_train shape: (60000,)
"""


#image preprossesing

img = x_train[0] #ilk eğitim görüntüsü
stage={"original":img} #orijinal görüntü

#histogram eşitleme
img_eq = cv2.equalizeHist(img)
stage["histogram_equalization"] = img_eq

#gaussian blur
img_blur = cv2.GaussianBlur(img, (5, 5), 0)
stage["gaussian_blur"] = img_blur

#canny edge detection
img_edges = cv2.Canny(img, 50, 150)
stage["canny_edges"] = img_edges


#görselleştirme
fig,axes = plt.subplots(2,2, figsize=(6,6))
axes=axes.flat

for ax, (title,im)in zip(axes,stage.items()):
    ax.imshow(im,cmap ="gray")
    ax.set_title(title)
    ax.axis("off")


plt.suptitle("MNIST Image Processing Stages")
plt.tight_layout()
plt.savefig("mnist_image_processing.png")



#preprocessing fonksiyonu
def preprocess_image(img):
    img_eq = cv2.equalizeHist(img)
    img_blur = cv2.GaussianBlur(img_eq, (5, 5), 0)
    img_edges = cv2.Canny(img_blur, 50, 150)
    features=img_edges.flatten()/255.0
    return features


num_train=10000
num_test=2000

X_train=np.array([preprocess_image(img) for img in x_train[:num_train]])
y_train_sub=y_train[:num_train]

X_test=np.array([preprocess_image(img) for img in x_test[:num_test]])
y_test_sub=y_test[:num_test]

#ann model creation
model =Sequential([
    Dense(128, activation="relu", input_shape=(784,)),#ilk katman ,128 nöron 28*28=784 boyutunda
    Dropout(0.5),#dropout katmanı, aşırı öğrenmeyi önlemek için(overfitting)
    Dense(64, activation="relu"),#ikinci katman,64 nöron
    Dropout(0.5),
    Dense(10, activation="softmax")#cikış katmanı,10 sınıf (0-9 rakamları)
])

#compile model
model.compile(
    optimizer=Adam(),
    loss="sparse_categorical_crossentropy",#çok sınıflı sınıflandırma için kayıp fonksiyonu
    metrics=["accuracy"])

print(model.summary())

#ann model training
history = model.fit(
    X_train, y_train_sub,
    validation_data=(X_test, y_test_sub),
    epochs=50, 
    batch_size=32, 
    verbose=2)
#model.fit(X_train, y_train_sub, epochs=10, batch_size=32, validation_split=0.2)

#evaluate model performance
test_loss, test_acc = model.evaluate(X_test, y_test_sub)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

#plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()




plt.tight_layout()
plt.show()
plt.savefig("training_history.png")
