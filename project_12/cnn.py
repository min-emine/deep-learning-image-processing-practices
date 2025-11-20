#import libraries
import tensorflow as tf
from tensorflow.keras import layers , models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


#veri seti yükleme
(X_train , y_train) , (X_test , y_test) = mnist.load_data()

#görsellei ters çevir
#mnist normal :siyah üzerine beyaz rakamlar-> beyaz üzerine siyah rakamlar
X_train = 255 - X_train
X_test = 255 - X_test


#goruntuyu gorsellestır
plt.figure(figsize=(9,3))
for i in range(3):
    plt.subplot(1,3,i+1)
    plt.imshow(X_train[i] , cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()


#normalizasyon ve reshape
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0


#data augmentation
datagen= ImageDataGenerator(
    rotation_range=10,#rastgele 10 dereceye kadar döndürme
    zoom_range=0.1,#%10 yakınlaştırma veya uzaklaştırma
    width_shift_range=0.1,#genişliğin %10 yataysağa sola  kaydırma
    height_shift_range=0.1#%10 dikey kaydırma
    )


#modeli oluşturma
model = models.Sequential([
    #feature extraction
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    #classification
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

print(model.summary())


#modeli derleme
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#modeli eğit ve kaydet
model.fit(datagen.flow(X_train, y_train, batch_size=64),
          epochs=5, validation_data=(X_test, y_test))


model.save('mnist_cnn_model.h5')
print("Model basariyla kaydedildi.")