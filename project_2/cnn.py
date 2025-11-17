"""
flowers dataset: 
    rgb:224x224

CNN ile  modeli oluşturma ve problemi çözme
"""
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


#import libraries
from tensorflow_datasets import load
from tensorflow.data import AUTOTUNE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout)


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau , ModelCheckpoint)





#veri seti yükleme
(ds_train,ds_val),ds_info=load(
    "tf_flowers",
    split=['train[:80%]', 'train[80%:]'],
    as_supervised=True,
    with_info=True,
)
print(ds_info.features)
print("Number of classes:", ds_info.features['label'].num_classes)



fig=plt.figure(figsize=(10,5))
for i, (image,label) in enumerate(ds_train.take(3)):
    ax=fig.add_subplot(1,3,i+1)
    ax.imshow(image.numpy().astype("uint8"))
    ax.set_title(f"Etiket: {label.numpy()}")
    ax.axis('off')



plt.tight_layout()
#plt.show()
#örnek veri görselleştirme



IMG_SIZE=(180,180)
#data augmentation+ preprocessing

def preprocess_train(image,label):
    image=tf.image.resize(image, IMG_SIZE)
    image=tf.image.random_flip_left_right(image)
    image=tf.image.random_brightness(image, max_delta=0.1)
    image=tf.image.random_contrast(image, lower=0.9, upper=1.2)
    image=tf.image.random_crop(image,size=(160,160,3))
    image=tf.image.resize(image,IMG_SIZE)
    iamge=tf.cast(image, tf.float32)/255.0
    return image,label

def preprocess_val(image,label):
    image=tf.image.resize(image, IMG_SIZE)
    image=tf.cast(image, tf.float32)/255.0
    return image,label


#veri setini hazırlama
ds_train=(
    ds_train
    .map(preprocess_train, num_parallel_calls=AUTOTUNE)
    .shuffle(1000)
    .batch(32)
    .prefetch(buffer_size=AUTOTUNE)
)

ds_val=(
    ds_val
    .map(preprocess_val, num_parallel_calls=AUTOTUNE)
    .batch(32)
    .prefetch(buffer_size=AUTOTUNE) 
)



#CNN modeli oluşturma
model=Sequential([
    Conv2D(32,(3,3),activation='relu', input_shape=(*IMG_SIZE,3)),#32 filtre ,3x3 kernel, relu aktivasyon ,3 kanal rgb
    MaxPooling2D((2,2)),
    Conv2D(64,(3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128,(3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(ds_info.features['label'].num_classes, activation='softmax')
])

#callbacks
callbacks=[
    EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),#eger val loss 3 epoch boyunca iyileşmezse eğitimi durdurur ve en iyi ağırlıkları geri yükler
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2,verbose=1, min_lr=1e-9),#val loss 2 epoch boyunca iyileşmezse öğrenme oranını 0.2 katına düşürür
    ModelCheckpoint('best_model.h5', save_best_only=True)#her epoch sonunda en iyi modeli kaydeder
]

#derleme
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']

)


print(model.summary())
#training

history=model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=30,
    callbacks=callbacks,
    verbose=1
)

#model değerlendirme ve görselleştirme

plt.figure(figsize=(12,5))


plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama(validasyon) Doğruluğu')
plt.title('Doğruluk')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()


plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama(validasyon) Kaybı')
plt.title('Kayıp')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()



plt.tight_layout()
plt.show()