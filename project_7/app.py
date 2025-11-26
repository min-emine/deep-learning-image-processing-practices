import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

def load_dataset(root, img_size=(128, 128)):
    images, masks = [], []
    for tile in sorted(os.listdir(root)):
        img_dir = os.path.join(root, tile, 'images')
        mask_dir = os.path.join(root, tile, 'masks')
        if not os.path.isdir(img_dir): continue
        
        for f in os.listdir(img_dir):
            if not f.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            img_path = os.path.join(img_dir, f)

            mask_path = os.path.join(mask_dir, os.path.splitext(f)[0] + '.png') 
            if not os.path.exists(mask_path): continue

            # Görüntü İşleme
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size) / 255.0
            
            # Maske İşleme 
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, img_size) / 255.0 
            

            mask = np.expand_dims(mask, axis=-1) 
            
            images.append(img)
            masks.append(mask)
            
    return np.array(images, dtype="float32"), np.array(masks, dtype="float32")
X,y=load_dataset('aerial_dataset', img_size=(128, 128))
print(f"Toplam Örnek:{len(X)}")                


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
print(f"Toplam Train : {len(X_train)} , Test Örnek : {len(X_val)}")

def unet_model(input_size=(128, 128, 3)):
    inputs = keras.Input(input_size)
    

    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1) 

    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2) 
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D()(c3) 

    c4 = layers.Conv2D(128, 3, activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(128, 3, activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D()(c4) 


    c5 = layers.Conv2D(256, 3, activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(256, 3, activation='relu', padding='same')(c5)


    u6 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(128, 3, activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(128, 3, activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(64, 3, activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(64, 3, activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(32, 3, activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(32, 3, activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(16, 2, strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(16, 3, activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(16, 3, activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c9)

    return keras.Model(inputs, outputs)

unet_model= unet_model()
unet_model.compile(optimizer='adam', loss='binary_crossentropy')

callbacks = [
    keras.callbacks.ModelCheckpoint('model_best.h5', save_best_only=True),
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau()
]
history = unet_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    callbacks=callbacks
)

plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.legend()
plt.show()

def show_predictions(idx=0, save_path="tahmin_sonucu.png"):
    img = X_val[idx]
    mask_true = y_val[idx].squeeze()    
    pred_raw = unet_model.predict(img[None, ...])[0].squeeze()
    

    mask_pred = (pred_raw > 0.5).astype("float32") 
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask_true, cmap='gray')
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_raw, cmap='gray') 
    plt.title("Prediction")
    plt.axis("off")

    plt.tight_layout()
    
    filename = f"sonuc_{idx}.png" 
    plt.savefig(filename, bbox_inches='tight')
    print(f"Görsel başarıyla kaydedildi: {filename}")
    
    plt.show()

show_predictions(1)