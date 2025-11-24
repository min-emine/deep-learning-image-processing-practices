import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanAbsoluteError


#import libraries



#veri seti yükleme ve temizleme

df=pd.read_csv("boneage-training-dataset.csv")
#klasörde görseli gerçekten var olan resimleri alalım
image_folder="boneage-training-dataset"
available_files=set(os.listdir(image_folder))
available_ids=set(f.replace('.png','') for f in available_files if f.endswith('.png'))
df=df[df['id'].astype(str).isin(available_ids)].reset_index(drop=True)


#kemik yaşını normalizasyon
df['boneage']=df['boneage']/240.0
df['path']=df['id'].apply(lambda x: os.path.join(image_folder, f"{x}.png")) 
print(df.head())


#yaş dağılımı
plt.hist(df['boneage']*240, bins=50)
plt.xlabel("Bone Age (months)")
plt.ylabel("Frequency")
plt.title("Distribution of Bone Age")
plt.tight_layout()
#plt.show()
#görüntüleri okuma ve ön işleme
def load_images(df,img_size=128):
    images=[]
    valid_indices=[]
    for i, path in enumerate(df['path']):
        img=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is  None:
            print("uyari")
            continue
        img=cv2.resize(img, (img_size,img_size))
        img=img/255.0
        images.append(img)
        valid_indices.append(i)
    new_df=df.iloc[valid_indices].reset_index(drop=True)
    return np.array(images).reshape(-1,img_size,img_size,1), new_df['boneage'].values
X,y=load_images(df)
print(X.shape)


#egitim ve test veri seti olusturma

X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.15,random_state=42)

#data augmentation
datagen=ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

datagen.fit(X_train)

#cnn modeli

model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(128,128,1)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='linear'))#regresyon çıktısı

#model compile
model.compile(optimizer=Adam(learning_rate=0.001), loss='mae', metrics=[MeanAbsoluteError()])
#callback tanımlama:erken durdurma ,model kaydı ve LR ayarlaması
callbacks=[
    EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True),
    ModelCheckpoint('bone_age_model.keras', monitor='val_loss', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)]



#model egitimi
history=model.fit(datagen.flow(X_train,y_train,batch_size=32),
                  validation_data=(X_val,y_val),
                  epochs=50,
                  callbacks=callbacks)


#model evaluation
plt.plot(history.history['loss'], label='Training mae')
plt.plot(history.history['val_loss'], label='Validation mae')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.title('Training performance')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

preds=model.predict(X_val)*240
actuals=y_val*240

plt.figure()
for i in range(10):
    plt.subplot(7,5,i+1)
    plt.imshow(X_val[i].reshape(128,128),cmap='gray')
    plt.title(f"tahmin: {preds[i][0]:.0f}\ngercek: {actuals[i]:.1f}")
    plt.axis('off')


plt.suptitle("kemik yaş tahmin sonuclari") 
plt.tight_layout()
plt.show()   