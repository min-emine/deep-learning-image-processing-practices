from tensorflow.keras.preprocessing.image  import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D,Dense,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau

import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix 

train_datagen = ImageDataGenerator(rescale=1./255.0,
                   horizontal_flip=True,
                   rotation_range=10,
                   brightness_range=[0.8,1.2],
                   validation_split = 0.1 )

test_datagen = ImageDataGenerator(rescale=1./255.0)

DATA_DIR = "chest_xray"
IMG_SIZE = (224, 224)

train_gen = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'train'),
        target_size=IMG_SIZE,
        batch_size=64,
        class_mode='binary',
        subset='training',
        shuffle=True,)

val_gen = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'train'),
        target_size=IMG_SIZE,
        batch_size=64,
        class_mode='binary',
        subset='validation',
        shuffle=False,
)

# The corrected code for test_gen
test_gen = test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'test'),
        target_size=IMG_SIZE,
        batch_size=64,
        class_mode='binary',
        # subset='validation', <-- REMOVE THIS LINE
        shuffle=False,
)
    
class_names = list(train_gen.class_indices.keys())
images,labels = next(train_gen)
plt.figure(figsize=(10,4))
for i in range(4):
    ax =plt.subplot(1,4,i+1)
    ax.imshow(images[i])
    ax.set_title(class_names[int(labels[i])])
    ax.axis('off')
    plt.tight_layout()
plt.show()

base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
base_model.trainable = False
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(128, activation='relu')(x)
x=Dropout(0.5)(x)
pred=Dense(1, activation='sigmoid')(x)

model=Model(inputs=base_model.input, outputs=pred)
 
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

callbacks = [EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6),
                ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

        
               ]


print("model summary:")
model.summary()
history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=2,
        callbacks=callbacks,
        verbose=1
)
pred_probs = model.predict(test_gen)
pred_labels = (pred_probs>0.5).astype(int).ravel()
true_labels=test_gen.classes


cm=confusion_matrix(true_labels, pred_labels)
disp=ConfusionMatrixDisplay(cm,display_labels=class_names)

plt.figure(figsize=(8,8))
disp.plot(cmap="Blues",colorbar=False)
plt.title("test seti Confusion Matrix")
plt.show()