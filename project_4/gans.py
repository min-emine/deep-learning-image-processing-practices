"""
GAN fashion mnist verii seti  ile  moda ürünü tasarımı



"""

#import libraries
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.datasets import fashion_mnist


BUFFER_SIZE=60000
BATCH_SIZE=128
NOIDE_DIM=100#genetator için gürültü vektör boyutu
EPOCHS=10

IMG_SHAPE=(28,28,1)
#veri seti yükle

(train_images, _), (_, _) = fashion_mnist.load_data()
train_images = train_images.reshape(-1, 28, 28, 1).astype('float32')#şekillendir ve floata çevir
train_images = (train_images - 127.5) / 127.5  # Normalize to [-1, 1]
train_dataset= tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE) #veri setini shuffle et ve batchle

#generator modeli tanımla:fake görüntü üretir
def make_generator_model():
    model =tf.keras.Sequential([
        layers.Dense(7*7*256, use_bias=False, input_shape=(NOIDE_DIM,)),#ilk tam bağlı katman ,gürültüyü özellik haritasına çevirir
        layers.BatchNormalization(),#eğitim stabilitesini arttırır
        layers.LeakyReLU(),#aktivasyon fonksiyonu negatif girişleri yumuşatır
        layers.Reshape((7, 7, 256)),#özellik haritasını 7x7x256 şekline getirir  tek boyutlu vektörü 3d ye çevirir
        
        
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),#ilk transpoze konvolüsyon katmanı
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),#ilk transpoze konvolüsyon katmanı
        layers.BatchNormalization(),
        layers.LeakyReLU(),
       
       
        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False,activation="tanh"),#ikinci transpoze konvolüsyon katmanı
    ])
    return model

generator =make_generator_model()



#disciriminator modeli tanımla: gerçek mi sahte mi ayırt eder
def make_discriminator_model():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=IMG_SHAPE),#ilk konvolüsyon katmanı
        layers.LeakyReLU(),
        layers.Dropout(0.3),#aşırı öğrenmeyi önler
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),#ikinci konvolüsyon katmanı
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),#çok boyutlu veriyi tek boyuta indirger
        layers.Dense(1)#çıkış katmanı gerçek mi sahte mi karar verir
    ])
    return model


dicsriminator = make_discriminator_model()

#KAYIP LOSS FUNCTİON TANIMLA,GENERATOR VE DİSCRİMİNATOR MODELLERİNİ EĞİTECEK
cross_entropy=tf.keras.losses.BinaryCrossentropy()

def dicriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)#gerçek görüntüler 1 etiketine sahip olsun
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)#sahte görüntüler için kayıp hedef 0 olsun
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)#sahte görüntüler için kayıp hedef 1 olsun  generator sahte görüntüyü 1 gibi göstericek


generator=make_generator_model()
dicsriminator=make_discriminator_model()

generator_optimizer = tf.keras.optimizers.Adam(1e-4)#optimizer tanımla
dicsriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

#yardımcı fonksiyonları tanımla 

seed =tf.random.normal([16, NOIDE_DIM])#sabit gürültü vektörü oluştur

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)#modeli kullanarak görüntü üret
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((predictions[i, :, :, 0]+1)/2, cmap='gray')#görüntüyü geri ölçeklendir
        plt.axis('off')
    if not os.path.exists('generated_images'):
        os.makedirs('generated_images')
    plt.savefig(f'generated_images/image_at_epoch_{epoch:03d}.png')#görüntüyü kaydet
    plt.close()


#eğitim fonsiyonu tanımla :generator ile dicriminator mmodellerini eğitecek
def train(dataset, epochs):


    for epoch in range(1,epochs+1):
        gen_loss_total=0
        disc_loss_total=0
        batch_output=0
    
        for image_batch in dataset:
            noise = tf.random.normal([BATCH_SIZE, NOIDE_DIM])#gürültü vektörü oluştur
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = generator(noise, training=True)#sahte görüntü üret
                real_output = dicsriminator(image_batch, training=True)#gerçek görüntüleri değerlendir
                fake_output = dicsriminator(generated_images, training=True)#sahte görüntüleri değerlendir
                gen_loss = generator_loss(fake_output)#generator kaybını hesapla
                disc_loss = dicriminator_loss(real_output, fake_output)#disciriminator kaybını hesapla
            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)#generator için gradyanları hesapla
            gradients_of_discriminator = disc_tape.gradient(disc_loss, dicsriminator.trainable_variables)#disciriminator için gradyanları hesapla
            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))#generatoru güncelle
            dicsriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, dicsriminator.trainable_variables))#disciriminatoru güncelle

            gen_loss_total += gen_loss
            disc_loss_total += disc_loss
            batch_output += 1
        print(f'Epoch: {epoch}/{epochs}, Generator Loss: {gen_loss_total/batch_output:.3f}, Discriminator Loss: {disc_loss_total/batch_output:.3f}')
        generate_and_save_images(generator, epoch , seed)#her epoch sonunda görüntü üret ve kaydet

train(train_dataset, EPOCHS)