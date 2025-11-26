# Derin Öğrenme ve Görüntü İşleme Projeleri

Bu repository, BTK Akademi'nin "Derin Öğrenme ile Görüntü İşleme" eğitim serisindeki pratikleri ve bu konular üzerine geliştirdiğim ek başlangıç projelerini içermektedir.

## Projeler

Bu koleksiyondaki projelerin bir listesi:

1.  **[Proje 01: MNIST ile Görüntü Ön İşleme ve ANN](./project_1)**
    * OpenCV kullanarak temel filtreleme (blur, sharpening) teknikleri.
2.  **[Proje 02: CNN ile Çiçek Görüntüsü Sınıflandırma](./project_2)**
    * CNN ile $5$ farklı çiçek sınıfını ayırt etme (Çok Sınıflı Sınıflandırma)
    * TensorFlow Datasets (tfds) ve gelişmiş veri artırma teknikleri.
3.  **[Proje 03: Transfer Learning ile Zatürre Hastalığı Tespiti](./project_3)**
    * Transfer Öğrenme (DenseNet121) kullanarak Göğüs Röntgeni görüntülerinde Zatürre (Pneumonia) tespiti (İkili Sınıflandırma).
4.  **[Proje 04: GAN ile moda ürünü tasarımı](./project_4)**
    * Üretken Çekişmeli Ağlar (GAN/DCGAN) mimarisi ile sentetik moda ürünleri görüntüleri tasarlama.
    * Conv2DTranspose ve tanh aktivasyonu kullanımı.
5.  **[Proje 05: YOLO ile Trafik Levhaları Tespiti](./project_5)**
    * Nesne Tespiti için Ultralytics YOLOv8 modelinin özel bir veri seti üzerinde eğitilmesi ve gerçek zamanlı tespit (Inference).
6.  **[Proje 06: YOLO ile Şehir İçi Trafikte Araç Takibi](./project_6)**
    * YOLOv8 modelini kullanarak şehir içi trafik videolarında araçları tespit etme ve ByteTrack algoritması ile sürekli takip etme..
7.  **[Proje 07: U-Net ile Uydu Görüntülerini Bölütleme](./project_7)**
    * U-Net mimarisini kullanarak uydu görüntülerinde farklı arazi örtüsü sınıflarını (yol, bina, su, vb.) piksel düzeyinde bölütleme (Semantic Segmentation). Model, Encoder-Decoder yapısıyla ve skip connections ile eğitilmiştir..
8.  **[Proje 08: Mediapipe ile Squat Sayacı](./project_8)**
    * MediaPipe Pose ile anahtar eklem noktası tespiti ve geometrik açılarla hareket analizi.
9.  **[Proje 09: Mediapipe ile Yüz İfadesinden Duygu Tanıma](./project_9/)**
    * MediaPipe Face Mesh kullanarak yüz mimiklerine dayalı kural tabanlı duygu sınıflandırması.
10. **[Proje 10: BLIP ile Resimden Açıklama Oluşturma](./project_10)**
    * BLIP ve ViT-GPT2 gibi Vizyon-Dil modellerini kullanarak bir görselin içeriğini analiz edip, doğal dilde otomatik açıklama (Image Captioning) metni üretme.
11. **[Proje 11: PyTorch ve VGG-19 sinir ağı tabanlı Nöral Stil Transferi](./project_11)**
    * PyTorch ve VGG-19 modelini kullanarak İçerik ve Stil Kaybı optimizasyonu ile görüntü dönüşümü.
12. **[Proje 12: Gerçek Zamanlı Görüntü İşleme ile Rakam Sınıflandırma](./project_12)**
    * Eğitilmiş CNN modelini kullanarak web kamerası üzerinden gerçek zamanlı el yazısı rakam tanıma..
13. **[Proje 13: Video Üzerinden YOLO ile Araç ve İnsan Sayma](./project_13)**
    * YOLOv8 ve ByteTrack ile nesneleri takip ederek sanal çizgi geçişlerine göre sayım ve yön analizi.
14. **[Proje 14: Evrişimsel Sinir Ağı (CNN) Kullanarak El Radyografilerinden Kemik Yaşı Tahmini](./project_14)**
    * Görüntü tabanlı regresyon (kemik yaşı tahmini) için CNN ve Data Augmentation kullanımı.


## Kullanılan Teknolojiler

* Python 3.x
* OpenCV
* PyTorch
* Ultralytics YOLOv8 & ByteTrack
* MediaPipe
* Numpy / Pandas
* TensorFlow / Keras
* Matplotlib
