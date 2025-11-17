from ultralytics import YOLO
import cv2

# Modeli yükle
model = YOLO('runs/detect/traffic-sign-model/weights/best.pt') 

# Görüntü yolu ve okuma
image_path = 'test1.jpg'
image = cv2.imread(image_path)

# Modeli çalıştır
results = model(image_path)

# Sonuçlar listesini döngüye al (tek resim için bu döngü bir kez çalışır)
for res in results:
    # Tespit edilen her kutu (bounding box) için döngü
    for box in res.boxes:
        # Koordinatları al
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Sınıf ID'sini al
        cls_id = int(box.cls[0])
        # Güven skorunu al
        confidence = float(box.conf[0])
        # Etiketi oluştur (sınıf adı + güven skoru)
        label = (f"{res.names[cls_id]} conf:{confidence:.2f}")
        
        # Görüntü üzerine dikdörtgen çiz
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Görüntü üzerine etiketi yaz
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Sonucu göster
cv2.imshow("Prediction", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Sonucu kaydet
cv2.imwrite("prediction_result.jpg", image)