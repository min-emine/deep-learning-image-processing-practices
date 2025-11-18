import cv2
import mediapipe as mp
import numpy as np

# Açı hesaplayan yardımcı fonksiyon
def calculate_angle(a, b, c):
    a = np.array(a)  # Kalça
    b = np.array(b)  # Diz
    c = np.array(c)  # Ayak Bileği

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Mediapipe tanımlamaları
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Video kaynağı 
cap = cv2.VideoCapture("squat_test1.avi")



frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width, frame_height))

# Değişkenler
counter = 0
stage = "down" # Başlangıç durumu
display_state = "Bekleniyor" # Ekrana yazılacak durum

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
 
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # İşleme
        results = pose.process(image)
        
        # RGB -> BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Koordinatları al
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            # Açıyı hesapla
            knee_angle = calculate_angle(hip, knee, ankle)
            
  
            
            # 1. Durum Göstergesi (Görsel Bilgi)
            if knee_angle > 160:
                display_state = "AYAKTA (Standing)"
            elif knee_angle < 90:
                display_state = "SQUAT (Squatting)"
            else:
                display_state = "HAREKET HALINDE"

            # 2. Sayaç Mantığı (Counter Logic)
            # Eğer açı 160'tan büyükse bacak düzdür (UP konumu)
            if knee_angle > 160:
                stage = "up"
                
            # Eğer açı 90'dan küçükse ve önceki durum "up" ise sayacı artır
            # Not: Genellikle squat kalkarken değil, tam eğilince veya tam kalkınca sayılır.
            # Burada 'down' durumuna geçince hazırlıyoruz, tekrar 'up' olunca sayacağız.
            if knee_angle < 90 and stage == "up":
                stage = "down"
                counter += 1
                print(f"Squat Sayısı: {counter}")
                
        except:
            pass
        
        # Görselleştirme - Bilgi Kutusu
        cv2.rectangle(image, (0,0), (350, 180), (245, 117, 16), -1) # -1 içini doldurur
        
        # Açı
        cv2.putText(image, 'ACI', (15,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(int(knee_angle)), (15,70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
        # Sayaç
        cv2.putText(image, 'SAYAC', (150,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (150,70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
        # Durum
        cv2.putText(image, 'DURUM', (15,110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, display_state, (15,150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

        # İskeleti Çiz
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                ) 
        
        # Sonucu kaydet 
        out.write(image)
        
        # Ekranda göster
        cv2.imshow('Mediapipe Feed', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
out.release() # Kayıt dosyasını serbest bırak
cv2.destroyAllWindows()