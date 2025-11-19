#import libraries
from operator import index
import cv2
import mediapipe as mp
import numpy as np

#yardımcı fonksiyonlar

#mediapipe face mesh modülünü başlatma
mp_face_mesh = mp.solutions.face_mesh
face_mash = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,refine_landmarks=True,
                                 min_detection_confidence=0.5, min_tracking_confidence=0.5)


#opencv ile kamera başlat
cap=cv2.VideoCapture(0)

#yüz mashinden kullanılacak önemli landmark indislerini al
LEFT_EYE=[159,145]
MOUTH=[13,14]
MOUTH_LEFT_RIGHT=[69,291]
LEFT_BROW=[65,158]


#göz açıklığı ve ağız açıklığına göre duygu tanıma fonksiyonu
def detect_emotion(landmarks, image_width, image_height): # <-- Düzeltme 1: w ve h eklendi
    def get_point(index):
        lm=landmarks[index]
        # Düzeltme 2: image_width ve image_height kullanıldı
        return np.array([int(lm.x*image_width), int(lm.y*image_height)])

#kaş ve göz noktaları(sol taraf)
    brow_point=get_point(65)
    eye_point=get_point(158)
    brow_lift=np.linalg.norm(eye_point - brow_point)


#dudak sol ve sağ noktaları
    mouth_left =get_point(69)
    mouth_right =get_point(291)
    mouth_width = np.linalg.norm(mouth_right - mouth_left)


    if brow_lift>25:
        return "saskin"
    elif mouth_width>100:
        return "mutlu"
    else:
        return "nötr"
#web cam üzerinden duygu tanıma

while True:
    sucess,frame =cap.read()
    if not sucess:
        break

    #görüntüyü rgbye çevir bunu mediapipe için yapıyoruz
    rgb_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mash.process(rgb_frame)

    #ekran boyutlari
    h,w,_ = frame.shape
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks=face_landmarks.landmark
            
            # Düzeltme 3: w ve h parametreleri gönderildi
            emotion=detect_emotion(landmarks, w, h)

            cv2.putText(frame, f'Duygu: {emotion}', (30,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

            #yüz mesh noktalarını çiz
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_tesselation_style())
            
    cv2.imshow("Canli mimik ve duygu takibi", frame)
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()