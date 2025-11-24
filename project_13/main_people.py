import cv2
import numpy as np
from ultralytics import YOLO

# Model yükleme
model=YOLO("yolov8n.pt")

# Video dosyası açılması
cap=cv2.VideoCapture("2.mp4")

# 1 frame oku ve video çalışıyor mu diye kontrol et
success , frame = cap.read()
if not success:
    exit("Video açılamadı")

# Yeniden boyutlandır
frame=cv2.resize(frame,(0,0),fx=0.6,fy=0.6)
frame_height, frame_width = frame.shape[:2]

# --- Video Kaydetme İçin Yeni Kısım Başlangıcı ---
# Kodek (Codec) belirleme ve VideoWriter nesnesi oluşturma
# Dört karakterli bir kodek kodu (FourCC) belirlenir. 'mp4v' yaygın kullanılan bir MP4 kodeğidir.
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
# Çıkış dosyası adı, kodek, FPS (giriş videosunun FPS'si kullanılabilir, burada varsayılan 20), ve frame boyutu
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height)) 
# --- Video Kaydetme İçin Yeni Kısım Sonu ---

# Ortaya dikey çizgi
line_x = int(frame_width*0.5) 
offset=10 # Çizgi kalınlığı için (kodda kullanılmıyor ama ayarlanmış)

# Sayaclar
giren=0
cikan=0
counted_ids=set()
person_last_x={}

# Yolo ile insan sayma
while True:
    success, frame =cap.read()
    if not success:
        break

    frame=cv2.resize(frame,(0,0),fx=0.6,fy=0.6)
    
    # Takip ve tespit
    results=model.track(frame,persist=True, stream=False,conf=0.25,iou=0.3,tracker="bytetrack.yaml")

    if results[0].boxes.id is not None: # Eğer takip edilen obje varsa
        ids=results[0].boxes.id.int().tolist() # Tüm id'leri al
        classes = results[0].boxes.cls.int().tolist() # Tüm class'ları al
        xyxy=results[0].boxes.xyxy # Koordinatları al

        for i,box in enumerate(xyxy):
            cls_id=classes[i]
            track_id=ids[i]
            class_name=model.names[cls_id]
            
            if class_name!="person":
                continue
                
            x1,y1,x2,y2=map(int,box)
            cx=int((x1+x2)/2)
            cy=int((y1+y2)/2)

            previous_x=person_last_x.get(track_id,None)
            person_last_x[track_id]=cx
            
            if previous_x is not None:
                # Çizgiyi geçme kontrolü (Sol'dan Sağ'a = Giren, Sağ'dan Sol'a = Çıkan kabul edilmiş)
                if previous_x > line_x >= cx and track_id not in counted_ids: # Sağdan Sola (Çıkan)
                    cikan += 1
                    counted_ids.add(track_id)
                elif previous_x < line_x <= cx and track_id not in counted_ids: # Soldan Sağ'a (Giren)
                    giren += 1
                    counted_ids.add(track_id)
                    
            # Çizimler
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f" ID:{track_id}",(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
            cv2.circle(frame,(cx,cy),4,(255,0,0),-1)

    # Çizgi ve sayaçları çizme
    cv2.line(frame,(line_x,0),(line_x,frame_height),(0,0,255),2)

    cv2.putText(frame,f"Giren(saga): {giren}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
    cv2.putText(frame,f"Cikan(sol): {cikan}",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

    # --- Video Kaydetme Kısımı ---
    out.write(frame) # İşlenmiş frame'i dosyaya yaz

    # Görüntüleme
    cv2.imshow("avm yon takibi", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release() 
out.release() # VideoWriter'ı serbest bırakmak önemli!
cv2.destroyAllWindows()