import cv2
import numpy as np
from ultralytics import YOLO


def get_line_side(x,y,line_start,line_end):
    return np.sign((line_end[0]-line_start[0])*(y-line_start[1]) - 
                   (line_end[1]-line_start[1])*(x-line_start[0]))


model=YOLO("yolov8n.pt")
cap=cv2.VideoCapture("IMG_5268.MOV")

success , frame = cap.read()
if not success:
    exit("acilmadi")

frame=cv2.resize(frame,(0,0),fx=0.6,fy=0.6)
frame_height, frame_width = frame.shape[:2]

#çapraz çizgi tanımla

line_start=(int(frame_height*0.5),frame_height)
line_end=(frame_width, int(frame_width*0.2))


#obje tipi
counts={"car":0, "truck":0, "bus":0, "motorcycle":0, "bicycle":0}
counted_ids=set()
object_last_side={}




while True:
    success , frame = cap.read()
    if not success:
         exit("acilmadi")

    frame=cv2.resize(frame,(0,0),fx=0.6,fy=0.6) 

#tracking ve tespit
    results=model.track(frame,persist=True, stream=False,conf=0.5,iou=0.5,tracker="bytetrack.yaml")

    if results[0].boxes.id is not None:#eger takip edilen obje varsa
        ids=results[0].boxes.id.int().tolist()#tüm id leri al
        classes = results[0].boxes.cls.int().tolist()#tüm class ları al
        xyxy=results[0].boxes.xyxy#koordinatları al

        for i,box in enumerate(xyxy):
            cls_id=classes[i]
            track_id=ids[i]
            class_name=model.names[cls_id]
            if class_name not in counts:
                continue
            x1,y1,x2,y2=map(int,box)
            cx=int((x1+x2)/2)
            cy=int((y1+y2)/2)

            current_side=get_line_side(cx,cy,line_start,line_end)#aracın çizginin hangi tarafında olduğunu al
            previous_side=object_last_side.get(track_id,None)#önceki tarafı al
            object_last_side[track_id]=current_side#şu anki tarafı kaydet
            
            if previous_side is not None and previous_side != current_side:
                if track_id not in counted_ids:
                    counted_ids.add(track_id)
                    counts[class_name] += 1
                    
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            
            cv2.putText(frame,f"{class_name} ID:{track_id}",(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
            cv2.circle(frame,(cx,cy),4,(255,0,0),-1)    

            



    #line çiz    
    cv2.line(frame, line_start, line_end, (0,0,255), 2)

    #sayacları göster
    y_offset=30
    for cls,count in counts.items():
        text=f"{cls}: {count}"
        cv2.putText(frame,text,(10,y_offset),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
        y_offset +=30




    cv2.imshow("arac takip ve sayim", frame)
    #çıktıyı kaydet
    cv2.imwrite("output_frame.jpg", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




cap.release()
cv2.destroyAllWindows()