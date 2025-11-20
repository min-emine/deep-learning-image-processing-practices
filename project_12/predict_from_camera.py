#import libraries
import cv2 
import numpy as np
from tensorflow.keras.models import load_model#cnn modeli yukle


#modeli yukle

model=load_model('mnist_cnn_model.h5')


#kamerayi baslat
cap=cv2.VideoCapture(0)
print("Bir kagida siyah kalemle rakam yaz ve kameraya goster. Kameradan cikmak icin 'q' tusuna bas.")

#kameradan gelen goruntuleri tahmin et

while True:
    success, frame=cap.read()
    if not success:
        break

    #goruntuyu gri tonlamaya cevir
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #ROI(region of interest) belirle
    h,w=gray.shape
    box_size=200
    top_left=((w - box_size)//2, (h - box_size)//2)
    bottom_right=((w + box_size)//2, (h + box_size)//2)
    cv2.rectangle(frame, top_left, bottom_right, (0,255,0), 2)
    
    
    #roiden sayi tahmini yapma
    roi=gray[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    roi=cv2.resize(roi, (28,28))#yeniden boyutlandirma
    roi=roi.astype("float32")/255.0
    roi=np.expand_dims(roi, axis=-1)
    roi=np.expand_dims(roi, axis=0)
    #tahmin yap
    pred=model.predict(roi,verbose=0)#olasiliksal değerler[0.1,0.4,0.001,...,0.3]
    digit=np.argmax(pred)

    #tahmini ekrana yaz

    cv2.putText(frame,f"Tamin:{digit}", (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
   


    cv2.imshow("Tahmin ekrani:", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break