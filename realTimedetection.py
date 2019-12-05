#import required library
from keras.models  import load_model
import numpy as np
import cv2

#load your model here MNIST-CNN.model
model = load_model("MNIST-CNN.model")

#save your result video in .avi form
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output2.avi', fourcc, 20.0, (640,480))

#load  video
cap = cv2.VideoCapture('video2.mp4')
while(True):
    #read frame frame from video
    ret, image = cap.read()

    #perform basic operation to smooth image
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    #find threshold
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    #find contours and draw contours
    ctrs, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image,ctrs,-1,(255,255,0),2)
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    for rect in rects:
        x,y,w,h = rect
        if  h > 50 and h < 300  or w > 10 :
            #draw rectangel on image
            cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
            leng = int(rect[3] * 1.6)
            pt1 = abs(int(rect[1] + rect[3] // 2 - leng // 2))
            pt2 = abs(int(rect[0] + rect[2] // 2 - leng // 2))
            roi = img[pt1:pt1+leng, pt2:pt2+leng]
            roi = cv2.resize(roi,(28, 28), interpolation=cv2.INTER_AREA)
            #resize image
            roi = roi.reshape(-1,28, 28, 1)
            roi = np.array(roi, dtype='float32')
            roi /= 255
            pred_array = model.predict(roi)
            pred_array = np.argmax(pred_array)
            print('Result: {0}'.format(pred_array))
            cv2.putText(image, str(pred_array), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)
    #show frame
    cv2.imshow("Result",image)
    out.write(image)
    k = cv2.waitKey(27)
    if k==27:
        break
cv2.destroyAllWindows()
cap.release()
out.release()