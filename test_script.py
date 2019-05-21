from keras.models  import load_model
import numpy as np
import cv2
model = load_model("digitcnnmodel.model") # open saved model/weights from .h5 file
image = cv2.imread("photo_1.jpg")
image1 = cv2.imread("photo_1.jpg")

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.GaussianBlur(image, (5, 5), 0)
ret, image = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY_INV)
ctrs, hier = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(ctr) for ctr in ctrs]
for rect in rects:
    cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = image[pt1:pt1+leng, pt2:pt2+leng]
    roi = cv2.resize(roi,(28,28),interpolation=cv2.INTER_AREA)
    roi = roi.reshape(1,784)
    roi=cv2.dilate(roi,(3,3))



    roi = np.array(roi, dtype='float32')
    roi /= 255
    pred_array = model.predict(roi)
    pred_array = np.argmax(pred_array)
    #result = model.predict[np.argmax(roi)]
    #score = float("%0.2f" % (max(pred_array[0]) * 100))
    print('Result: {0}'.format(pred_array))
    cv2.putText(image1, str(pred_array), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)    #return result, score
    cv2.imshow("Result",image1)
cv2.waitKey(0)
