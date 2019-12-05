#import required library
from keras.models  import load_model
import numpy as np
import cv2

#load your model
model = load_model("MNIST-CNN.model")

#load your image to recognit image
image = cv2.imread("image4.png")
image1 = cv2.imread("image4.png")

#perform some basic operation to smooth image
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.GaussianBlur(image, (5, 5), 0)

#find threshold image
ret, image = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#find contours on image and draw it.
ctrs, hier = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(ctr) for ctr in ctrs]
for rect in rects:
    #draw rectangle on image using contours
    cv2.rectangle(image1, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
    leng = int(rect[3] * 1.6)
    pt1 = abs(int(rect[1] + rect[3] // 2 - leng // 2))
    pt2 = abs(int(rect[0] + rect[2] // 2 - leng // 2))
    #resize image
    roi = image[pt1:pt1+leng, pt2:pt2+leng]
    roi = cv2.resize(roi,(28, 28), interpolation=cv2.INTER_AREA)

    #reshape your image according to your model
    roi = roi.reshape(-1,28, 28, 1)
    roi = np.array(roi, dtype='float32')
    roi /= 255
    #to perform prediction on your image
    pred_array = model.predict(roi)
    pred_array = np.argmax(pred_array)

    #print result
    print('Result: {0}'.format(pred_array))

    #print text on your image
    cv2.putText(image1, str(pred_array), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)
    #show your image
cv2.imshow("Result4",image1)

#save your image
cv2.imwrite("image4.jpg",image1)

cv2.waitKey(0)
