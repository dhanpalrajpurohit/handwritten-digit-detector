import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import math
import tensorflow as tf
from keras.models import load_model
from keras import backend as K
from keras.models import load_model
model = tf.keras.models.load_model('D:\Openv\my_model.h5')
img  = cv2.imread('photo.jpg',0)
img_org  = cv2.imread('photo.jpg')
rows,cols = img.shape
img = cv2.GaussianBlur(img,(9,9),0)
ret, img = cv2.threshold(img, 100, 255, 0)
img = cv2.bitwise_not(img)
__, contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
for j, cnt in enumerate(contours):
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    hull = cv2.convexHull(cnt)
    k = cv2.isContourConvex(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    if (w > 1 and h > 50):
        cv2.rectangle(img_org, (x, y-10), (x + w+10, y + h+10), (0, 255, 0), 2)
        roi = img[y:y + h, x:x + w]
        org_size = 28
        img_size = 28
        rows, cols = roi.shape
        roi = img
        if rows > cols:
            factor = org_size / rows
            rows = org_size
            cols = int(round(cols * factor))
        else:
            factor = org_size / cols
            cols = org_size
            rows = int(round(rows * factor))
            img = cv2.resize(roi, (28,28))
            rows, cols = roi.shape[:2]

        img = cv2.resize(roi, (cols, rows))
        colsPadding = (int(math.ceil((img_size - cols) / 2.0)), int(math.floor((img_size - cols) / 2.0)))
        rowsPadding = (int(math.ceil((img_size - rows) / 2.0)), int(math.floor((img_size - rows) / 2.0)))
        img = np.lib.pad(img, (rowsPadding, colsPadding), 'constant')
        test_image = img.reshape(-1, 28, 28)
        pred = np.argmax(model.predict(test_image))
        print(pred)
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        def put_label(img, pred, x, y):
            font = cv2.FONT_HERSHEY_SIMPLEX
            rect_x = int(x) + 10
            rect_y = int(y) + 10
            cv2.rectangle(img_org, (rect_x, rect_y + 5), (rect_x + 35, rect_y - 35), (0, 0, 255), cv2.FILLED)
            cv2.putText(img_org, str(pred), (rect_x, rect_y), font, 1.5, (255, 0, 0), 1, cv2.LINE_AA)
            return img_org
        img_org = put_label(img, pred, x, y)
cv2.imshow("image1", img_org)
cv2.waitKey(0)
cv2.destroyAllWindows()

