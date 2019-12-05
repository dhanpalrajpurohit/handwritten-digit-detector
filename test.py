# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import ndimage
# import math
# import tensorflow as tf
# from keras.models import load_model
# from keras import backend as K
# from keras.models import load_model
#
# model = tf.keras.models.load_model('MNIST-CNN.model')
# img  = cv2.imread('digitImage.jpg',0)
# img_org  = cv2.imread('digitImage.jpg')
#
# rows,cols = img.shape
# img = cv2.GaussianBlur(img,(9,9),0)
# ret, img = cv2.threshold(img, 100, 255, 0)
# img = cv2.bitwise_not(img)
# contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
# for j, cnt in enumerate(contours):
#     epsilon = 0.01 * cv2.arcLength(cnt, True)
#     approx = cv2.approxPolyDP(cnt, epsilon, True)
#     hull = cv2.convexHull(cnt)
#     k = cv2.isContourConvex(cnt)
#     x, y, w, h = cv2.boundingRect(cnt)
#     if (w > 1 and h > 50):
#         cv2.rectangle(img_org, (x, y-10), (x + w+10, y + h+10), (0, 255, 0), 2)
#         roi = img[y:y + h, x:x + w]
#         org_size = 28
#         img_size = 28
#         rows, cols = roi.shape
#         roi = img
#         if rows > cols:
#             factor = org_size / rows
#             rows = org_size
#             cols = int(round(cols * factor))
#         else:
#             factor = org_size / cols
#             cols = org_size
#             rows = int(round(rows * factor))
#             img = cv2.resize(roi, (28,28))
#             rows, cols = roi.shape[:2]
#         print(rows, cols)
#         img = cv2.resize(roi, (cols, rows))
#         colsPadding = ((abs(math.ceil((img_size - cols) / 2.0))),(abs(math.floor((img_size - cols) / 2.0))))
#         rowsPadding = (abs(math.ceil((img_size - rows) / 2.0)), abs(math.floor((img_size - rows) / 2.0)))
#         print(rowsPadding, colsPadding)
#         img = np.lib.pad(img, (rowsPadding, colsPadding), 'constant')
#         test_image = img.reshape(-1, 28, 28, 1)
#         pred = np.argmax(model.predict(test_image))
#         print(pred)
#         (x, y), radius = cv2.minEnclosingCircle(cnt)
#         def put_label(img, pred, x, y):
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             rect_x = int(x) + 10
#             rect_y = int(y) + 10
#             cv2.rectangle(img_org, (rect_x, rect_y + 5), (rect_x + 35, rect_y - 35), (0, 0, 255), cv2.FILLED)
#             cv2.putText(img_org, str(pred), (rect_x, rect_y), font, 1.5, (255, 0, 0), 1, cv2.LINE_AA)
#             return img_org
#         img_org = put_label(img, pred, x, y)
# cv2.imshow("image1", img_org)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#

from keras.models  import load_model
import numpy as np
import cv2
model = load_model("MNIST-CNN.model")
image = cv2.imread("image4.png")
image1 = cv2.imread("image4.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.GaussianBlur(image, (5, 5), 0)

ret, image = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
ctrs, hier = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(ctr) for ctr in ctrs]
for rect in rects:
    cv2.rectangle(image1, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
    leng = int(rect[3] * 1.6)
    pt1 = abs(int(rect[1] + rect[3] // 2 - leng // 2))
    pt2 = abs(int(rect[0] + rect[2] // 2 - leng // 2))
    roi = image[pt1:pt1+leng, pt2:pt2+leng]
    roi = cv2.resize(roi,(28, 28), interpolation=cv2.INTER_AREA)
    # roi = roi.reshape(1,784)
    roi = roi.reshape(-1,28, 28, 1)
    # roi=cv2.dilate(roi,(3,3))
    roi = np.array(roi, dtype='float32')
    roi /= 255
    pred_array = model.predict(roi)
    pred_array = np.argmax(pred_array)
    print('Result: {0}'.format(pred_array))
    cv2.putText(image1, str(pred_array), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)
cv2.imshow("Result4",image1)
cv2.imwrite("image4.jpg",image1)

cv2.waitKey(0)

#
# import cv2
# import numpy as np
# from time import perf_counter
# import pyautogui
# import imutils
# # from PIL import ImageGrab
# import pyscreenshot as ImageGrab
#
# while True:
#     img = pyautogui.screenshot()
#     frame = np.array(img)
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     cv2.imshow("screenshot",imutils.resize(frame, width=800,height=800))
#     if cv2.waitKey(1) == ord("q"):
#         break
#
# # make sure everything is closed when exited
# cv2.destroyAllWindows()
# out.release()