import cv2
import os

# image_path = os.path.join('.','data','image.jpg')
# image_path = os.path.join('.','data','hand.png')
image_path = os.path.join('.','data','parking.jpg')
img  =  cv2.imread(image_path)
print(img.shape)
# cv2.imwrite(os.path.join('.','data','img_out.jpeg'),img)
# img = cv2.resize(img,(1000,750))
# print(img.shape)

k_size =5
img_blur = cv2.blur(img, (k_size,k_size))
# # img__gaus_blur = cv2.GaussianBlur(img, (k_size,k_size),5)
# img_median_blur  = cv2.medianBlur(img,k_size)
# cropped_img = img[50:1300,700:1800]

grey_img =cv2.cvtColor(img_blur,cv2.COLOR_BGR2GRAY)
print(grey_img.shape)

ret, thresh = cv2.threshold(grey_img,130,255, cv2.THRESH_BINARY_INV)
# thresh = cv2.blur(thresh, (10,10))
# ret, thresh = cv2.threshold(thresh,105,255, cv2.THRESH_BINARY)
counters, hirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# img_edge = cv2.Canny(img,200,500)
for cnt in counters:
   if cv2.contourArea(cnt) > 10:
        # cv2.drawContours(img, cnt, -1,(0,255,0),5 )

        x1,y1,w,h =  cv2.boundingRect(cnt)
        cv2.rectangle(img, (x1,y1), (x1+w,y1+h), (0,255, 0),1)

cv2.imshow('Image1', img)
cv2.imshow('Image', img_blur)
cv2.imshow('thresh', thresh)
# cv2.imshow('img_blur', img_blur)
# # cv2.imshow('img__gaus_blur', img__gaus_blur)
# cv2.imshow('img_median_blur', img_median_blur)

cv2.waitKey(0)