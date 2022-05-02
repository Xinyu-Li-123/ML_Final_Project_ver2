import cv2 as cv

img_path = 'g:\\我的云端硬盘\\models\\pytorch-neural-style-transfer\\data\\content-images\\figures.jpg'
img = cv.imread(img_path)
print(img.shape, img.dtype)