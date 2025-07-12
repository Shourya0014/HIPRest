import cv2
import os
from glob import glob

input_dir = r'D:\intel\Restormer\Motion_Deblurring\Datasets\test\GoPro\input'
output_dir = r'D:\intel\Restormer\Motion_Deblurring\Datasets\test\GoPro\input_resized'
os.makedirs(output_dir, exist_ok=True)

resize_to = (640, 360)  # Width, Height

for file in glob(os.path.join(input_dir, '*.png')):
    img = cv2.imread(file)
    img_resized = cv2.resize(img, resize_to, interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(output_dir, os.path.basename(file)), img_resized)
