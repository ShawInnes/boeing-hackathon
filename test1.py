# import the necessary packages
import numpy as np
import argparse
import cv2

image_path = 'images/image1.png'
# Load an color image in grayscale
img = cv2.imread(image_path, 0)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

