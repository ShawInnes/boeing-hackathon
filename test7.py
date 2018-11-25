#USAGE
# python test2.py --video race.mp4

# import the necessary packages
import numpy as np
import argparse
import cv2
import imutils
import time
#from PIL import Image

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
 help="path to input video file")
ap.add_argument("-o", "--output", type=str,
 help="path to optional output video file")
args = vars(ap.parse_args())

# create images to paste on
house = cv2.imread("house.png")

# initialize the video stream and output video writer
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(args["video"])
writer = None

def nothing(x):
   pass

cv2.namedWindow('image')

cv2.createTrackbar('hl','image',int(255.0 * 0.189),255,nothing)
cv2.createTrackbar('hh','image',int(255.0 * 0.378),255,nothing)
cv2.createTrackbar('sl','image',int(255.0 * 0.858),255,nothing)
cv2.createTrackbar('sh','image',int(255.0 * 1.000),255,nothing)
cv2.createTrackbar('vl','image',int(255.0 * 0.683),255,nothing)
cv2.createTrackbar('vh','image',int(255.0 * 1.000),255,nothing)

# loop over frames from the video file stream
while True:
 # grab the next frame from the video file
 (grabbed, frame) = vs.read()

 # check to see if we have reached the end of the video file
 if frame is None:
   break

 hsv = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR) #COLOR_BGR2HSV
 objectsMask = cv2.inRange(hsv, (0,200,0), (255,255,0))

 hl = cv2.getTrackbarPos('hl', 'image') / 255.0
 hh = cv2.getTrackbarPos('hh', 'image') / 255.0
 sl = cv2.getTrackbarPos('sl', 'image') / 255.0
 sh = cv2.getTrackbarPos('sh', 'image') / 255.0
 vl = cv2.getTrackbarPos('vl', 'image') / 255.0
 vh = cv2.getTrackbarPos('vh', 'image') / 255.0

 edged = cv2.Canny(objectsMask, 30, 150)
 contours = cv2.findContours(edged.copy(), cv2.RETR_TREE,
 	cv2.CHAIN_APPROX_SIMPLE)
 contours = contours[0] if imutils.is_cv2() else contours[1]
 contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

 coins = frame.copy()
 for (i, c) in enumerate(contours):
   rect = cv2.boundingRect(c)

   # exit loop if abnormal size is detected
   if rect[2] > 100 or rect[3] > 100 or rect[2] < 6 or rect[3] < 6:   # means that the cars/houses will detected, assuming approximate max size
      continue

   area = cv2.contourArea(c)

   # check for cars
   if rect[3] < 15 and rect[2] < 15:
        x, y, w, h = rect
        cv2.circle(coins, (x, y), int((w + h)/2), (232, 140, 83), 2)

    # check for houses
   if rect[3] > 60:
       x, y, w, h = rect
       pasteHouse = cv2.resize(house, (int(w/2), int(w/2)))
       coins[y:y+int(w/2), x:x+int(w/2)] = pasteHouse


 cv2.imshow('image', coins)

 key = cv2.waitKey(1) & 0xFF

 # if the `q` key was pressed, break from the loop
 if key == ord("q"):
   break

 time.sleep(0.01)



# check to see if we need to release the video writer pointer
if writer is not None:e
writer.release()

# do a bit of cleanup
cv2.destroyAllWindows()
