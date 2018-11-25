#USAGE
# python test2.py --video race.mp4

# import the necessary packages

import numpy as np
import argparse
import cv2
import imutils

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
 help="path to input video file")
ap.add_argument("-o", "--output", type=str,
 help="path to optional output video file")
args = vars(ap.parse_args())

# initialize the video stream and output video writer
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(args["video"])
writer = None

def nothing(x):
   pass

cv2.namedWindow('image')

cv2.createTrackbar('hl','image',int(255.0 * 0.191),255,nothing)
cv2.createTrackbar('hh','image',int(255.0 * 0.490),255,nothing)
cv2.createTrackbar('sl','image',int(255.0 * 0.081),255,nothing)
cv2.createTrackbar('sh','image',int(255.0 * 1.000),255,nothing)
cv2.createTrackbar('vl','image',int(255.0 * 0.108),255,nothing)
cv2.createTrackbar('vh','image',int(255.0 * 0.629),255,nothing)

# loop over frames from the video file stream
while True:
 # grab the next frame from the video file
 (grabbed, frame) = vs.read()

 # check to see if we have reached the end of the video file
 if frame is None:
   break

 hsv = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR) #COLOR_BGR2HSV
 #hsv = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

 # if we are supposed to be writing a video to disk, initialize
 # the writer
 if args["output"] is not None and writer is None:
   fourcc = cv2.VideoWriter_fourcc(*"MJPG")
   writer = cv2.VideoWriter(args["output"], fourcc, 30,
     (frame.shape[1], frame.shape[0]), True)

 # check to see if we should write the frame to disk
 if writer is not None:
   writer.write(frame)

 hl = cv2.getTrackbarPos('hl', 'image')
 sl = cv2.getTrackbarPos('sl', 'image')
 vl = cv2.getTrackbarPos('vl', 'image')
 hh = cv2.getTrackbarPos('hh', 'image')
 sh = cv2.getTrackbarPos('sh', 'image')
 vh = cv2.getTrackbarPos('vh', 'image')

# test it out for mask and houses, purple mask, and then find green!
 blurred = cv2.GaussianBlur(hsv, (11, 11), 0)
 mask = cv2.inRange(blurred, (hl, sl, vl), (hh, sh, vh))
 otherMask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

 # show the output frame
 bitwiseAnd = cv2.bitwise_and(frame, otherMask)
 out = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

 edged = cv2.Canny(blurred, 30, 150)
 contours = cv2.findContours(edged.copy(), cv2.RETR_TREE,
 	cv2.CHAIN_APPROX_SIMPLE)
 contours = contours[0] if imutils.is_cv2() else contours[1]
 #additionalMask, contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


 coins = frame.copy()
 for (i, c) in enumerate(contours):
   rect = cv2.boundingRect(c)
   if rect[2] > 100 or rect[3] > 100:   # means that the cars/houses will detected, assuming approximate max size
     continue

   area = cv2.contourArea(c)
   # if area > 3000 and area < 10000:
   x, y, w, h = rect
   #to separate houses from cars, can distinguish area and draw a circle around the car instead
   cv2.rectangle(coins, (x, y), (x + w, y + h), (0, 255, 0), 2)

 cv2.imshow('image', coins) #coins

 key = cv2.waitKey(1) & 0xFF

 # if the `q` key was pressed, break from the loop
 if key == ord("q"):
   break


# check to see if we need to release the video writer pointer
if writer is not None:e
writer.release()

# do a bit of cleanup
cv2.destroyAllWindows()
