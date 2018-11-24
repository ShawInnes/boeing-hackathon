# USAGE
# python test2.py --video race.mp4

# import the necessary packages

import numpy as np
import argparse
import cv2

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

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# if we are supposed to be writing a video to disk, initialize
	# the writer
	if args["output"] is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

	# check to see if we should write the frame to disk
	if writer is not None:
		writer.write(frame)

	out = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

	cv2.imshow('image', out)

	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break


# check to see if we need to release the video writer pointer
if writer is not None:
	writer.release()

# do a bit of cleanup
cv2.destroyAllWindows()
