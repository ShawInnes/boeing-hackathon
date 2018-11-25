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

mode = '0'

def nothing(x):
    pass

cv2.namedWindow('image', cv2.WINDOW_GUI_EXPANDED)

def processYolo(frame):
	return frame

def processInfrared(frame):
	return cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

def processHarris(frame):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = np.float32(gray)
	dst = cv2.cornerHarris(gray, 2, 3, 0.04)
	kernel = np.ones((5, 5), np.uint8)
	dst = cv2.dilate(dst, kernel)
	frame[dst > 0.01 * dst.max()] = [0, 0, 255]
	return frame

# loop over frames from the video file stream
while True:
	(grabbed, frame) = vs.read()

	# check to see if we have reached the end of the video file
	if frame is None:
		break

	if mode == '1':
		frame = processHarris(frame)
	elif mode == '2':
		frame = processYolo(frame)
	elif mode == '3':
		frame = processInfrared(frame)

	cv2.imshow('image', frame)

	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	if key == ord("1"):
		mode = '1'
	if key == ord("2"):
		mode = '2'
	if key == ord("3"):
		mode = '3'

	# The following block will pause/unpause
	if (key == ord('p')):
		while ((cv2.waitKey(1) & 0xFF) != ord('p')):
			continue


# check to see if we need to release the video writer pointer
if writer is not None:
	writer.release()

# do a bit of cleanup
cv2.destroyAllWindows()
