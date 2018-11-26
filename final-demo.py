# USAGE
# python test2.py --video race.mp4

# import the necessary packages

import numpy as np
import argparse
import cv2
import imutils
import time

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,	help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="csrt",	help="OpenCV object tracker type")
args = vars(ap.parse_args())

print("[INFO] starting video stream...")
vs = cv2.VideoCapture(args["video"])

framecount = 0
mode = '0'

OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"boosting": cv2.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.TrackerTLD_create,
	"medianflow": cv2.TrackerMedianFlow_create,
	"mosse": cv2.TrackerMOSSE_create
}

# initialize OpenCV's special multi-object tracker
trackers = cv2.MultiTracker_create()

house = cv2.imread("house.png")

def nothing(x):
    pass

cv2.namedWindow('image', cv2.WINDOW_GUI_EXPANDED)

cv2.createTrackbar('hl', 'image', 1, 255, nothing)
cv2.createTrackbar('sl', 'image', 1, 255, nothing)
cv2.createTrackbar('vl', 'image', 1, 255, nothing)
cv2.createTrackbar('hh', 'image', 255, 255, nothing)
cv2.createTrackbar('sh', 'image', 255, 255, nothing)
cv2.createTrackbar('vh', 'image', 255, 255, nothing)

def processYolo(frame):
	return frame

def processHouse(frame):
	hsv = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)  # COLOR_BGR2HSV

	objectsMask = cv2.inRange(hsv, (0, 200, 0), (255, 255, 0))
	edged = cv2.Canny(objectsMask, 30, 150)
	contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours = contours[0] if imutils.is_cv3() else contours[1]

	copy = frame.copy()
	for (i, c) in enumerate(contours):
		rect = cv2.boundingRect(c)

		# exit loop if abnormal size is detected
		if rect[2] > 100 or rect[3] > 100 or rect[2] < 6 or rect[
			3] < 6:  # means that the cars/houses will detected, assuming approximate max size
			continue

		# check for cars
		if rect[3] < 15 and rect[2] < 15:
			x, y, w, h = rect
			cv2.circle(copy, (x, y), int((w + h) / 2), (232, 140, 83), 2)

		# check for houses
		if rect[3] > 60:
			x, y, w, h = rect
			pasteHouse = cv2.resize(house, (int(w / 2), int(w / 2)))
			copy[y:y + int(w / 2), x:x + int(w / 2)] = pasteHouse

	return copy

def processInfrared(frame):
	return cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

def processROI(frame):
	(success, boxes) = trackers.update(frame)

	# loop over the bounding boxes and draw then on the frame
	for box in boxes:
		(x, y, w, h) = [int(v) for v in box]
		height, width, channel = frame.shape
		if (x > 0) and (y > 0) and (x + w < width) and (y + h < height):
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	return frame

def processEdges(frame):
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	hl = cv2.getTrackbarPos('hl', 'image')
	hh = cv2.getTrackbarPos('hh', 'image')
	sl = cv2.getTrackbarPos('sl', 'image')
	sh = cv2.getTrackbarPos('sh', 'image')
	vl = cv2.getTrackbarPos('vl', 'image')
	vh = cv2.getTrackbarPos('vh', 'image')

	blurred = cv2.GaussianBlur(hsv, (11, 11), 0)
	mask = cv2.inRange(blurred, (hl, sl, vl), (hh, sh, vh))
	otherMask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
	bitwiseAnd = cv2.bitwise_and(frame, otherMask)

	out = cv2.cvtColor(bitwiseAnd, cv2.COLOR_HSV2BGR)

	return out

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
		frame = processInfrared(frame)
	elif mode == '3':
		frame = processEdges(frame)
	elif mode == '4':
		frame = processHouse(frame)

	frame = processROI(frame)

	framecount = framecount + 1

	if framecount > 661 and framecount < 761:
		cv2.putText(frame, 'AUSSIE WORLD!!!!', (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, 8)

	cv2.imshow('image', frame)

	key = cv2.waitKey(1) & 0xFF

	if key == ord("1"):
		mode = '1'
	if key == ord("2"):
		mode = '2'
	if key == ord("3"):
		mode = '3'
	if key == ord("4"):
		mode = '4'
	if key == ord("5"):
		mode = '5'
	if key == ord("q"):
		break

	if key == ord("s"):
		# select the bounding box of the object we want to track (make
		# sure you press ENTER or SPACE after selecting the ROI)
		box = cv2.selectROI("image", frame, fromCenter=False, showCrosshair=True)

		# create a new object tracker for the bounding box and add it
		# to our multi-object tracker
		tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
		trackers.add(tracker, frame, box)

	# The following block will pause/unpause
	if (key == ord('p')):
		while ((cv2.waitKey(1) & 0xFF) != ord('p')):
			continue

# do a bit of cleanup
cv2.destroyAllWindows()
