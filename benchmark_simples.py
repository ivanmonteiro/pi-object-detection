# USAGE
# python pi_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from multiprocessing import Process
from multiprocessing import Queue
import imutils
from imutils.video import FPS
import numpy as np
import argparse
import time
import cv2

# construct the argument parse 
parser = argparse.ArgumentParser(
    description='Script to run MobileNet-SSD object detection network ')
parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
args = parser.parse_args()

# load our serialized model from disk
print("[INFO] loading model...")
#net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
#net = cv2.dnn.readNetFromTensorflow('models/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb', 'models/ssd_mobilenet_v1_coco_2017_11_17/ssd_mobilenet_v1_coco_2017_11_17.pbtxt')
net = cv2.dnn.readNetFromTensorflow('models/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03/frozen_inference_graph.pb', 'models/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03/graph.pbtxt')

cap = cv2.VideoCapture(args.video)

time.sleep(2.0)
fps = FPS().start()

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it, and
    # grab its imensions
    ret, frame = cap.read()
    if frame is None:
        break

    frame = cv2.resize(frame,(300,300)) # resize frame for prediction

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
cap.release()
