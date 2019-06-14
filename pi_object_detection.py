# USAGE
# python pi_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from imutils.video import FileVideoStream
from multiprocessing import Process
from multiprocessing import Queue
import numpy as np
import argparse
import imutils
import time
import cv2

NET_INPUT_SIZE = 128
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
ap.add_argument("--prototxt", required=False,
    help="path to Caffe 'deploy' prototxt file",
	default='models/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03/graph.pbtxt')
ap.add_argument("--model", required=False,
    help="path to Caffe pre-trained model",
	default='models/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03/frozen_inference_graph.pb')
ap.add_argument("--confidence", type=float, default=0.55,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

def classify_frame(net, inputQueue, outputQueue):
    # keep looping
    while True:
        # check to see if there is a frame in our input queue
        if not inputQueue.empty():
            # grab the frame from the input queue, resize it, and
            # construct a blob from it
            frame = inputQueue.get()
            #frame = cv2.resize(frame, (NET_INPUT_SIZE, NET_INPUT_SIZE))
            #frame_resized = imutils.resize(frame, width=NET_INPUT_SIZE)
            #blob = cv2.dnn.blobFromImage(frame, 0.007843,
            #    (300, 300), 127.5)
            blob = cv2.dnn.blobFromImage(frame, size=(NET_INPUT_SIZE, NET_INPUT_SIZE), swapRB=True, crop=False)
            # set the blob as input to our deep learning object
            # detector and obtain the detections
            net.setInput(blob)
            detections = net.forward()

            # write the detections to the output queue
            outputQueue.put(detections)

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
"""
CLASSES = ["background", "aviao", "bicicleta", "passaro", "barco",
    "garrafa", "onibus", "carro", "gato", "cadeira", "vaca", "mesa",
    "cachorro", "cavalo", "moto", "pessoa", "planta", "ovelha",
    "sofa", "trem", "tv"]

CLASSES_FILTRAR = ["onibus", "carro", "moto", "bicicleta", "pessoa"]
"""
"""
CLASSES = { 0: 'background',
    1: 'pessoa', 2: 'bicycle', 3: 'carro', 4: 'moto',
    5: 'bottle', 6: 'onibus', 7: 'car', 8: 'carro', 9: 'chair',
    10: 'semaforo', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor', 21: 'tvmonitor', 22: 'tvmonitor', 23: 'tvmonitor' }
"""
CLASSES = {0: 'background',
              1: 'pessoa', 2: 'bicycle', 3: 'veiculo', 4: 'motorcycle', 5: 'airplane', 6: 'veiculo',
              7: 'train', 8: 'veiculo', 9: 'veiculo', 10: 'traffic light', 11: 'fire hydrant',
              13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
              18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
              24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
              32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
              37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
              41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
              46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
              51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
              56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
              61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
              67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
              75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
              80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
              86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

VEICULO_CLASSES = [2, 3, 4, 6]
VEICULO_COR = (230, 230, 0)
PEDESTRE_CLASSES = [1]
PEDESTRE_COR = (255, 40, 255)

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

for classe_index in VEICULO_CLASSES:
	COLORS[classe_index] = VEICULO_COR

for classe_index in PEDESTRE_CLASSES:
	COLORS[classe_index] = VEICULO_COR
print(COLORS)

# load our serialized model from disk
print("[INFO] loading model...")
#net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
#net = cv2.dnn.readNetFromTensorflow('models/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb', 'models/ssd_mobilenet_v1_coco_2017_11_17/ssd_mobilenet_v1_coco_2017_11_17.pbtxt')
net = cv2.dnn.readNetFromTensorflow('models/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03/frozen_inference_graph.pb', 'models/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03/graph.pbtxt')

# initialize the input queue (frames), output queue (detections),
# and the list of actual detections returned by the child process
inputQueue = Queue(maxsize=1)
outputQueue = Queue(maxsize=1)
detections = None

# construct a child process *indepedent* from our main process of
# execution
print("[INFO] starting process...")
p = Process(target=classify_frame, args=(net, inputQueue,
    outputQueue,))
p.daemon = True
p.start()

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
# Open video file or capture device. 
if args["video"]:
	vs = FileVideoStream(args["video"]).start()
else:
	#vs = VideoStream(usePiCamera=True).start()
    vs = VideoStream(src=0).start()

time.sleep(2.0)
fps = FPS().start()

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it, and
    # grab its imensions
    frame = vs.read()

	#exit if video ended
    if frame is None:
        break

    frame = imutils.resize(frame, width=300)
    frame_for_net = cv2.resize(frame, (NET_INPUT_SIZE, NET_INPUT_SIZE))
    (fH, fW) = frame.shape[:2]

    # if the input queue *is* empty, give the current frame to
    # classify
    if inputQueue.empty():
        inputQueue.put(frame_for_net)

    # if the output queue *is not* empty, grab the detections
    if not outputQueue.empty():
        detections = outputQueue.get()

    # check to see if our detectios are not None (and if so, we'll
    # draw the detections on the frame)
    if detections is not None:
        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated
            # with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence`
            # is greater than the minimum confidence
            if confidence < args["confidence"]:
                continue

            # otherwise, extract the index of the class label from
            # the `detections`, then compute the (x, y)-coordinates
            # of the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            dims = np.array([fW, fH, fW, fH])
            box = detections[0, 0, i, 3:7] * dims
            (startX, startY, endX, endY) = box.astype("int")

            # draw the prediction on the frame
            #if CLASSES[idx] in CLASSES_FILTRAR:
            label = "{}: {:.2f}%".format(CLASSES[idx],
                confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

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
vs.stop()
