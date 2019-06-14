# USAGE
# python pi_deep_learning.py --prototxt models/bvlc_googlenet.prototxt --model models/bvlc_googlenet.caffemodel --labels synset_words.txt --image images/barbershop.png
# python pi_deep_learning.py --prototxt models/squeezenet_v1.0.prototxt --model models/squeezenet_v1.0.caffemodel --labels synset_words.txt --image images/barbershop.png

# import the necessary packages
import numpy as np
import argparse
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-p", "--prototxt", required=False,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=False,
	help="path to Caffe pre-trained model")
ap.add_argument("-l", "--labels", required=False,
	help="path to ImageNet labels (i.e., syn-sets)")
args = vars(ap.parse_args())

NET_INPUT_SIZE = 128

# load the class labels from disk
#rows = open(args["labels"]).read().strip().split("\n")
#classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
CLASSES = {0: 'background',
              1: 'pessoa', 2: 'veiculo', 3: 'veiculo', 4: 'veiculo', 5: 'airplane', 6: 'veiculo',
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

# load the input image from disk
image = cv2.imread(args["image"])

image_resized = cv2.resize(image, (NET_INPUT_SIZE, NET_INPUT_SIZE))
(fH, fW) = image.shape[:2]
# our CNN requires fixed spatial dimensions for our input image(s)
# so we need to ensure it is resized to 224x224 pixels while
# performing mean subtraction (104, 117, 123) to normalize the input;
# after executing this command our "blob" now has the shape:
# (1, 3, 224, 224)
#blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))
blob = cv2.dnn.blobFromImage(image_resized, size=(NET_INPUT_SIZE, NET_INPUT_SIZE), swapRB=True, crop=False)
# load our serialized model from disk
print("[INFO] loading model...")
#net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
net = cv2.dnn.readNetFromTensorflow('models/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03/frozen_inference_graph.pb', 'models/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03/graph.pbtxt')

# set the blob as input to the network and perform a forward-pass to
# obtain our output classification
net.setInput(blob)
start = time.time()
detections = net.forward()
end = time.time()
print("[INFO] classification took {:.5} seconds".format(end - start))

# sort the indexes of the probabilities in descending order (higher
# probabilitiy first) and grab the top-5 predictions
#detections = preds.reshape((1, len(classes)))
#idxs = np.argsort(preds[0])[::-1][:5]
if detections is not None:
    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated
        # with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence`
        # is greater than the minimum confidence
        if confidence < 0.6:
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

        #if idx not in VEICULO_CLASSES and idx not in PEDESTRE_CLASSES:
        #    print(label)
        
        cv2.rectangle(image, (startX, startY), (endX, endY),
            (255,255,255), 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image, label, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
"""
# loop over the top-5 predictions and display them
for (i, idx) in enumerate(idxs):
	# draw the top prediction on the input image
	if i == 0:
		text = "Label: {}, {:.2f}%".format(classes[idx],
			preds[0][idx] * 100)
		cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX,
			0.7, (0, 0, 255), 2)

	# display the predicted label + associated probability to the
	# console	
	print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1,
		classes[idx], preds[0][idx]))
"""

# display the output image
cv2.imshow("Image", image)
cv2.waitKey(0)