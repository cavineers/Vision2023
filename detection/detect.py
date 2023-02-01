import cv2
import numpy as np
import math
import onnx
import onnxruntime as ort
import time

from ultralytics import YOLO
from utils import xywh2xyxy, nms, draw_detections


# Initialize YOLOv8 model





class angleSolver():
    
    def __init__(self, object):
        self.object = object.object
        self.classID = object.classID

        self.horizontalFOV = 0
        self.screenWidth = 640

        self.centerOfBox = object.getCenter()
        self.centerOfScreen = self.screenWidth/2

        self.angleToObj = self.solveAngle()
    
    def getMagToObject(self): #Gets the straight line object center of screen -> center of box
        return math.dist(self.centerOfBox[0], self.centerOfScreen)
    
    def solveAngle(self):
        degreesPerPixel = self.horizontalFOV / self.screenWidth
        distanceFromCenter = self.getMagToObject() # Distance from center of screen to center of box in pixels
        angle = distanceFromCenter * degreesPerPixel # Get Angle by Multiplying number of pixels by degrees per pixel
        return angle
    


    

        
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
class Object():

    def __init__(self, object):
        self.screenWidth = 640
        self.screenHeight = 384
        self.object = object
        self.classID = object[5]

        self.points = self.getPoints()
        self.deltas = self.getDeltas(self.points)
        self.area = self.getArea(self.deltas)
        self.center = self.getCenter(self.deltas)
        self.proximity = self.getProximityCenter(self.center)

        self.weight = self.calculateWeight()

    def getPoints(self):
        #Top Left Corner on a Non-mirrored input
        print(object)
        x1 = self.object[0] #x val
        y1 = self.object[1] #y val

        #Bottom Right Corner on a Non-mirrored input
        x2 = self.object[2]
        y2 = self.object[3]
        return [x1, y1, x2, y2]

    def getDeltas(self, points):
        #Returns the delta (change) in x and y as an array
        deltX = (points[2] - points[0])
        deltY = (points[3] - points[1])
        return [deltX, deltY]

    def getCenter(self, deltas):
        #Returns the center of the object
        deltX = [deltas[0]]
        deltY = [deltas[1]]

        #Determine Midpoints

        midX = (deltX[0] / 2)
        midY = (deltY[0] / 2)
        return [midX, midY] # Returns the midpoint/center of the object

    def getArea(self, deltas):
        area = (deltas[0] * deltas[1])
        return area

    def getProximityCenter(self, midpoint): # Gets the magnitude of a midpoint to the center of the screen
        screenCenter = [self.screenWidth / 2, self.screenHeight / 2]
        x = midpoint[0] - screenCenter[0]
        y = midpoint[1] - screenCenter[1]

        z = np.sqrt(x**2 + y**2)
        return z

    def calculateWeight(self): #Calculates weights
        weight = (self.area / 0.001) + (self.proximity * 1)
        #print(f'{(self.area * 0.001)} + {(self.proximity * 1)} + {(self.distance * 2)}')
        return weight
    
    def compareWeight(self, f_weight):
        if self.weight > f_weight:
            return True
        else:
            return False

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


class YOLOv8:

    def __init__(self, path, conf_thres=0.7, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        # Initialize model
        self.initialize_model(path)

    def __call__(self, image):
        return self.detect_objects(image)

    def initialize_model(self, path):
        self.session = ort.InferenceSession(path)
        # Get model info
        self.get_input_details()
        self.get_output_details()


    def detect_objects(self, image):
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        self.boxes, self.scores, self.class_ids = self.process_output(outputs)

        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor


    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        # print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_output(self, output):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):

        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):

        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha)

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]




class imageHandler():
    def __init__(self, img):
        self.img = img
        self.imgHeight = img.shape[0]
        self.imgWidth = img.shape[1]
        self.processedImg = self.preProcess()
        
    def preProcess(self):
        self.processedImg = cv2.resize(self.img, (640, 640))
        
class cameraHandler():
    def __init__(self, src):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
    
    def getFrame(self):
        ret, frame = self.cap.read()
        return frame




if __name__ == '__main__':
    cam = cameraHandler(1)
    while True:
        model_path = "detection/finalweights/model.onnx"

        # Initialize YOLOv8 object detector
        yolov8Detector = YOLOv8(model_path, conf_thres=0.6, iou_thres=0.5)

        img = cam.getFrame()
        if type(img) == type(None):
            continue
        # Detect Objects
        yolov8Detector(img)

        # Draw detections
        combined_img = yolov8Detector.draw_detections(img)
        cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
        cv2.imshow("Output", combined_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


