import cv2
import numpy as np

from ultralytics import YOLO

# Initialize YOLOv8 model
model_path = "training/runs/detect/train7/weights/best.pt" #training/runs/detect/train/weights/best.pt
model = YOLO(model_path)




class Object():

    def __init__(self, object):
        self.screenWidth = 640
        self.screenHeight = 384

        self.points = self.getPoints(object)
        self.deltas = self.getDeltas(self.points)
        self.area = self.getArea(self.deltas)
        self.center = self.getCenter(self.deltas)
        self.proximity = self.getProximityCenter(self.center)
        self.distance = 1 # Placeholder for distance

        self.weight = self.calculateWeight(self.area, self.proximity, self.distance)

    def getPoints(object):
        #Top Left Corner on a Non-mirrored input
        x1 = object[0] #x val
        y1 = object[1] #y val

        #Bottom Right Corner on a Non-mirrored input
        x2 = object[2]
        y2 = object[3]
        return [x1, y1, x2, y2]

    def getDeltas(points):
        #Returns the delta (change) in x and y as an array
        deltX = (points[2] - points[0])
        deltY = (points[3] - points[1])
        return [deltX, deltY]

    def getCenter(deltas):
        #Returns the center of the object
        deltX = [deltas[0]]
        deltY = [deltas[1]]

        #Determine Midpoints

        midX = (deltX[0] / 2)
        midY = (deltY[0] / 2)
        return [midX, midY] # Returns the midpoint/center of the object

    def getArea(deltas):
        area = (deltas[0] * deltas[1])
        return area

    def getProximityCenter(self, midpoint): # Gets the magnitude of a midpoint to the center of the screen
        screenCenter = [self.screenWidth / 2, self.screenHeight / 2]
        x = midpoint[0] - screenCenter[0]
        y = midpoint[1] - screenCenter[1]

        z = np.sqrt(x**2 + y**2)
        return z

    def calculateWeight(area, proximity, distance): #Calculates weights
        weight = (area * 0.5) + (proximity * 1) + (distance * 2)
        return weight
    
    def compareWeight(self, f_weight):
        if self.weight > f_weight:
            return True
        else:
            return False

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
'''
Solving for 3D Measurements of an object
'''




def determineFocus(object, f_weight): #Determines what object is in focus
    '''
        This function compares two objects based off a weight system
        
        Weights:
        Area: x0.5
        Position on Screen: x1
        Distance: x2
    '''







def main():
    
    results = model(source=1, imgsz=640, return_outputs=True, conf=.60) #Starts inferencing

    while True:
        focusedObj = None
        focusedWeight = 0
        for resultDict in results:
            if resultDict:
                for object in resultDict['det']:
                    focused, weight = determineFocus(object, focusedWeight)
                    if focused: # If the object is determined to be focused
                        focusedObj = object
                        focusedWeight = weight
            else: # No results -- Dict doesn't exist
                print('No results')

if __name__ == '__main__':
    main()

