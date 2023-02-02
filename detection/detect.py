import cv2
import numpy as np
import math

from ultralytics import YOLO

# Initialize YOLOv8 model
model_path = "training/runs/detect/train7/weights/best.pt" #training/runs/detect/train/weights/best.pt
model = YOLO(model_path)


class angleSolver():
    
    def __init__(self, object):
        self.object = object.object
        self.classID = object.classID

        self.horizontalFOV = 70.4
        self.screenWidth = 384

        self.centerOfBox = object.getCenter()
        self.centerOfScreen = self.screenWidth/2

        self.angleToObj = self.solveAngle()
    
    def getMagToObject(self): #Gets the straight line object center of screen -> center of box
        mag = math.fabs(self.centerOfScreen - self.centerOfBox[0])
        return mag
    
    def solveAngle(self):
        degreesPerPixel = self.horizontalFOV / self.screenWidth
        distanceFromCenter = self.getMagToObject() # Distance from center of screen to center of box in pixels
        print(distanceFromCenter)
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
        self.center = self.getCenter()
        self.proximity = self.getProximityCenter(self.center)

        self.weight = self.calculateWeight()

    def getPoints(self):
        #Top Left Corner on a Non-mirrored input

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

    def getCenter(self):
        #Returns the center of the object
        deltX = [self.deltas[0]]
        deltY = [self.deltas[1]]

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
        weight = (math.sqrt(self.area)) + (self.proximity * 1)
        #print(f'{(self.area * 0.001)} + {(self.proximity * 1)} + {(self.distance * 2)}')
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







def main():
    
    results = model(source=0, imgsz=640, return_outputs=True, conf=.60) #Starts inferencing
    while True:
        print("Initilizing")
        for dict in results:
            focusedObj = None # Placeholder for focused object
            if not dict: # If no results, continue
                print("No Results") 
                continue 
            for object in dict['det']:
                m_object = Object(object)
                if not focusedObj: #See if there is a focused object if not set it to the first object
                    focusedObj = m_object
                elif m_object.compareWeight(focusedObj.weight):
                    focusedObj = m_object
                else:
                    continue
                angleSolved = angleSolver(focusedObj)
                print(f'Angle to object {angleSolved.angleToObj}')
                    
            #print("Focused Object: ", focusedObj.classID)
        
                    

if __name__ == '__main__':
    main()

