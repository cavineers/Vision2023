import cv2
import numpy as np

from ultralytics import YOLO

# Initialize YOLOv8 model
model_path = "training/runs/detect/train7/weights/best.pt" #training/runs/detect/train/weights/best.pt
model = YOLO(model_path)


class angleSolver():
    
    def __init__(self, object):
        self.object = object.object
        self.classID = object.classID
        self.points = object.getPoints()
        self.imgPoints = np.array(self.getAllPoints())
        self.cubeDimensions = [0.24, 0.24, 0.24] #in meters Length, Width, Height
        self.coneDimensions = [0.21, 0.21, 0.33] #in meters Length, Width, Height
        self.rvec, self.tvec = self.solvePNP()
    
    def getAllPoints(self): #From a two points return the missing points of a rectangle
        bottomLeft = [self.points[0], self.points[1]] #0X1, 1Y1, 2X2, 3Y2
        bottomRight = [self.points[2], self.points[1]]
        topLeft = [self.points[0], self.points[3]]
        topRight = [self.points[2], self.points[3]]
        return [bottomLeft,  topRight, bottomRight, topLeft]
    

        
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
        self.distance = 1 # Placeholder for distance

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
        weight = (self.area / 0.001) + (self.proximity * 1) + (self.distance * 2)
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
                m_pnpChild = pnpSolver(m_object)
                
                if m_object.compareWeight(focusedObj.weight):
                    focusedObj = m_object
            #print("Focused Object: ", focusedObj.classID)
        
                    

if __name__ == '__main__':
    main()

