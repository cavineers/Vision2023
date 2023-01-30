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
        
        self.classID = object[5]

        self.points = self.getPoints(object)
        self.deltas = self.getDeltas(self.points)
        self.area = self.getArea(self.deltas)
        self.center = self.getCenter(self.deltas)
        self.proximity = self.getProximityCenter(self.center)
        self.distance = 1 # Placeholder for distance

        self.weight = self.calculateWeight(self.area, self.proximity, self.distance)

    def getPoints(self, object):
        #Top Left Corner on a Non-mirrored input
        x1 = object[0] #x val
        y1 = object[1] #y val

        #Bottom Right Corner on a Non-mirrored input
        x2 = object[2]
        y2 = object[3]
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

    def calculateWeight(self, area, proximity, distance): #Calculates weights
        weight = (area * 0.5) + (proximity * 1) + (distance * 2)
        return weight
    
    def compareWeight(self, f_weight):
        print("comparing weights")
        if self.weight > f_weight:
            return True
        else:
            return False

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
'''
Solving for 3D Measurements of an object
'''







def main():
    
    results = model(source=1, imgsz=640, return_outputs=True, conf=.60) #Starts inferencing
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
                    continue
                if m_object.compareWeight(focusedObj.weight):
                    focusedObj = m_object
            print("Focused Object: ", focusedObj.classID)
        
                    

if __name__ == '__main__':
    main()

