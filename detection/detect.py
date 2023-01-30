import cv2
import numpy as np

from ultralytics import YOLO

# Initialize YOLOv8 model
model_path = "training/runs/detect/train7/weights/best.pt" #training/runs/detect/train/weights/best.pt
model = YOLO(model_path)




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

def compareWeight(weight, f_weight):
    if weight > f_weight:
        return True
    else:
        return False

def determineFocus(object, f_weight): #Determines what object is in focus
    '''
        This function compares two objects based off a weight system
        
        Weights:
        Area: x1
        Position on Screen: x1
        Distance: x2
    '''
    #Basic Current Object Info
    points = getPoints(object)
    deltas = getDeltas(points)
    area = getArea(deltas)

    #Determine Proximity to center of screen



    #Pose Estimation
    return True, 0
    #Determine Weights







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

