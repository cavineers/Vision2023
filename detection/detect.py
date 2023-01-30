import cv2

from ultralytics import YOLO

# Initialize YOLOv8 model
model_path = "training/runs/detect/train7/weights/best.pt" #training/runs/detect/train/weights/best.pt
model = YOLO(model_path)

results = model.predict(source=1, return_outputs=True) #Starts inferencing


def getPoints(object):
    #Top Left Corner on a Non-mirrored input
    x1 = object[0] #x val
    y1 = object[1] #y val

    #Bottom Right Corner on a Non-mirrored input
    x2 = object[2]
    y2 = object[3]
    return [x1, y1, x2, y2]
def getDelta(points):
    #Returns the delta (change) in x and y as an array
    deltX = (points[2] - points[0])
    deltY = (points[3] - points[1])
    return [deltX, deltY]

def getArea(deltas):
    area = (deltas[0] * deltas[1])
    return area

def compareArea(largest, currentArea):
    if currentArea > currentArea:
        return currentArea
    else:
        return largest

def determineFocus(object):
    points = getPoints(object=object)
    deltas = getDelta(points)
    




def main():
    while True:
        focusedObj = None # Reset the focused obj every full iteration
        for resultDict in results:
            if resultDict:
                focused = determineFocus(object)
                if focused: # If the object is determined to be focused
                    focusedObj = object 
            else: # No results -- Dict doesn't exist
                print('No results')

if __name__ == '__main__':
    main()

