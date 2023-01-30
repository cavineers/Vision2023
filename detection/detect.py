import cv2

from ultralytics import YOLO

# Initialize YOLOv8 model
model_path = "training/runs/detect/train7/weights/best.pt" #training/runs/detect/train/weights/best.pt
model = YOLO(model_path)

results = model.predict(source=1, return_outputs=True) #Starts inferencing


def getDelta(x1, y1, x2, y2):
    #Returns the delta (change) in x and y as an array
    deltX = (x2 - x1)
    deltY = (y2 - y1)
    return [deltX, deltY]

def getArea(deltaX, deltaY):
    area = (deltaX * deltaY)
    return area

def main():
    for resultDict in results:
        if resultDict:
            #resultDict Format --> {'det': array([[       1666(X1),         398(Y1),        1914(X2),         593(Y2),     0.78452(Conf),           1]], dtype=float32)}ß
            for object in resultDict['det']: #object contains a list containing object data 
                #Top Left Corner on a Non-mirrored input
                x1 = object[0] #x val
                y1 = object[1] #y val

                #Bottom Right Corner on a Non-mirrored input
                x2 = object[2]
                y2 = object[3]
                #print(f'X1: {x1} Y1: {y1} \n X2: {x2} Y2: {y2}')

        else: # No results -- Dict doesn't exist
            print('No results')

if __name__ == '__main__':
    main()

