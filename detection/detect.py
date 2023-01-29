import cv2

from ultralytics import YOLO

# Initialize YOLOv8 model
model_path = "training/runs/detect/train7/weights/best.pt" #training/runs/detect/train/weights/best.pt
model = YOLO(model_path)

results = model.predict(source=1, return_outputs=True)

for resultDict in results:
    if resultDict:
        #print(resultDict) Format {'det': array([[       1666(X1),         398(Y1),        1914(X2),         593(Y2),     0.78452(Conf),           1]], dtype=float32)}ß
        for object in resultDict['det']: #object contains a list containing object data 

            #Bottom Left Corner on a Non-mirrored input
            x1 = object[0] #x val
            y1 = object[1] #y val

            #Top Right Corner on a Non-mirrored input
            x2 = object[2]
            y2 = object[3]

    else:
        print('No results')

print("^Done")

