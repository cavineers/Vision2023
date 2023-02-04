## yolo v8 provides an extremely concise and simple method of training

# pip install ultralytics --install dependancies and libraries

import ultralytics
ultralytics.checks() #Checks for successful install/import

from ultralytics import YOLO

# Load a model

def train():
    model = YOLO("training/runs/detect/train/weights/best.pt")  # load the previous model here OR use a yolov8 model if you want to restart the transfer training ("yolov8s.pt")

    # Use the model
    training = model.train(data="training/datasets/FRC Object Detection-4/data.yaml", epochs=100)  #epochs can just be 25-30...
    training = model.val()  # evaluate model performance on the validation set

def export():
    model = YOLO("training/runs/detect/train7/weights/best.pt")

    model.val()
    model.export("training/runs/detect/train7/weights/best.onnx")

if __name__ == "__main__":
    export()
