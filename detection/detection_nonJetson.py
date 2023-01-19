import cv2
from cap_from_youtube import cap_from_youtube

from ultralytics import YOLO

#If you are using a video

#If you are inputting a yt video
videoUrl = 'https://www.youtube.com/watch?v=s3r1axRoDyQ'
cap = cap_from_youtube(videoUrl)

# Initialize YOLOv8 model
model_path = "training/runs/detect/train6/weights/best.pt" #training/runs/detect/train/weights/best.pt
model = YOLO(model_path)

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

if cap.isOpened() == False:
    cap.Open()

while cap.isOpened():
#   Press key q to stop
    if cv2.waitKey(1) == ord('q'):
        break
# Read frame from the video
    ret, frame = cap.read()
    if not ret:
        break 
    results = model.predict(source=f'{videoUrl}', save=False, conf=.60)#f'{videoUrl}'
