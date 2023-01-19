from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput

net = detectNet("detection/finalweights/best.engine", threshold=0.6)
camera = videoSource("csi://0")      # '/dev/video0' for V4L2
display = videoOutput("display://0") # 'my_video.mp4' for file

while display.IsStreaming():
	img = camera.Capture()
	detections = net.Detect(img)
	display.Render(img)
	display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))