import ntcore

class Network:
    def __init__(self):
        print("Initializing Network")
        self.table = ntcore.NetworkTableInstance.getDefault().getTable("ObjectDetectionNetwork")
        self.angleToObj = self.table.getDoubleTopic("angleToObj").Publish()
        self.classID = ntcore.DoubleEntry("classID").Publish()


        self.angleToObj.setDouble(0.0)
        self.classID.setDouble(0.0)
    def publish(self, values):
        angle = values[0]
        classID = values[1]
        self.angleToObj.setDouble(angle)
        self.classID.setDouble(classID)

