import ntcore

class Network:
    def __init__(self):
        print("Initializing Network")
        self.table = ntcore.NetworkTableInstance.getDefault().getTable("ObjectDetectionNetwork")
        self.angleToObj = self.table.getDoubleTopic("angleToObj").publish()
        self.classID = self.table.getDoubleTopic("classID").publish()

       



        self.angleToObj.set(0.0)
        self.classID.set(0.0)
    def publish(self, values):
        angle = values[0]
        classID = values[1]
        self.angleToObj.set(angle)
        self.classID.set(classID)

