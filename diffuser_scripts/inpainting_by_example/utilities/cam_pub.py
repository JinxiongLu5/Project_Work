#!/usr/bin/env python3

from picamera2 import Picamera2

import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class CameraPub():
    def __init__(self):
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
        self.picam2.start()
        rospy.init_node('Campubnode', anonymous=False)     
        self.publisher = rospy.Publisher("/automobile/image_raw", Image, queue_size=1)
    def run(self):

        while True:
                image = self.picam2.capture_array()
                imageObject = CvBridge().cv2_to_imgmsg(image)
                imageObject.header.stamp = rospy.Time.now()
                self.publisher.publish(imageObject)

if __name__ == '__main__':
    try:
        nod = CameraPub()
        nod.run()
    except rospy.ROSInterruptException:
        pass
