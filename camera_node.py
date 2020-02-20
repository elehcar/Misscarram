#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import time
from picamera import PiCamera
from picamera.array import PiRGBArray

# nodo che pubblica sui due topic "image_topic" e "qr_topic" le immagini acquisite dalla PiCamera
class ImagePublisher(object):
    # metodo init invocato al momento della creazione di un oggetto della classe
    def __init__(self):
        self.node_rate = 10
        self.image_pub = rospy.Publisher("image_topic", Image, queue_size=1)
        self.landmark_pub = rospy.Publisher("qr_topic", Image, queue_size=1)
        self.bridge = CvBridge()
        self.camera = PiCamera()
        self.camera.resolution = (640, 480)
        self.camera.framerate = 32
        self.rawCapture = PiRGBArray(self.camera, size=(640, 480))
    # metodo get_img che permette di ottenere un'immagine dalla piCamera e pubblicarla
    def get_img(self):
        self.camera.capture(self.rawCapture, format='bgr')
        cv_image = self.rawCapture.array
        image_message = None
        try:
            image_message = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
        except CVBridgeError as e:
            print(e)
	#immagine pubblicata sui due topic
        self.image_pub.publish(image_message)
        self.landmark_pub.publish(image_message)
        print('{CAMERA_NODE} Publishing! ')
        self.rawCapture.truncate(0)


if __name__ == '__main__':
    image_getter = ImagePublisher()
    rospy.init_node('image_getter', anonymous=True)
    loop = rospy.Rate(image_getter.node_rate)
    while not rospy.is_shutdown():
	#chiamata del metodo get_img su un oggetto della classe ImagePublisher
        image_getter.get_img()
        loop.sleep()
