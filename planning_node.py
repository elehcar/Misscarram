#!/usr/bin/env python
# coding=utf8
import rospy
import cv2
from landmark import Landmark
from std_msgs.msg import IntString, Float32
from sensor_msgs.msg import Image
import csv
from cv_bridge import CvBridge, CvBridgeError

# nodo che è publisher per il topic "planning_topic" e subscriber per "qr_topic" e "magnetometer_topic" 
class PlannerNode:

    def __init__(self):
        self.bridge_object = CvBridge()
        self.curve = False
        self.landmark = Landmark() # oggetto della classe Landmark
        self.north = 0
        self.qr_code = "11"
        self.qr_distance = 200
        self.lowlimit = 0
        self.upperlimit = 359

        self.pub = rospy.Publisher("planning_topic", IntString, queue_size=10)
        rospy.Subscriber("qr_topic", Image, self.planner_callback, 0)
        rospy.Subscriber("magnetometer_topic", Float32, self.planner_callback, 1)

    # callback invocata ogni qual volta viene pubblicato qualcosa su uno dei due topic per cui il nodo è 
    # subscriber
    def planner_callback(self, data, arg):
        raw_grades = 5  # range di tolleranza per i gradi
        msg = IntString() # creazione del messaggio da pubblicare su "planning_topic"
        if arg == 1: # callback relativa al magnetometer_topic
            self.north = data.data
        else: # callback relativa al qr_topic
            cv_image = self.bridge_object.imgmsg_to_cv2(data, "bgr8")
            cropped = cv_image[0:250, 0:640]  # immagine ritagliata per centrare il landmark
            self.qr_code, self.qr_distance = self.landmark.get_qrdata(cropped) # si invoca il metodo della classe Landmark get_qrdata
									       # per avere il contenuto e la distanza dal qr_code

        msg.distance = self.qr_distance

        if self.qr_code == "2" or self.qr_code == "3": # se il landmark è il n.2 o il n.3
	    # si impostano i limiti superiore e inferiore che la bussola può segnare per avere 
	    # l'orientazione giusta nel primo lato della pista 
            self.lowlimit = 280
            self.upperlimit = 350
            if self.qr_code == "3": # se il landmark visto è quello di curva 
                self.curve = True # imposto il flag di curva a true
            else:
                self.curve = False # non è un landmark di curva e si imposta il flag a false
        elif self.qr_code == "4" or self.qr_code == "5": # se il landmark è il n.4 o il n.5
	    # si impostano i limiti superiore e inferiore che la bussola può segnare per avere 
	    # l'orientazione giusta nel secondo lato della pista
            self.lowlimit = 115
            self.upperlimit = 180
            if self.qr_code == "5": # se il landmark visto è quello di curva 
                self.curve = True # imposto il flag di curva a true
            else:
                self.curve = False # non è un landmark di curva e si imposta il flag a false
        elif self.qr_code == "1" or self.qr_code == "6":  # se il landmark è il n.1 o il n.6
	    # si impostano i limiti superiore e inferiore che la bussola può segnare per avere 
	    # l'orientazione giusta nel terzo lato della pista
            self.lowlimit = 30
            self.upperlimit = 75
            if self.qr_code == "1": # se il landmark visto è quello di curva
                self.curve = True # imposto il flag di curva a true
            else:
                self.curve = False  # non è un landmark di curva e si imposta il flag a false
        else:  # landmark non visibile
            self.lowlimit = 0
            self.upperlimit = 359
            self.curve = False

        msg.curve = self.curve

        if self.lowlimit <= self.north <= self.upperlimit: # la bussola segna un'orientazione nei limiti impostati a seconda del tratto 
        # di pista che si sta percorrendo
            msg.rotation = 0  # Normal
        elif (self.lowlimit-raw_grades <= self.north < self.lowlimit) or (self.upperlimit < self.north <= self.upperlimit + raw_grades):
	    # Turn Opposite ( nel caso in cui la bussola segni un'orientazione sotto il limite inferiore di una quantità non maggiore 
	    # a raw_grades oppure sopra il limite superiore di una quantità non maggiore a raw_grades)
            msg.rotation = 1
        else:  # il robot risulta con un'orientazione completamente sbagliata -> inversione
            msg.rotation = 2  # Invert
        print("{PLANNING_NODE} Rotation: " + str(msg.rotation) + ", QR_CODE: " + str(self.qr_code), "Distance: " + str(msg.distance) + ",  Curva: " + str(msg.curve))
        self.pub.publish(msg)


if __name__ == "__main__":
    rospy.init_node("planning_node", anonymous=True)
    ob = PlannerNode()
    rospy.spin()

