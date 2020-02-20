#!/usr/bin/env python
# coding=utf8

import rospy
from dynamic_distance import Distance
from sensor_msgs.msg import Image
from std_msgs.msg import TwoFloat
from geometry_msgs.msg import Twist
import time
import cv2
from cv_bridge import CvBridge, CvBridgeError

# nodo che prende i valori pubblicati sui topic "image_topic" e "ultrasuoni_topic" e pubblica sul topic "change_obstacle" i valori di velocità
# angolare e lineare che permettono al robot di evitare gli eventuali ostacoli rilevati dalla camera e dai sonar

class ObstacleAvoidance(object):

    def __init__(self, linear_vel_base, angular_vel_base):
	# fattore utilizzato per modificare la rotazione del robot a seconda della distanza dall'ostacolo colorato
        self.factor = 1
        self.node_rate = 10
        self.area = 200000
        self.target_x = 0
        self.w = 640 # larghezza del frame
        self.h = 480 # altezza del frame
	# distanze dagli ostacoli inizializzate a 0
        self.distanza_ostacolo = 200
        self.distanza_dx = 0
        self.distanza_sx = 0
        self.bridge_object = CvBridge()
        rospy.Subscriber("image_topic", Image, self.callback, 0)
        rospy.Subscriber("ultrasuoni_topic", TwoFloat, self.callback, 1)
        self.pub = rospy.Publisher("change_obstacle", Twist, queue_size=1)
	# viene creato un oggetto della classe Distance        
	self.distance = Distance()
	# i valori di velocità lineare e angolare di base sono inizializzati con i valori passati come parametri
	# al momento della creazione di un oggetto della classe
        self.linear_vel_base = linear_vel_base
        self.angular_vel_base = angular_vel_base
	
    # callback invocata ogni volta che viene pubblicato qualcosa sui due topic per cui il nodo è subscriber
    def callback(self, data, args):
        if args == 0: # la callback è relativa a "image_topic"
            cv_image = self.bridge_object.imgmsg_to_cv2(data, "bgr8")
            cropped = cv_image[100:480, 0:640] # immagine ritagliata per evitare di prendere anche altri oggetti colorati fuori dalla pista
            # cv2.imshow("im", cropped)
            # cv2.waitKey(0)
	    # calcolo dell'area e del target dell'ostacolo eventualmente rilevato ricorrendo alla classe Distance
            self.area, self.target_x = self.distance.find_area(cropped)
	    # calcolo della distanza dal possibile ostacolo ricorrendo nuovamente alla classe Distance
            self.distanza_ostacolo = self.distance.distancetoCamera(self.area)

            if 70 < self.distanza_ostacolo <= 100: # oggetto è ancora lontano 
                self.factor = 0.9 # mantengo la stessa velocità angolare
            elif 50 < self.distanza_ostacolo <= 70: # oggetto è più vicino
                self.factor = 1 # aumento la velocità di rotazione
            else: # ostacolo imminente
                self.factor = 1.2 # aumento maggiormente la velocità angolare
        else: # la callback è relativa a "ultrasuoni_topic" e prende semplicemente i valori di distanza pubblicati
            self.distanza_sx = data.left_us
            self.distanza_dx = data.right_us
    # funzione per calcolare le velocità da pubblicare sul topic "change_obstacle" in base a quanto pubblicato sui topic
    # di cui il nodo è subscriber
    def calc_speed(self):
        speed = Twist() # creazione del messaggio come tipo Twist
        if 100 < self.distanza_ostacolo <= 200:  # ostacolo non visibile o troppo lontano
            if self.distanza_sx < 15 and self.distanza_dx < 15:  # se ho ostacoli a dx e sx
                speed.linear.x = self.linear_vel_base  # mantengo la stessa velocità lineare di base
                speed.angular.z = 0 # non ruoto
                print("{OBSTACLE_AVOIDANCE} Linear: " + str(speed.linear.x) + ", Angular: " + str(speed.angular.z))

            elif self.distanza_sx < 15:  # se ho ostacolo a sx vado a dx
                # imposto una velocità angolare che lo fa spostare un po' verso dx (segno negativo)
                speed.linear.x = self.linear_vel_base * -1
                speed.angular.z = self.angular_vel_base * -1
                print("{OBSTACLE_AVOIDANCE} Linear: " + str(speed.linear.x) + ", Angular: " + str(speed.angular.z))

            elif self.distanza_dx < 15:  # se ho ostacolo a dx vado a sx
                # imposto una velocità angolare che lo fa spostare un po' verso sx
                speed.linear.x = self.linear_vel_base * -1
                speed.angular.z = self.angular_vel_base
                print("{OBSTACLE_AVOIDANCE} Linear: " + str(speed.linear.x) + ", Angular: " + str(speed.angular.z))

            else:  # se non ho ostacoli a destra o sinistra continuo a fare quello che stavo facendo e passo dei valori di default
                speed.linear.x = -1000
                speed.angular.z = -1000
        else:  # ostacolo visibile alla camera
            if self.distanza_sx > 15 and self.distanza_dx > 15:  # non ho ostacoli vicini a dx e a sx -> considero solo l'ostacolo davanti
                if self.target_x < self.w / 2: # se il centroide dell'ostacolo è a sinistra rispetto al centro dell'immagine
                    speed.linear.x = self.linear_vel_base
                    speed.angular.z = self.angular_vel_base * self.factor*-1 # vado a destra considerano il fattore impostato nella callback
                elif self.target_x > self.w / 2: # se il centroide dell'ostacolo è a destra rispetto al centro dell'immagine
                    speed.linear.x = self.linear_vel_base
                    speed.angular.z = self.angular_vel_base * self.factor # vado a sinistra considerando il fattore nella callback
                else:
                    speed.linear.x = self.linear_vel_base # se il centroide dell'ostacolo è centrato per una scelta casuale vado a sx
                    speed.angular.z = self.angular_vel_base * self.factor
            elif self.distanza_dx < 15 and self.distanza_sx < 15:  # ho ostacoli in tutte le direzioni -> mi fermo
                speed.linear.x = 0
                speed.angular.z = 0
            elif self.distanza_sx < 15:  # ho ostacolo davanti e a sx
                if self.target_x <= self.w / 2: # se il centroide è a sx rispetto al centro dell'immagine
                    speed.linear.x = self.linear_vel_base
                    speed.angular.z = self.angular_vel_base * self.factor*-1 # vado a dx proporzionalmente al fattore
                else: # se il centroide è a dx vado dritto 
                    speed.linear.x = self.linear_vel_base
                    speed.angular.z = 0
            else:  # ho ostacolo davanti e a dx
                if self.target_x >= self.w/2: # il centroide è a dx
                    speed.linear.x = self.linear_vel_base
                    speed.angular.z = self.angular_vel_base* self.factor # vado a sx
                else: # centroide è a sx vado dritto
                    speed.linear.x = self.linear_vel_base
                    speed.angular.z = 0

            print("{OBSTACLE_AVOIDANCE + CAMERA} Linear: " + str(speed.linear.x) + ", Angular: " + str(speed.angular.z) + ", Distanza: " + str(self.distanza_ostacolo) + ", Target: " + str(self.target_x) + ", Area: " + str(self.area))
        self.pub.publish(speed)


if __name__ == '__main__':
    rospy.init_node('obstacle_avoidance', anonymous=True)
    obstacle_avoidance = ObstacleAvoidance(0.12, 0.28)
    loop = rospy.Rate(obstacle_avoidance.node_rate)
    while not rospy.is_shutdown():
        obstacle_avoidance.calc_speed()
        loop.sleep()
