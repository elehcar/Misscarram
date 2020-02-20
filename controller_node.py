#!/usr/bin/env python
# coding=utf8
import rospy
import cv2
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import IntString

# nodo controllore che fa da subscriber per tre topic su cui pubblicano il nodo relativo agli ostacoli, quello che permette di seguire la linea e il
# nodo di pianificazione. E' publisher per il topic "cmd_vel_topic" a cui passa i valori di velocità lineare e angolare secondo
# dei criteri di priorità.
class ControllerNode(object):

    def __init__(self):
        self.orientation = None # serve per gestire le situazioni in cui il robot gira a dx (o a sx) per un ostacolo e quando
                                # vede nuovamente la linea sempre a dx (o a sx) continua a girare troppo andano a sbattere contro
                                # la parete. Quindi in questa situazione lo facciamo girare in direzione opposta
        self.curve = False
        self.qr_distance = 200
        self.ob_linear = -1000
        self.line_linear = -1000
        self.ob_angular = -1000
        self.line_angular = -1000
        self.linear = -1000
        self.angular = -1000
        rospy.Subscriber("line_foll_topic", Twist, self.decision_callback, 0)
        rospy.Subscriber("change_obstacle", Twist, self.decision_callback, 1)
        rospy.Subscriber("planning_topic", IntString, self.decision_callback, 2)
        self.pub = rospy.Publisher("cmd_vel_topic", Twist, queue_size=1)

        # callback invocata ogni volta che qualcosa è pubblicato su uno dei nodi per cui è subscriber
    def decision_callback(self, data, args):
        rotation = 0 # inizializzata a 0 ogni volta che si chiama una callback, serve per identificare configurazioni non permesse al robot
        speed = Twist() # creazione del messaggio di tipo Twist da pubblicare
        if args == 0: # callback relativa al "line_foll_topic" aggiorna i valori di velocità lineare e angolare della linea
            self.line_linear = data.linear.x 
            self.line_angular = data.angular.z
        elif args == 2: # callback relativa al "planning_topic", aggiorna il valore di rotation, distanza dal landmark e dice se il landmark visto è prossimo a una curva
            rotation = data.rotation
            self.qr_distance = data.distance
            self.curve = data.curve
        else: # callback relativa a "change_obstacle",  imposta i valori delle velocità angolare e lineare degli ostacoli 
            self.ob_linear = data.linear.x
            self.ob_angular = data.angular.z

        # se non ci sono ostacoli da considerare
        if (self.ob_linear == -1000) and (self.ob_angular == -1000):
            # se il robot era già andato a sinistra e velocità angolare della linea suggerisce di andare di nuovo a destra
            # oppure se il robot erà già andato a destra e la velocità angolare della linea suggerisce di andare di nuovo a 
            # sinistra -> lo facciamo girare in verso opposto per evitare che girando troppo in una direzione vada a sbattere
            # contro la parete
            if (self.orientation == 0 and self.line_angular < 0) or (self.orientation == 1 and self.line_angular > 0):
                self.angular = self.line_angular * -1
            else: # se non siamo nella situazione critica precedente passa la velocità angolare così come pubblicata
                  # dal nodo line_follower
                self.angular = self.line_angular
            self.orientation = None
            self.linear = self.line_linear

        # se invece ci sono gli ostacoli da considerare
        else:
            if self.ob_angular < 0:  # velocità angolare negativa, il robot gira a sx
                self.orientation = 0
            else:  # velocità angolare positiva il robot gira a dx
                self.orientation = 1
            self.linear = self.ob_linear
            self.angular = self.ob_angular

        if rotation == 1: # il robot deve tornare indietro perchè sta assumendo una configurazione non ammessa
            if (self.curve and self.qr_distance > 30) or not self.curve:
                # se l'ultimo landmark visto era uno di curva e la distanza da esso era > 30 oppure se non era di curva , significa
                # che il robot sta andando contro le pareti
                print("Turn opposite")
                self.linear = self.linear * -1 # il robot fa un passo indietro
                self.qr_distance = 0 
            else: # il robot sta facendo la curva 
                if self.angular < -0.2:
                    self.angular = self.angular * 0.7
                print("Doing curve!")
        elif rotation == 2: #il robot è in una configurazione non ammessa , quindi ruota di 180°
            self.linear = 0
            self.angular = self.angular * 1.5
        elif rotation == 3:
            self.linear = self.linear * -1
        else:
            print("Doing great!")
        
        speed.linear.x = self.linear
        speed.angular.z = self.angular

        self.pub.publish(speed)
        print("{CONTROLLER_NODE}  Linear: " + str(speed.linear.x) + ", Angular: " + str(speed.angular.z))


if __name__ == "__main__":
    rospy.init_node("decision_node", anonymous=True)
    ob = ControllerNode()
    rospy.spin()
