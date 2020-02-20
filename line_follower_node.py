#!/usr/bin/env python
# coding=utf8
import rospy
import cv2
import numpy as np
from std_msgs.msg import TwoBool, Int16
from geometry_msgs.msg import Twist

# nodo che prendendo i valori restituiti dagli infrarossi attraverso il topic "ir_node_topic" cambia i valori di velocità angolare e lineare
# per permettere al robot di seguire la linea

class LineFollower(object):

    def __init__(self, linear_vel_base, angular_vel_base):
        self.ir_sub = rospy.Subscriber("ir_node_topic", TwoBool, self.line_foll_callback)
        self.line_foll_pub = rospy.Publisher("line_foll_topic", Twist, queue_size=1)
	# creazione del messaggio che sarà pubblicato su "line_foll_topic", i cui valori di velocità lineare e angolare sono
	# inizializzati entrambi a 0
        self.cmd_vel = Twist()
        self.cmd_vel.linear.x = 0.0
        self.cmd_vel.angular.z = 0.0
	# valori di base della velocità lineare e angolare passati come parametri al momento della creazione di un oggetto della classe
        self.linear_vel_base = linear_vel_base
        self.angular_vel_base = angular_vel_base
	
    # callback invocata quando viene pubblicato qualcosa sul topic "ir_node_topic" e che pubblica sul topic "line_foll_topic"
    # i valori relativi alla velocità angolare e lineare che permettono di seguire la linea.
    def line_foll_callback(self, ir_sensors):
        left_ir = ir_sensors.left_ir
        right_ir = ir_sensors.right_ir

        if left_ir and right_ir:
            # è arrivato alla linea di inizio/fine giro
            self.cmd_vel.angular.z = -1000
            self.cmd_vel.linear.x = -1000
        elif left_ir and not right_ir:
            # il sensore sx rileva la linea -> vado a sinistra
            self.cmd_vel.angular.z = self.angular_vel_base # rendo la velocità angolare di base positiva
            self.cmd_vel.linear.x = self.linear_vel_base # la velocità lineare è quella di base
            print("{LINE_FOLLOWER} SPEED==>[" + str(self.cmd_vel.linear.x) + "," + str(self.cmd_vel.angular.z) + "]")

        elif right_ir and not left_ir:
            # il sensore dx rileva la linea -> vado a destra
            self.cmd_vel.angular.z = self.angular_vel_base *-1 # velocità angolare negativa per andare a dx
            self.cmd_vel.linear.x = self.linear_vel_base
            print("{LINE_FOLLOWER} SPEED==>[" + str(self.cmd_vel.linear.x) + "," + str(self.cmd_vel.angular.z) + "]")

        else:
            # sono nella carreggiata -> pubblico i valori di default
            self.cmd_vel.angular.z = -1000
            self.cmd_vel.linear.x = -1000

        self.line_foll_pub.publish(self.cmd_vel)


if __name__ == "__main__":
    rospy.init_node("line_follower", anonymous=True)
    line_follower = LineFollower(0.10, 0.28)
    rospy.spin()
