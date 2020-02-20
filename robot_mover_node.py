#!/usr/bin/env python
# coding=utf8

import rospy
import sys
from geometry_msgs.msg import Twist
from motor_driver import MotorDriver

# nodo subscriber per il topic "cmd_vel_topic" , che tramite la classe MotorDriver cambia la velocità lineare e angolare delle ruote.
class RobotMover(object):

    def __init__(self):
	# valori di velocità angolare e lineare precedenti , inizializzati a 0 e 0.1
        self.previous_angular = 0
        self.previous_linear = 0.1
        rospy.Subscriber("cmd_vel_topic", Twist, self.cmd_vel_callback)
	# creazione di un oggetto della classe MotorDriver
        self.motion_driver = MotorDriver()
	
    # callback invocata ogni qual volta venga pubblicato qualcosa sul topic "cmd_vel_topic" 
    def cmd_vel_callback(self, data):
	# velocità lineare e angolare pubblicate sul topic
        linear_speed = data.linear.x
        angular_speed = data.angular.z
	# se sono dei valori di default, si passano a change speed le velocità precedenti, continua a fare quello che faceva
        if (linear_speed == -1000) or (angular_speed == -1000) or (linear_speed == 1000) or (angular_speed == 1000):
            if self.previous_linear < 0: # se è negativa, la impostiamo positiva in modo da fargli fare solo un passo indietro
                self.previous_linear = self.previous_linear * -1
            self.motion_driver.change_speed(self.previous_linear, self.previous_angular)
            print("{ROBOT_MOVER} Linear: " + str(self.previous_linear) + ", Angular: " + str(self.previous_angular))
        else: # se le velocità sono cambiate, aggiorno i valori di quelle precedenti e li passo a change_speed
            self.previous_linear = linear_speed
            self.previous_angular = angular_speed
            self.motion_driver.change_speed(linear_speed, angular_speed)
            print("{ROBOT_MOVER} Linear: " + str(linear_speed) + ", Angular: " + str(angular_speed))


if __name__ == '__main__':
    rospy.init_node('vel_listener', anonymous=True)
    robot_mover = RobotMover()
    rospy.spin()
