#!/usr/bin/env python
# coding=utf8
import rospy
import cv2
import RPi.GPIO as IO
import time
from std_msgs.msg import TwoBool
IO.setwarnings(False)

# nodo che pubblica sul topic "ir_node_topic" i due valori booleani relativi all'infrarosso destro e all'infrarosso sinistro
class IrNode(object):

    def __init__(self):
        self.node_rate = 10
        IO.setmode(IO.BCM)
        self.line_foll_pub = rospy.Publisher("ir_node_topic", TwoBool, queue_size=1)
        # settiamo le GPIO 10 e 9 in pin di input
        IO.setup(10, IO.IN)  # LEFT IR
        IO.setup(9, IO.IN)  # RIGHT IR

    def ir_function(self):
	# creazione del messaggio di tipo TwoBool e assegnazione alle sue componenti left e right
	# di ciò che leggono rispettivamente la GPIO 10 e la GPIO 9. Se gli infrarossi rilevano la linea nera restituiscono 
	# true altrimenti false. 
        ir_sensors = TwoBool()
        ir_sensors.left_ir = IO.input(10)
        ir_sensors.right_ir = IO.input(9)
        print("{INFRAROSSI}: sensing!")
	# pubblicazione del messaggio sul topic
        self.line_foll_pub.publish(ir_sensors)


if __name__ == "__main__":
    rospy.init_node("ir_node", anonymous=True)
    ir_node = IrNode()
    loop = rospy.Rate(ir_node.node_rate)
    while not rospy.is_shutdown():
        ir_node.ir_function()
        loop.sleep()
