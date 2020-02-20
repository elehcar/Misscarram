#!/usr/bin/env python
# coding=utf8

import rospy
import RPi.GPIO as GPIO
import time
from time import sleep
from std_msgs.msg import TwoFloat

#nodo ultrasuoni che pubblica sul topic "ultrasonic_topic" i valori di distanza dagli ostacoli rilevati dall'ultrasuono dx e dall'ultrasuono sx.
class UltraSuoni(object):

    def __init__(self):
        self.node_rate = 10
        self.pub = rospy.Publisher("ultrasuoni_topic", TwoFloat, queue_size=1)
	#specifichiamo che usiamo il sistema di numerazione per le GPIO definito dal chip Broadcom
	# BCM = Broadcom SOC channel
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
	#pin “Trigger” che deve essere attivato per inviare il segnale ad ultrasuoni.
        self.GPIO_TRIGGER = 4
	#pin "Echo" che producono un impulso che si interrompe quando viene ricevuto il segnale riflesso dall’oggetto.
        self.ECHO_RIGHT = 16
        self.ECHO_LEFT = 26
	# settiamo il pin di trigger come output e quelli di echo come input
        GPIO.setup(self.GPIO_TRIGGER, GPIO.OUT)
        GPIO.setup(self.ECHO_RIGHT, GPIO.IN)
        GPIO.setup(self.ECHO_LEFT, GPIO.IN)
    # metodo che restituisce la distanza rilevata da uno specifico ultrasuono
    def distance(self, echo):
        # massimo tempo di attesa per la risposta in caso qualcosa si perda
        MAX_TIME = 0.04 
        # Definisco lo stato del pin relativo al trigger come alto per iniziare la misura
        GPIO.output(self.GPIO_TRIGGER, True)
        time.sleep(0.00001)
        GPIO.output(self.GPIO_TRIGGER, False)
        # ci assicuriamo che il tempo di inizio sia settato in caso di ritorno molto rapido
        start = time.time()
	# e impostiamo un timeout pari al tempo di inizio più il MAX_TIME
        timeout = start + MAX_TIME

        # imposto la linea ad input per verificare l'inizio della risposta echo
        while GPIO.input(echo) == 0 and start <= timeout:
	    # start viene aggiornato
            start = time.time()
	# a stop viene assegnato il valore dell'istante in cui GPIO.input(echo) da 0 passa a 1
        stop = time.time()
	# il valore della variabile timeout è aggiornato al valore di stop rilevato precedentemente 
	# a cui è sommato la variabile MAX_TIME.
        timeout = stop + MAX_TIME
        # Si attende per la fine della risposta echo
        while GPIO.input(echo) == 1 and stop <= timeout:
	    # stop viene aggiornato 
            stop = time.time()
	# il setup successivo mi sembra inutile
        GPIO.setup(self.GPIO_TRIGGER, GPIO.OUT)
	# Definisco lo stato del pin relativo al trigger come basso per fermare la misura
        GPIO.output(self.GPIO_TRIGGER, False)
	
	# si calcola il tempo trascorso come differenza tra i valori delle variabili stop e start
        elapsed = stop - start
	# la distanza dall'ostacolo è ottenuta moltiplicando il tempo trascorso per la velocità del suono nell'aria
	# diviso per 2 considerando che nel tempo trascorso sono compresi sia l'andata che il ritorno.
        dista = (elapsed * 34300) / 2.0
        time.sleep(0.02)
        return dista
	
    # metodo per calcolare le distanza dagli ostacoli di entrambi gli ultrasuoni utilizzando la funzione definita
    # precedentemente e pubblicarle sul topic . 
    def run_distance(self):
        right_dist1 = self.distance(self.ECHO_RIGHT)
        left_dist1 = self.distance(self.ECHO_LEFT)

        right_dist2 = self.distance(self.ECHO_RIGHT)
        left_dist2 = self.distance(self.ECHO_LEFT)
	# si considerano le differenze fra i valori di due misure successive per evitare la 
	# presenza di valori spuri restituiti casualmente dal sensore a causa della sua imprecisione.
        difference_right = abs(right_dist1 - right_dist2)
        difference_left = abs(left_dist1 - left_dist2)

        ultra = TwoFloat()
	# se le 2 misure consecutive hanno ambedue le differenze minori di una certa soglia (20) allora viene pubblicata una delle due
	# altrimenti si ignorano entrambe.
        if difference_left < 20 and difference_right < 20:
            ultra.left_us = left_dist1
            ultra.right_us = right_dist1
            self.pub.publish(ultra)
            rospy.loginfo('{ULTRASUONI} Distanza sx: ' + str(left_dist1) + ", Distanza dx: " + str(right_dist1))
        else:
            pass


if __name__ == "__main__":
    ultra_suoni = UltraSuoni()
    rospy.init_node("ultrasuoni", anonymous=True)
    loop = rospy.Rate(ultra_suoni.node_rate)
    while not rospy.is_shutdown():
        ultra_suoni.run_distance()
        loop.sleep()
    GPIO.cleanup()
