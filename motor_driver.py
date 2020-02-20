#!/usr/bin/env python
# coding=utf8

import rospy
import RPi.GPIO as GPIO
import time

# classe che implementa la guida differenziale

class MotorDriver(object):

    def __init__(self):
        #distanza tra le ruote
        self.wheel_distance = 0.158
        # raggio delle ruote
        self.wheel_radius = 0.0172
        self.PWM1 = 0
        self.PWM2 = 0
        # pin ruota sinistra
        self.in1 = 27
        self.in2 = 17
        # pin ruota destra
        self.in3 = 6
        self.in4 = 5
        # pin di enable
        self.en1 = 13
        self.en2 = 12

        self.BASE_PWM = 30
        self.MAX_PWM = 100

        self.MULTIPLIER_STANDARD = 0.3
        self.MULTIPLIER_PIVOT = 0.3

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        # setto i 3 pin delle ruote come pin di output
        GPIO.setup(self.in1, GPIO.OUT)
        GPIO.setup(self.in2, GPIO.OUT)
        GPIO.setup(self.en1, GPIO.OUT)
        GPIO.setup(self.in3, GPIO.OUT)
        GPIO.setup(self.in4, GPIO.OUT)
        GPIO.setup(self.en2, GPIO.OUT)
        # impostiamo i due pin di PWM con una frequenza di 1KHz
        self.p1 = GPIO.PWM(self.en1, 1000)
        self.p2 = GPIO.PWM(self.en2, 1000)
        # imposta un duty cicle dello 0%
        self.p1.start(self.PWM1)
        self.p2.start(self.PWM2)


    def __del__(self):
        GPIO.cleanup()

    def set_motor(self, v1, v2, v3, v4):
        GPIO.output(self.in1, v1)
        GPIO.output(self.in2, v2)
        GPIO.output(self.in3, v3)
        GPIO.output(self.in4, v4)
        
    # imposto alti o bassi i valori dei pin a cui sono collegate le ruote per ottenere comportamenti diversi
    def forward(self):
        self.set_motor(GPIO.HIGH, GPIO.LOW, GPIO.HIGH, GPIO.LOW)

    def stop(self):
        self.set_motor(GPIO.LOW, GPIO.LOW, GPIO.LOW, GPIO.LOW)

    def reverse(self):
        self.set_motor(GPIO.LOW, GPIO.HIGH, GPIO.LOW, GPIO.HIGH)

    def pivot_right(self):
        self.set_motor(GPIO.HIGH, GPIO.LOW, GPIO.LOW, GPIO.HIGH)

    def right(self):
        self.set_motor(GPIO.HIGH, GPIO.LOW, GPIO.LOW, GPIO.LOW)

    def pivot_left(self):
        self.set_motor(GPIO.LOW, GPIO.HIGH, GPIO.HIGH, GPIO.LOW)

    def left(self):
        self.set_motor(GPIO.LOW, GPIO.LOW, GPIO.HIGH, GPIO.LOW)

    def left_reverse(self):
        self.set_motor(GPIO.LOW, GPIO.HIGH, GPIO.LOW, GPIO.LOW)

    def right_reverse(self):
        self.set_motor(GPIO.LOW, GPIO.LOW, GPIO.LOW, GPIO.HIGH)
        
    # metodo per settare i due PWM delle ruote sulla base della velocità della ruota, di un moltiplicatore e del PWM di base, limitati 
    # considerando il minimo tra questa quantità e MAX_PWM possibile
    def set_speed(self, rpm_speed_1, rpm_speed_2, multiplier):
        self.PWM1 = min(int(rpm_speed_1 * multiplier * self.BASE_PWM), self.MAX_PWM)
        self.PWM2 = min(int(rpm_speed_2 * multiplier * self.BASE_PWM), self.MAX_PWM)
        self.p1.ChangeDutyCycle(self.PWM1)
        self.p2.ChangeDutyCycle(self.PWM2)
        
    # sulla base dei valori di velocità lineare e angolare passati come argomento si calcola il body turn radius, infatti in base
    # al rapporto fra questi due valori il cerchio che il robot deve compiere spostandosi sarà più piccolo o più grande. Se il valore
    # del body turn radius è più piccolo significa che il robot dovrà ruotare di più rispetto ad andare dritto, se il valore è più grande
    # vale il viceversa.
    def calculate_body_turn_radius(self, linear_speed, angular_speed):
        if angular_speed != 0.0: 
            body_turn_radius = linear_speed / angular_speed
        else: # se avessimo una velocità angolare negativa, avremmo un body turn radius infinito -> lo impostiamo a None
            body_turn_radius = None
        return body_turn_radius
    
    # metodo per calcolare il raggio di rotazione delle ruote sulla base del body_turn_radius, a seconda se 
    # il robot deve girare a destra o sinistra una delle due ruote dovrà girare più dell'altra. Se deve girare a dx
    # la ruota dx deve girare meno rispetto alla ruota sinistra.
    def calculate_wheel_turn_radius(self, body_turn_radius, wheel):
        if body_turn_radius is not None:
            if wheel == 'right':
                wheel_sign = 1
            else:
                wheel_sign = -1
            # sommiamo(ruota dx) o sottraiamo(ruota sx) al body turn radius la metà della distanza fra le ruote per ottenere
            # il raggio di rotazione della singola ruota
            wheel_turn_radius = body_turn_radius + (wheel_sign * (self.wheel_distance / 2.0))
        else: # nel caso di velocità angolare nulla, ovvero se il robot va dritto
            wheel_turn_radius = None
        return wheel_turn_radius
    
    # metodo per calcolare la velocità con cui le ruote girano
    def calculate_wheel_rpm(self, linear_speed, angular_speed, wheel_turn_radius):
        if wheel_turn_radius is not None: # robot sta girando
            wheel_rpm = (angular_speed * wheel_turn_radius) / self.wheel_radius # calcoliamo la velocità lineare di ogni ruota
            # come velocità angolare * raggio di rotazione della singola ruota
        else: # robot sta andando dritto o indietro, quindi la velocità lineare è la stessa per ambedue le ruote
            wheel_rpm = linear_speed / self.wheel_radius

        return wheel_rpm

    def set_wheel_movement(self, right_wheel_rpm, left_wheel_rpm):
        # sulla base del fatto che i due valori passati come argomento siano positivi, negativi o
        # nulli si settano prima le velocità delle due ruote con il metodo set_speed e poi si setta la direzione
        # con il metodo opportuno
        if right_wheel_rpm > 0.0 and left_wheel_rpm > 0.0: # entrambe le velocità delle ruote sono positive, si vuole andare dritto
            self.set_speed(abs(right_wheel_rpm), abs(left_wheel_rpm), self.MULTIPLIER_STANDARD)
            self.forward()

        elif right_wheel_rpm > 0.0 and left_wheel_rpm == 0.0: # si vuole andare a sinistra
            self.set_speed(abs(right_wheel_rpm), abs(left_wheel_rpm), self.MULTIPLIER_STANDARD)
            self.left()

        elif right_wheel_rpm > 0.0 and left_wheel_rpm < 0.0:
            self.set_speed(abs(right_wheel_rpm), abs(left_wheel_rpm), self.MULTIPLIER_PIVOT)
            self.pivot_left()


        elif right_wheel_rpm == 0.0 and left_wheel_rpm > 0.0:
            self.set_speed(abs(right_wheel_rpm), abs(left_wheel_rpm), self.MULTIPLIER_STANDARD)

            self.right()

        elif right_wheel_rpm < 0.0 and left_wheel_rpm > 0.0:
            self.set_speed(abs(right_wheel_rpm), abs(left_wheel_rpm), self.MULTIPLIER_PIVOT)

            self.pivot_right()

        elif right_wheel_rpm < 0.0 and left_wheel_rpm < 0.0: # si vuole andare indietro
            self.set_speed(abs(right_wheel_rpm), abs(left_wheel_rpm), self.MULTIPLIER_STANDARD)

            self.reverse()

        elif right_wheel_rpm == 0.0 and left_wheel_rpm == 0.0:
            self.set_speed(abs(right_wheel_rpm), abs(left_wheel_rpm), self.MULTIPLIER_STANDARD)
            self.stop()

        else:
            pass

    def change_speed(self, ls, a_s):
        # stabiliamo se il cerchio che dobbiamo percorrere è piccolo o grande 
        # a seconda dei valori di velocità lineare e velocità angolare
        body_turn_radius = self.calculate_body_turn_radius(ls, a_s)

        # calcoliamo sulla base di quanto deve girare il robot quale deve essere il raggio della curva delle singole ruote
        wheel = 'right'
        right_wheel_turn_radius = self.calculate_wheel_turn_radius(body_turn_radius, wheel)
        wheel = 'left'
        left_wheel_turn_radius = self.calculate_wheel_turn_radius(body_turn_radius, wheel)

        # calcoliamo la velocità di rotazione delle singole ruote
        right_wheel_rpm = self.calculate_wheel_rpm(ls, a_s, right_wheel_turn_radius)
        left_wheel_rpm = self.calculate_wheel_rpm(ls, a_s, left_wheel_turn_radius)

        self.set_wheel_movement(right_wheel_rpm, left_wheel_rpm)
