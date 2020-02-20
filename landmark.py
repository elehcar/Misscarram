#!/usr/bin/env python
# coding=utf8

import cv2
import numpy as np
from scipy import interpolate
import csv

# classe utilizzata per ottenere le informazioni contenute nel qr code del landmark e la distanza della camera da esso
class Landmark:

    def __init__(self):
        self.distance = 200
        self.qr_data = "11"
        self.detector = cv2.QRCodeDetector()
	# creazione dei vettori in cui vengono inseriti con un'iterazione
	# i valori delle distanze contenuti dal file landmark.csv
        self.x = np.array([]) # vettore delle aree
        self.y = np.array([]) # vettore delle distanze
        file_name = open('/home/pi/ros_catkin_ws/src/robot/src/scripts/landmark_measure.csv', 'rt')
        reader = csv.reader(file_name)
        for row in reader:
            self.x = np.append(self.x, row[1])
            self.y = np.append(self.y, row[0])
        self.x = self.x.astype(float)
        self.y = self.y.astype(float)
    # funzione per ottenere informazioni dal landmark all'interno dell'immagine passata
    # come argomento
    def get_qrdata(self, img):
        data, bbox, _ = self.detector.detectAndDecode(img)
        if data:  # aggiorno distanza e qr_code quando vedo un landmark
            self.distance = self.get_distance(bbox)
            self.qr_data = data
        else:
            self.distance = 200
            self.qr_data = "11"
        return self.qr_data, self.distance

    # funzione per ottenere la distanza di un landmark identificato dal vettore di punti "points" passato come argomento 
    def get_distance(self, points):
        x1 = points.item(0)
        x2 = points.item(2)
        y2 = points.item(3)
        y3 = points.item(5)
	# calcolo l'area come base per altezza con i vertici opportuni
        h = abs(x1-x2)
        b = abs(y2-y3)
        area = h*b
	# se l'area è maggiore di quella massima contenuta nel file si restituisce 0
        if area > 1399.06:
            dist = 0 # landmark troppo vicino
	# se l'area è minore di quella minima contenuta nel file si restituisce 200
        elif area < 558.97:
            dist = 200 # landmark troppo lontano o non visibile
	# se non siamo in nessuno dei due casi precedenti ricaviamo la distanza 
	# attraverso un'interpolazione sui vettori di area e distanza ricavati dai valori del file landmark.csv        
        else:
            f = interpolate.interp1d(self.x, self.y)
            dist = f(area)
        return dist
