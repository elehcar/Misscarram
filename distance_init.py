# coding=utf8

import csv
import sys
import cv2
import numpy as np
from picamera import PiCamera
from picamera.array import PiRGBArray
import time
# codice utilizzato per inizializzare i file .csv delle misure delle distanze dagli ostacoli e dai landmark del percorso

# metodo che permette di calcolare l'area e la coordinata x del centroide della superficie, individuata con un range di colore, di dimensione massima all'interno dell'immagine passata come argomento 
def find_area(image):
    # immagine sfumata per eliminare eventuale rumore
    image_blur = cv2.GaussianBlur(image, (5, 5), 0)
    # conversione dell'immagine da BGR ad HSV
    img_hsv = cv2.cvtColor(image_blur, cv2.COLOR_BGR2HSV)
    # estremi del range di colore utilizzato per il filtraggio
    lowblue = np.array([80, 70, 20])
    highblue = np.array([120, 255, 255])
    # immagine binaria in cui con il bianco sono indicate le aree in cui pixel hanno valori HSV che ricadono all'interno del range
    # il background è nero
    blue_mask = cv2.inRange(img_hsv, lowblue, highblue)
    mask_filter = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # Immagine in cui sono indicati i bordi delle aree
    edged_img = cv2.Canny(mask_filter.copy(), 35, 125)
    # cv2.imshow('Edged', edged_img)
    # cv2.waitKey(1000)
    # restituisce tutti i contorni chiusi di un'immagine binaria 
    cnts, hierarchy = cv2.findContours(edged_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # se non ci sono contorni rilevati si restituisce 0 come area e 0 per il target_x
    if not cnts:
        return 0, 0
    else:
	# se invece ci sono dei contorni rilevati si inizializzano l'area dell'ostacolo e il target_x a 0
        area_obstacle = 0
        target_x = 0
        if len(cnts) > 0:
	    # si inizializza la variabile che individua il contorno più grande a 0 
            largest = 0
	    # per ogni contorno restituito dalla find contours
            for i in range(len(cnts)):
                # si ottiene l'area dell'iesimo contorno
                temp_area = cv2.contourArea(cnts[i])
                # se l'area ottenuta è la più grande che abbiamo incontrato, la prendiamo e il contorno più grande diventa l'iesimo
                if temp_area > area_obstacle:
                    area_obstacle = temp_area
                    largest = i
            # calcola i momenti del contorno più grande
            coordinates = cv2.moments(cnts[largest])
            # img_cnts = cv2.drawContours(image.copy(), cnts[largest], -1, (40, 255, 255))

            if coordinates["m00"] != 0:
		# calcoliamo la coordinata x del centroide con la seguente formula 
                target_x = int(coordinates['m10'] / coordinates['m00'])
            else:
                target_x = 0
   
    return area_obstacle, target_x

# funzione per calcolare l'area di un qr code presente nell'immagine
def get_qrarea(image):
    detector = cv2.QRCodeDetector()
    # ottengo i dati e i punti che individuano i vertici del qr code
    data, points, _ = detector.detectAndDecode(image)
    area_qr = 0
    # se dei dati sono effettivamente rilevati
    if data:
        x1 = points.item(0)
        x2 = points.item(2)
        y2 = points.item(3)
        y3 = points.item(5)
	# calcolo base e altezza e le moltiplico per ottenere l'area
        h = abs(x1 - x2)
        b = abs(y2 - y3)
        area_qr = h * b
    return area_qr


camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
time.sleep(2)
f = open(sys.argv[1], 'wt')
writer = csv.writer(f)
j = 50
# inserimento delle righe nel file csv finchè la variabile j < 150 come coppia (indice, area dell'immagine)
while j < 110:
    camera.capture(rawCapture, format='bgr')
    imm = rawCapture.array
    if imm is not None:
        area = get_qrarea(imm)
        if not area == 0:
            j = j + 10
            print("OK!")
            writer.writerow((j, area))
    cv2.imshow('Imm', imm)
    cv2.waitKey(0)
    rawCapture.truncate(0)
