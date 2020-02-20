# coding=utf8
import numpy as np
import cv2
from scipy import interpolate
import csv

# classe utilizzata per calcolare la distanza di un oggetto rilevato dalla fotocamera.

class Distance:

    def __init__(self):
        # creazione dei vettori x e y con i valori contenuti all'interno del file csv
        self.x = np.array([])
        self.y = np.array([])
        file_name = open('/home/pi/ros_catkin_ws/src/robot/src/scripts/Obstacle_little.csv', 'rt')
        reader = csv.reader(file_name)
        for row in reader:
            self.x = np.append(self.x, row[1])
            self.y = np.append(self.y, row[0])
        self.x = self.x.astype(float)
        self.y = self.y.astype(float)
    # funzione che dato un vettore di contorni restituisce l'area e il centroide del contorno più grande
    def aux(self, cnts, image):
        area = 0
        target_x = 0
        # se cnts è il vettore vuoto restituisce 0,0
        if not cnts:
            return 0, 0
        else:
            if len(cnts) > 0:
                # si inizializza la variabile che individua il contorno più grande a 0
                largest = 0
                for i in range(len(cnts)):
                    # prende l'area del contorno i-esimo
                    temp_area = cv2.contourArea(cnts[i])
                    # se è l'area più grande vista, la prende
                    if temp_area > area:
                        area = temp_area
                        largest = i
                # si calcolano i momenti del contorno più grande
                coordinates = cv2.moments(cnts[largest])
                img_cnts = cv2.drawContours(image.copy(), cnts[largest], -1, (40, 255, 255))

                if coordinates['m00'] != 0:
                    # si calcola la coordinata x del centroide
                    target_x = int(coordinates['m10'] / coordinates['m00'])
                else:
                    target_x = 0
        return area, target_x
# funzione che permette di calcolare l'area e la coordinata x del centroide dell'area massima,
# individuata con un range di colore, all'interno dell'immagine passata come argomento

    def find_area(self, image):
        final_area = 0
        final_target = 0
        # elaborazione per migliorare il rilevamento degli oggetti
        image_blur = cv2.GaussianBlur(image, (5, 5), 0)
        img_hsv = cv2.cvtColor(image_blur, cv2.COLOR_BGR2HSV)

        # estremi dei vari range di colore utilizzati per il filtraggio
        lowred = np.array([0, 132, 106])
        highred = np.array([21, 255, 255])


        lowgreen = np.array([41, 147, 50])
        highgreen = np.array([90, 255, 255])

        lowblue = np.array([72, 131,77])
        highblue = np.array([110, 255, 255])

        # immagini binarie in cui sono indicati in bianco i pixel i cui valori hsv ricadono all'interno dei diversi range
        # mentre il background è nero
        green_mask = cv2.inRange(img_hsv, lowgreen, highgreen)
        blue_mask = cv2.inRange(img_hsv, lowblue, highblue)
        red_mask = cv2.inRange(img_hsv, lowred, highred)

        mask_filter_green = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask_filter_red = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask_filter_blue = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # immagini in cui sono indicati i bordi delle aree
        edged_img_green = cv2.Canny(mask_filter_green.copy(), 35, 125)
        edged_img_red = cv2.Canny(mask_filter_red.copy(), 35, 125)
        edged_img_blue = cv2.Canny(mask_filter_blue.copy(), 35, 125)

        # inizializzazione dei vettori risultato delle aree e dei centroidi
        area_results = np.array([])
        target_results = np.array([])

        # otteniamo i tre vettori con tutti i contorni chiusi delle tre immagini binarie
        cnts_green, hierarchy_green = cv2.findContours(edged_img_green.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_red, hierarchy_red = cv2.findContours(edged_img_red.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_blue, hierarchy_blue = cv2.findContours(edged_img_blue.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # calcoliamo l'area massima e il centroide per ciascuno dei tre vettori
        area_green, target_green = self.aux(cnts_green, image)
        area_blue, target_blue = self.aux(cnts_blue, image)
        area_red, target_red = self.aux(cnts_red, image)

        area_results = np.append(area_results, area_green)
        area_results = np.append(area_results, area_blue)
        area_results = np.append(area_results, area_red)

        target_results = np.append(target_results, target_green)
        target_results = np.append(target_results, target_blue)
        target_results = np.append(target_results, target_red)

        # iterazione sul vettore delle aree
        for i in area_results:
            # se si trova un valore di area nel vettore che è maggiore di quella corrente si aggiorna l'area finale
            # e si aggiorna anche il centroide corrispondente
            if i > final_area:
                final_area = i
                index = np.where(area_results == i)
                final_target = target_results[index]

        return final_area, final_target

    # funzione che data in ingresso un valore di area tramite un'interpolazione restituisce la distanza
    def distancetoCamera(self, sup):
	# se l'area è maggiore del valore più alto possibile si restituisce 0 e l'ostacolo è considerato imminente
        print(sup)
        if sup > 111459.0:
            return 200
	# se l'area è minore del valore più basso possibile si restituisce 200 e l'ostacolo è considerato lontano o non visibile
        elif sup < 3937.5:
            return 200
        else:
            f = interpolate.interp1d(self.x, self.y)
            return f(sup)
