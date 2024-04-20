from preprocessor.loader import Loader
from preprocessor.converter import Converter
from matplotlib import pyplot as plt
import cv2
import numpy as np

class Preprocessor(Loader, Converter):
    def __init__(self):
        None
        
    def cut_detected_zones(self, imagenes, coords  = [[(1300, 400, 270, 170)]]):
        detected_subimages = []
        for coord, imagen in zip(coords, imagenes):
            for c in coord:
                im = self.cut_zone(imagen, c)
                detected_subimages.append(im)
        return detected_subimages
    
    def cut_zone(self, image, coord):
        (x,y,w,h) = coord
        im = image[y:y+h, x:x+w]
        resized_image = cv2.resize(im, (80,40))
        return resized_image
    
    '''
        Recibe una lista de imagenes a color y una lista de imagenes en grises 
        Devuelve tres listas:
            La primera una lista de listas, cada una de ellas con las regiones de cada imagen 
            La segunda una lista de mascaras donde estan guardadas las regiones detectadas 
            La tercera una lista de las regiones dibujadas 
        Ejemplo de llamada a la funcion:
            regiones, mascaras, regiones_dibujadas = mser(listaImagenes, listaGrises)
    '''
    def mser(list_images, grey_images): 
        list_images_regions = [] # Lista de imagenes con las regiones dibujadas
        list_of_masks = [] # Lista de listas de regiones detectadas en cada imagen 
        list_of_regions = [] # Lista de regiones de una imagen

        for i, img in enumerate(grey_images):  
            copy = list_images[i].copy()

            # Umbralizado adaptativo usando la media
            img_med = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,21, 10)

            # Canny para obtener la imagen de bordes
            filtered_image = cv2.Canny(img_med, 100, 200)

            # Dilatamos la imagen para hacer los bordes mas gruesos
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,3))
            dilated_image = cv2.dilate(filtered_image, kernel)

            mser = cv2.MSER_create(delta=10, max_variation=0.1, min_area=1000, max_area=45000)
            polygons, _ = mser.detectRegions(dilated_image)
            list_of_regions.append(polygons)

            # Creamos una mascara para almacenar las regiones detectadas 
            mask = np.zeros((dilated_image.shape[0], img.shape[1]), dtype=np.uint8)
            list_of_masks.append(mask)

            hulls = []
            for p in polygons:
                hull = cv2.convexHull(p.reshape(-1, 1, 2))
                hulls.append(hull)
            
            # Dibujamos las regiones
            cv2.polylines(copy, hulls, 1, (0, 255, 0), 2)
            list_images_regions.append(copy)

        # Mostramos la primera y la ultima imagen 
        cv2.imshow('Detected Traffic Signs', list_images_regions[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow('Detected Traffic Signs', list_images_regions[-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return list_images_regions, list_of_masks, list_of_regions
    
    '''
        Recibe una lista de imagenes a color, una lista de las mascaras detectadas con mser y la lista de las regiones dibujadas en mser 
    '''
    def mser(list_images, grey_images):