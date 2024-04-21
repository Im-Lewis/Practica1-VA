import cv2
import numpy as np


class MSERDetector:
    def __init__(self):
            super().__init__()
    
    '''
        Recibe una lista de imagenes a color y una lista de imagenes en grises 
        Devuelve tres listas:
            La primera una lista de listas, cada una de ellas con las regiones de cada imagen 
            La segunda una lista de mascaras donde estan guardadas las regiones detectadas 
            La tercera una lista de las regiones dibujadas 
        Ejemplo de llamada a la funcion:
            regiones, mascaras, regiones_dibujadas = mser(listaImagenes, listaGrises)
    '''
    def mser(self, list_images:list[np.ndarray], gray_images:list[np.ndarray]): 
        list_images_regions = [] # Lista de imagenes con las regiones dibujadas
        list_of_regions = [] # Lista de regiones de una imagen

        for img, gray_img in zip(list_images, gray_images):  
            copy_img = img.copy()
            copy_img_gray = gray_img.copy()
            # Umbralizado adaptativo usando la media
            img_med = cv2.adaptiveThreshold(copy_img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,21, 10)

            # Canny para obtener la imagen de bordes
            filtered_image = cv2.Canny(img_med, 100, 200)

            # Dilatamos los bordes de la imagen
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,3))
            dilated_image = cv2.dilate(filtered_image, kernel)

            # Detectamos los contornos
            mser = cv2.MSER_create(delta=10, max_variation=0.1, min_area=1000, max_area=45000)
            polygons, _ = mser.detectRegions(dilated_image)
            list_of_regions.append(polygons)

            # Envolvemos los puntos en un "casco convexo" mas ajustado
            hulls = []
            for p in polygons:
                hull = cv2.convexHull(p.reshape(-1, 1, 2))
                hulls.append(hull)
            
            # Dibujamos las regiones
            cv2.polylines(copy_img, hulls, 1, (0, 255, 0), 2)
            list_images_regions.append(copy_img)
        # TODO
        # Mostramos la primera y la ultima imagen 
        # cv2.imshow('Detected Traffic Signs', list_images_regions[0])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imshow('Detected Traffic Signs', list_images_regions[-1])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return list_images_regions, list_of_regions
    

    '''
        Recibe una lista de imagenes a color, una lista de las mascaras detectadas con mser y la lista de las regiones dibujadas en mser 
        Devuelve la lista de las imagenes con los rectangulos dibujados rodeando la zona detectada con mser()
        Ejemplo de llamada a una funcion:
            list_rectangles = rectangle_of_regions(listaImagenes, masks_list, regions_list)
    '''
    def rectangle_of_regions(self, list_images:list[np.ndarray], list_regions):
        list_images_with_rectangles = [] # Lista de imagenes con las regiones dentro de un rectangulo
        for image, regions in zip(list_images, list_regions):
            img_copy = image.copy()
            for region in regions:
                x, y, w, h = cv2.boundingRect(region)
                cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
            list_images_with_rectangles.append(img_copy)
        
        return list_images_with_rectangles
        # TODO
        # Mostramos la primera y la ultima imagen 
        # cv2.imshow('Detected Traffic Signs', list_images_rectangles[0])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imshow('Detected Traffic Signs', list_images_rectangles[-1])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return list_images_rectangles
    

    '''
        Recibe una lista de imagenes a color, una lista de las mascaras detectadas con mser y la lista de las regiones dibujadas en mser 
        Devuelve dos listas: 
            La primera es una lista de imagenes con los rectangulos dibujados 
            La segunda es una lista de las regiones(rectangulos) que cumplen con la relacion de aspecto 
        Ejemplo de llamada a la funcion:
            list_filtered_rectangles, list_filtered_regions = rectangle_filtered(listaImagenes, masks_list, regions_list)
    '''
    def rectangle_filtered(self, list_images:list[np.ndarray], list_regions):
        list_filtered_image_with_rectangles = [] # Lista de imagenes con los rectangulos que cumplen el tamanyo
        list_filtered_regions = [] # Lista de regiones filtradas 

        for image, regions in zip(list_images, list_regions):
            img_copy = image.copy()
            filtered_regions_act_img = [] # Regiones filtradas de la imagen actual

            for region in regions:
                x, y, w, h = cv2.boundingRect(region)
                aspect_ratio = w / float(h)
                
                # Filtramos las regiones en funcion de la relacion de aspecto 
                if aspect_ratio > 0.7 and aspect_ratio < 6.5:
                    filtered_regions_act_img.append(region)
                    cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)

            list_filtered_regions.append(filtered_regions_act_img)
            list_filtered_image_with_rectangles.append(img_copy)
        
        # TODO
        # Mostramos la primera y la ultima imagen 
        # cv2.imshow('Detected Traffic Signs', list_filtered_rectangles[0])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imshow('Detected Traffic Signs', list_filtered_rectangles[-1])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        return list_filtered_image_with_rectangles, list_filtered_regions 
    

    '''
        Recibe una lista de las imagenes a color y una listas con las regiones filtradas de las imagenes
        Devuelve una lista de las regiones de cada imagen con el recuadro ampliado 
        Ejemplo de llamada a la funcion:
            lista_coordenadas_regiones = extraer_regiones_rectangulares(listaImagenes, list_filtered_regions)

        NOTA:
        La lista devuelta por la funcion es de este tipo: 
        [[(x,y,w,h), (x,y,w,h), ...], [(x,y,w,h), (x,y,w,h), ...], ...]

        La lista contiene listas de tuplas, una lista de tuplas por cada imagen y en la misma posicion de lista que la imagen a la que corresponde 
        Cada lista de tuplas contiene una tupla, cada una de ellas pertenece a cada region detectada en la imagen, y contiene los valores x, y, w, h de la region
    '''
    def extract_enlarge_rectangles(self, list_images, list_regions):
        regions_of_images = [] # Lista de listas de las regiones de cada imagen
        for i,img in enumerate(list_images): 
            regions_of_actual_image = [] # Lista de coordenadas de las regiones de la imagen actual
            regiones_rectangulares = [] # Lista de rectangulos de la imagen actual 
            for region in list_regions[i]:
                x, y, w, h = cv2.boundingRect(region)

                # Ajustar los límites del rectángulo para que no excedan los límites de la imagen
                x_new = max(0, x - 10)  
                y_new = max(0, y - 10)  
                w_new = min(img.shape[1] - x_new, w + 20)  
                h_new = min(img.shape[0] - y_new, h + 20)  
                tuple_of_coords = (x_new, y_new, w_new, h_new)

                rectangulo = cv2.rectangle(list_images[i].copy(), (x_new, y_new), (x_new + w_new, y_new + h_new), (0, 0, 255), 2)
                pixels_region = list_images[i][y_new:y_new+h_new, x_new:x_new+w_new]
                regiones_rectangulares.append((rectangulo, pixels_region))
                regions_of_actual_image.append(tuple_of_coords) # Anyadimos las coordenadas a de la region a la lista
            # TODO
            # Recortamos los rectangulos de las regiones de la primera y ultima imagen y los mostramos en una ventana aparte
            # if(i == 0) or (i == 101):
            #     for rectangulo, pixels_region in regiones_rectangulares:
            #         cv2.imshow('rectangulo', rectangulo)
            #         cv2.imshow('region detectada', pixels_region)
            #         cv2.waitKey(0)
            #         cv2.destroyAllWindows()

            regions_of_images.append(regions_of_actual_image) # Anyadimos la lista de coordenadas de las regiones de la imagen actual 
        return regions_of_images