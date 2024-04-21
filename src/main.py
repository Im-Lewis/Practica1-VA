import cv2
from matplotlib import pyplot as plt
from preprocessor.preprocessor import Preprocessor
from detector.color_detector import ColorDetector
from detector.mser_detector import MSERDetector
from detector.nms import NMS
import shutil
import os

dir = "imagenesTest"
preprocessor = Preprocessor()
color_detector = ColorDetector()    
mser_detector = MSERDetector()
nms_detector = NMS()
imagen_list = None
gray_images = None

def image_upload():
    global imagen_list
    global gray_images
    
    imagen_list = preprocessor.load_all_images(dir)
    gray_images = preprocessor.convert_to_gray(imagen_list)
    preprocessor.show_first_and_last(imagen_list)
    preprocessor.show_first_and_last(gray_images)
    
def mser_detection():
    global imagen_list
    global gray_images
    
    # Llamamos a mser para detectar las regiones de cada imagen 
    list_images_with_regions, regiones = mser_detector.mser(imagen_list, gray_images)
    # Dibujamos un rectangulo alrededor de cada region detectada con mser 
    list_images_with_rectangles = mser_detector.rectangle_of_regions(imagen_list, regiones)
    # Filtramos los rectangulos para quedarnos con los que cumplan la relacion de aspecto
    list_filtered_image_with_rectangles, list_filtered_regions = mser_detector.rectangle_filtered(imagen_list, regiones)
    # Ampliamos los rectangulos anteriores y extraemos en una ventana cada rectangulo ampliado de la imagen 
    lista_coordenadas_regiones = mser_detector.extract_enlarge_rectangles(imagen_list, list_filtered_regions)
    
    return lista_coordenadas_regiones


def detection_with_correlation_masks(coords_list):
    global imagen_list
    
    # Mostramos la mascara ideal
    color_detector.show_ideal_mask()
    
    # Lista donde cada posicion es una lista de sub-carteles de una imagen de imagen_list 
    # detected_zones = [[subpanel, subpabel...], [subpanel], ...]
    detected_zones = color_detector.cut_detected_zones(imagen_list, coords_list)
    
    # Filtrado azul a cada mascara detectada                              
    masks_per_image = color_detector.apply_all_blue_filter(detected_zones)
    
    # Calculamos el Score para cada imagen y paneles
    #              Lista principal hace referencia a la imagen entera
    #              Lista interna a los subpaneles encontrados en la imagen entera
    score_list = color_detector.all_correlation(masks_per_image, coords_list)
    
    # score_list = [[(cord, score)]] = [[((x, y, w, h), score)]]
    return score_list

def remove_repeated_regions(scores_and_coords):
    # Borramos las imagenes que vamos a guardar y volvemos a crear el directorio
    if not os.path.exists('imagenesFinales'):
        os.makedirs('imagenesFinales')
    shutil.rmtree('imagenesFinales')
    os.makedirs('imagenesFinales')

    # Lista con las coordenadas de los cuadrados finales
    filtered_coords = []
    count = 0

    # Recoremos la lista que contiene las puntuaciones y coordendas de cada imagen
    for score_and_coords in scores_and_coords:
        scores = []
        boxes = []
        # Recoremos la lista de una imagen y añadimos sus puntuaciones y las coordenadas de las boxes
        for elem in score_and_coords:
            boxes.append(elem[0])
            scores.append(elem[1])
        imagen = imagen_list[count]
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        # Calculamos el NMS para la imagen seleccionada
        filtered_boxes = nms_detector.call(boxes, scores, 0.1)
        # Pintamos las regiones filtradas sobre la imagen dada
        imagenFiltrada = drawBoxesImage(filtered_boxes, imagen)
        imagenFiltrada = cv2.cvtColor(imagenFiltrada, cv2.COLOR_BGR2RGB)
        cv2.imwrite('imagenesFinales/'+str(count)+'.png', imagenFiltrada)
        count = count + 1
    return filtered_coords

def drawBoxesImage(regionDetectada, imagen):
    imagenCopia = imagen.copy()
    for polygon in regionDetectada:
        x, y, w, h = polygon
        cv2.rectangle(imagenCopia, (x, y), (x + w, y + h), (255, 0, 0), 10)
    return imagenCopia


if __name__ == "__main__":    
    
    image_upload()
    detected_coords = mser_detection()
    scoreds_and_coords = detection_with_correlation_masks(detected_coords)
    remove_repeated_regions(scoreds_and_coords)
