import cv2
from matplotlib import pyplot as plt
from preprocessor.preprocessor import Preprocessor
from detector.color_detector import ColorDetector
from preprocessor.loader import Loader
from preprocessor.converter import Converter
from detector.mser_detector import MSERDetector

preprocessor = Preprocessor()
color_detector = ColorDetector()   
loader = Loader()
converter = Converter()
mser_detector = MSERDetector()
imagen_list = None
gray_images = None

if __name__ == "__main__":  
    dir = 'imagenesTest' # Nombre de la carpeta de imagenes 
    imagen_list = loader.load_all_images(dir) 
    gray_images = converter.convert_to_gray(imagen_list) 
    
    # Llamamos a mser para detectar las regiones de cada imagen 
    regiones, mascaras, regiones_dibujadas = mser_detector.mser(imagen_list, gray_images)

    # Dibujamos un rectangulo alrededor de cada region detectada con mser 
    list_rectangles = mser_detector.rectangle_of_regions(imagen_list, mascaras, regiones_dibujadas)

    # Filtramos los rectangulos para quedarnos con los que cumplan la relacion de aspecto
    list_filtered_rectangles, list_filtered_regions = mser_detector.rectangle_filtered(imagen_list, mascaras, regiones_dibujadas)

    # Ampliamos los rectangulos anteriores y extraemos en una ventana cada rectangulo ampliado de la imagen 
    lista_coordenadas_regiones = mser_detector.extract_enlarge_rectangles(imagen_list, list_filtered_regions)


