import cv2
from matplotlib import pyplot as plt
from preprocessor.preprocessor import Preprocessor
from detector.color_detector import ColorDetector
from detector.mser_detector import MSERDetector

dir = "imagenesTest"
preprocessor = Preprocessor()
color_detector = ColorDetector()    
mser_detector = MSERDetector()
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
    regiones, mascaras, regiones_dibujadas = mser_detector.mser(imagen_list, gray_images)
    # Dibujamos un rectangulo alrededor de cada region detectada con mser 
    list_rectangles = mser_detector.rectangle_of_regions(imagen_list, mascaras, regiones_dibujadas)
    # Filtramos los rectangulos para quedarnos con los que cumplan la relacion de aspecto
    list_filtered_rectangles, list_filtered_regions = mser_detector.rectangle_filtered(imagen_list, mascaras, regiones_dibujadas)
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
    
    # Calculamos el Score para cada 
    # mack_and_score_list = [ [((x, y, w, h), score)] ] 
    #              Lista principal hace referencia a la imagen entera
    #              Lista interna a los subpaneles encontrados en la imagen entera
    score_list = color_detector.all_correlation(masks_per_image, coords_list)
    
    # score_list = [[(cord, score)]] = [[((x, y, w, h), score)]]
    return score_list
if __name__ == "__main__":    
    
    image_upload()
    detected_coords = mser_detection()
    scoreds_and_coords = detection_with_correlation_masks(detected_coords)
