from preprocessor.preprocessor import Preprocessor
from detector.color_detector import ColorDetector

dir = "imagenesTest"
preprocessor = Preprocessor()
color_detector = ColorDetector()    
imagen_list = None
gray_images = None

def image_upload():
    imagen_list = preprocessor.load_all_images(dir)
    gray_images = preprocessor.convert_to_gray(imagen_list)
    preprocessor.show_first_and_last(imagen_list)
    preprocessor.show_first_and_last(gray_images)
    
def detection_with_correlation_masks():
    color_detector.show_ideal_mask()
    color_detector.cut_detected_zones(None)
    color_detector.apply_blue_filter(None)

if __name__ == "__main__":    
    image_upload()
    detection_with_correlation_masks()