from preprocessor.preprocessor import Preprocessor
dir = "imagenesTest"
if __name__ == "__main__":
    preprocessor = Preprocessor()
    
    imagen_list = preprocessor.load_images(dir)
    preprocessor.show_first_and_last(imagen_list)

    gray_images = preprocessor.convert_to_gray(imagen_list)
    preprocessor.show_first_and_last(gray_images)
    