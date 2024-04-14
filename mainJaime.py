import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def cargarImagenesColor(path):
    filesImages = os.listdir(path)
    imagenesColor = []
    for file in filesImages:
        imagen = cv2.imread(path+"/"+file)
        imagenesColor.append(imagen)
    return imagenesColor

def transformarGrises(imagenesColor):
    imagenesGrises = []
    for imagen in imagenesColor:
        imagenesGrises.append(cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY))
    return imagenesGrises

def equializarImagenes(imagenes):
    imagenesEqualizadas = []
    for imagen in imagenes:
        imagenesEqualizadas.append(cv2.equalizeHist(imagen))
    return imagenesEqualizadas

def MSER(imagen):
    output = np.zeros((imagen.shape[0], imagen.shape[1]), dtype=np.uint8)  # Salida en escala de grises
    mser = cv2.MSER_create(delta=10, min_area=3000, max_area=50000, max_variation=0.1, min_diversity=0.2, max_evolution=200, area_threshold=1.01, min_margin=0.003, edge_blur_size=5)
    #mser = cv2.MSER_create(delta=20, max_variation=0.1, min_area = 10000, max_area=60000)
    #mser = cv2.MSER_create(delta=10, max_variation=1, min_area = 30000, max_area=60000)
    polygons, _ = mser.detectRegions(imagen)
    for polygon in polygons:
        mask = np.zeros((imagen.shape[0], imagen.shape[1]), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        output = cv2.bitwise_or(output, mask)
    return output, polygons


def graficar(imagen1, imagen2):
    plt.subplot(1, 2, 1)
    plt.imshow(imagen1, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(imagen2, cmap="gray")
    plt.show()

def umbralizar(imagen):
    salida = imagen.copy()
    alto, ancho = imagen.shape
    for i in range(ancho):
        for j in range(alto):
            if (imagen[j][i] >= 128):
                salida[j][i] = 255
    return salida


def erosionar(imagen):
    salida = imagen.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    salida = cv2.erode(salida,kernel)
    return salida

def obtenerCartel(regionDetectada, imagen):
    imagenCopia = imagen.copy()
    for polygon in regionDetectada:
        x, y, w, h = cv2.boundingRect(polygon)
        cv2.rectangle(imagenCopia, (x, y), (x + w, y + h), (255, 0, 0), 10)
    return imagenCopia

def obtenerSubpanelesFiltrado(regionDetectada, imagen):
    imagenCopia = imagen.copy()
    for polygon in regionDetectada:
        x, y, w, h = cv2.boundingRect(polygon)
        relacionAspecto = w/h
        if (relacionAspecto >= 2 and relacionAspecto < 4):
            cv2.rectangle(imagenCopia, (x, y), (x + w, y + h), (255, 0, 0), 10)
    return imagenCopia

if __name__ == '__main__':
    while True:
        imagenesColor = cargarImagenesColor("imagenesTest")
        imagenesGrises = transformarGrises(imagenesColor)
        imagenesEqualizadas = equializarImagenes(imagenesGrises)
        print("Index of list:")
        i = input()
        imagen = imagenesEqualizadas[int(i)]
        imagen = umbralizar(imagen)
        imagenMSER, regionesDetectadas = MSER(imagen)
        graficar(imagen, imagenMSER)

        imagen = cv2.cvtColor(imagenesColor[int(i)], cv2.COLOR_BGR2RGB)
        imagenRectangulos = obtenerCartel(regionesDetectadas, imagen)
        imagenSubpaneles = obtenerSubpanelesFiltrado(regionesDetectadas, imagen)
        graficar(imagenRectangulos, imagenSubpaneles)
