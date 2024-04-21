import cv2

class nms:
    def __init__(self):
        super().__init__()

    '''
    Se encarga de realizar el algoritmo Non-Maximun Suppresion (NMS)

    Parámetros:
    boxes (list): Contiene todas las cajas detectadas con sus coordenadas respectivas.
    scores (list): Contiene un número que corresponde a la puntuación de cada caja.
    umbral (int): Umbral.

    Retorna:
    list: Lista que contiene las cajas filtradas.
    '''
    def call(self, boxes, scores, umbral):
        scores, boxes = self.reorderScores(scores, boxes)
        boxesAux = boxes.copy()
        finalBoxes = []
        while(len(boxesAux) != 0):
            box = boxesAux[0]
            finalBoxes.append(box)
            boxesAux.remove(box)
            i = 0
            while i < len(boxesAux):
                remainBox = boxesAux[i]
                iou = self.intersectionOverUnion(box, remainBox)
                if iou >= umbral:
                    del boxesAux[i]
                else:
                    i += 1
        return finalBoxes

    '''
    Reordena las dos listas de forma descendente en función de las puntuaciones

    Parámetros:
    scores (list): Contiene un número que corresponde a la puntuación de cada caja .
    boxes (list): Contiene todas las cajas detectadas con sus coordenadas respectivas.

    Retorna:
    tuple: Tupla de dos listas reordenadas.
    '''
    def reorderScores(self, scores, boxes):
        boxesAux = []
        scoresAux = []
        for i in range (len(scores)):
            score = scores[i]
            box = boxes[i]
            if(len(scoresAux) == 0):
                scoresAux.append(score)
                boxesAux.append(box)
            else:
                for j in range(len(scoresAux)):
                    if (score >= scoresAux[j]):
                        scoresAux.insert(j, score)
                        boxesAux.insert(j, box)
                        break
                    elif (j == len(scoresAux)-1 and score < scoresAux[j]):
                        scoresAux.append(score)
                        boxesAux.append(box)
                        break
        return scoresAux, boxesAux
    
    '''
    Aplica la intersección sobre la unión sobre dos cajas

    Parámetros:
    box1 (list): Contiene las coordenadas de la caja 1 .
    box2 (list): Contiene las coordenadas de la caja 2 .

    Retorna:
    float: Resultado de aplicar IoU.
    '''
    def intersectionOverUnion(self, box1, box2):
        x, y, w, h = cv2.boundingRect(box1)
        xA1 = x
        yA1 = y
        x = x + w
        y = y + h
        xA2 = x
        yA2 = y
        x, y, w, h = cv2.boundingRect(box2)
        xB1 = x
        yB1 = y
        x = x + w
        y = y + h
        xB2 = x
        yB2 = y
        x1 = max(xA1, xB1)
        y1 = max(yA1, yB1)
        x2 = min(xA2, xB2)
        y2 = min(yA2, yB2)
        intersectionArea = (x2 - x1) * (y2 - y1)
        if (intersectionArea < 0):
            intersectionArea = 0
        box1Area = abs((xA2 - xA1) * (yA1 - yA2))
        box2Area = abs((xB2 - xB1) * (yB1 - yB2))
        return intersectionArea / (box1Area + box2Area - intersectionArea + 1e-6)