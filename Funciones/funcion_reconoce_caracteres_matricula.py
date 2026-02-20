import numpy as np
import cv2

def funcion_reconoce_caracteres_matricula(IsegEtiq, numObjetos, contornos):

    data = np.load('Material_Imagenes_Plantillas/00_Plantillas/plantillas.npz', allow_pickle=True)
    plantillas = data['plantillas']

    Caracteres = '0123456789ABCDFGHKLNRSTXYZ'
    AngulosPosibles = [-9, -6, -3, 0, 3, 6, 9]

    numCaracteres = len(Caracteres)
    numAngulos = len(AngulosPosibles)

    prediccion = ''
    angulos = np.zeros(numObjetos)
    datos = np.zeros((numCaracteres, numAngulos, numObjetos))

    for i in range(numObjetos):

        x,y,w,h = contornos[i+1]

        Icaracter = IsegEtiq[y:y+(h+1), x:x+(w+1)] != 0

        for j in range(numCaracteres):
            for k in range(numAngulos):

                redim = cv2.resize(Icaracter.astype(np.uint8), (plantillas[j, k].shape[1], plantillas[j, k].shape[0]), interpolation=cv2.INTER_NEAREST)

                datos[j, k, i] = Funcion_CorrelacionEntreMatrices(redim, plantillas[j, k])

        fila, columna = np.where(datos[:, :, i] == np.max(datos[:, :, i]))
        
        fila = fila[0]
        columna = columna[0]

        prediccion += Caracteres[fila]
        angulos[i] = angulos[columna]

    return prediccion, angulos, datos


def Funcion_CorrelacionEntreMatrices(Matriz1, Matriz2):
    Matriz1 = Matriz1.astype(np.float64)
    Matriz2 = Matriz2.astype(np.float64)

    Media1 = np.mean(Matriz1)
    Media2 = np.mean(Matriz2)

    numerador = np.sum((Matriz1 - Media1) * (Matriz2 - Media2))
    denominador = np.sqrt(np.sum((Matriz1 - Media1)**2) * np.sum((Matriz2 - Media2)**2))

    eps = np.finfo(float).eps

    ValorCorrelacion = numerador / (denominador + eps)

    return ValorCorrelacion
