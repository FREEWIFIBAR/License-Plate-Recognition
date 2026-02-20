import numpy as np
import cv2
import matplotlib.pyplot as plt # pip install matplotlib

def funcion_segmenta_caracteres_matricula(Ic_placa, flagFiguras):

    # Procesamiento
    R = Ic_placa[:, :, 2]
    RF = R.astype(np.float32)

    f, c = RF.shape

    W = np.round(((f * c) * 9) / (175 * 1092))
    W = int(W) if W % 2 != 0 else int(W + 1)

    RG = cv2.GaussianBlur(RF, (W, W), W/5, borderType=cv2.BORDER_REPLICATE)

    V = np.round(((f * c) * 5) / (175 * 1092))
    V = int(V) if V % 2 != 0 else int(V + 1)

    kernel = np.ones((V, V), np.uint8)

    RD = cv2.dilate(RG, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)

    if flagFiguras:
        plt.figure()
        plt.imshow(RD, cmap="gray", vmin=0, vmax=255)
        plt.title("Imagen Procesada")

    # Binarización
    Ib = funcion_correccion_iluminacion_segmentacion_global(RD, 65)

    if flagFiguras:
        plt.figure()
        plt.imshow(Ib*255, cmap="gray", vmin=0, vmax=255)
        plt.title("Imagen Binarizada")

    # Segmentación
    transpuesta = cv2.transpose(Ib)

    _, IEtiq, contornos, centroides = cv2.connectedComponentsWithStats(transpuesta)

    centroids = centroides[:, [1, 0]]
    contornos = contornos[:, [1, 0, 3, 2]]

    IEtiq = cv2.transpose(IEtiq)

    lineaCentral = IEtiq[IEtiq.shape[0]//2, :]

    valores = np.unique(lineaCentral)
    valores = valores[valores != 0]

    IbFiltEtiq = np.zeros_like(IEtiq)

    valoresFilt = []
    centroidesFilt = []
    contornosFilt = []

    for i in valores:
        componente = (IEtiq == i)
        area = np.sum(componente)
        
        if area >= 2715:
            IbFiltEtiq[componente] = i
            valoresFilt.append(i)
            centroidesFilt.append(centroids[i])
            contornosFilt.append(contornos[i])

    valoresFilt = valoresFilt[1:]

    IbSegEtiq = np.zeros_like(IbFiltEtiq)

    for i, j in zip(valoresFilt, range(len(valoresFilt))):
        componente = IbFiltEtiq==i
        IbSegEtiq[componente] = j+1

    numObjetos = len(valoresFilt)

    if flagFiguras:
        plt.figure()
        plt.imshow(IbSegEtiq*255, cmap="gray", vmin=0, vmax=255)
        plt.title("Imagen Segmentada")

    # Posprocesamiento
    IbSegEtiq = IbSegEtiq.astype(np.uint8)

    IbSegEtiqPos = cv2.dilate(IbSegEtiq, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)

    if flagFiguras:
        plt.figure()
        plt.imshow(IbSegEtiqPos*255, cmap="gray", vmin=0, vmax=255)
        plt.title("Imagen Posprocesada")

    return IbSegEtiqPos, numObjetos, centroidesFilt, contornosFilt


def funcion_segmentacion_local(I, W):
    
    I = I.astype(np.uint8)

    W = W if W % 2 == 1 else W + 1
    
    Ib = cv2.adaptiveThreshold(I, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, W, 10)

    return Ib


def funcion_segmentacion_local_validada_std(I, W):

    IFloat = I.astype(np.float32)

    V = np.ones((W, W), dtype=np.float32)

    kp = V / V.sum()

    mean = cv2.filter2D(IFloat, -1, kp, borderType=cv2.BORDER_REPLICATE)
    mean_sq = cv2.filter2D(IFloat*IFloat, -1, kp, borderType=cv2.BORDER_REPLICATE)

    std = np.sqrt(np.maximum(mean_sq - mean**2, 0))

    k = 0.5
    umbral = mean - k * std

    Ib = (IFloat < umbral).astype(np.uint8)
    
    return Ib


def funcion_correccion_iluminacion_segmentacion_global(I, W):

    IFloat = I.astype(np.float32)

    W = W if W % 2 == 1 else W + 1

    H = np.ones((W, W), np.float32) / (W*W)
    IF = cv2.filter2D(IFloat, -1, H, borderType=cv2.BORDER_REPLICATE)

    ID = IFloat - IF

    IC = cv2.normalize(ID, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    umbralOtsu, _ = cv2.threshold(IC, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    Ib = (IC < umbralOtsu).astype(np.uint8)
    
    return Ib


def funcion_correccion_iluminacion_segmentacion_local(I, W):

    IFloat = I.astype(np.float32)

    W = W if W % 2 == 1 else W + 1

    H = np.ones((W, W), np.float32) / (W*W)
    IF = cv2.filter2D(IFloat, -1, H, borderType=cv2.BORDER_REPLICATE)

    ID = IFloat - IF

    IC = cv2.normalize(ID, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    Ib = cv2.adaptiveThreshold(IC, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, W, 10)

    return Ib
