import cv2
import matplotlib.pyplot as plt # pip install matplotlib

def representa_resultados_segmentacion_reconocimiento(Ic_placa, prediccion, centroides, contornos):

    Irgb = cv2.cvtColor(Ic_placa, cv2.COLOR_BGR2RGB)

    plt.figure()
    plt.imshow(Irgb)
    plt.title(prediccion)

    for i in range(1, len(centroides)):
        cx, cy = centroides[i]
        plt.plot(cx, cy, marker='*', color='r')
        
        x, y, w, h = contornos[i]
        plt.plot([x, x+w], [y, y], 'g')
        plt.plot([x, x+w], [y+h, y+h], 'g')
        plt.plot([x, x], [y, y+h], 'g')
        plt.plot([x+w, x+w], [y, y+h], 'g')
