"""
    En este modulo se incluyen las funciones principales
    del proyecto de reconocimiento de colores
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


PATH = './assets/imagen, prueba.jpg'  # dirección de imagen

# capturamos el frame o la imagen que queremos
try:
    # (path, mode)
    img = cv2.imread(PATH)  # Leer imagen en BGR(defecto)
    # transformar a RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

except Exception as e:
    print(e)

# mostrar imagen
#plt.imshow(img)  # mostrar img
#plt.axis('off')  # quitar ejes
#plt.show()

# encapsulamos la imagen en una lista de pixeles

pixels = img.reshape((-1, 3))  # reshape trnasforma un aimagen a una lista de pixeles
                               # -1 calcula automaticamente el numero de filas
                               # 3 inidica que cada fila tiene 3 elementos, son los 3 canales R G B

# se aplica machine learning para encontrar los 8 colores principales
kmeans = KMeans(n_clusters=8, n_init=10)
kmeans.fit(pixels)

# Extracción de los colores principales
colors = kmeans.cluster_centers_.astype(int)

paleta = np.zeros((50, 300, 3), dtype='uint8')
step = 300 // len(colors)
for i, color in enumerate(colors):
    paleta[:, i*step:(i+1)*step] = color

plt.imshow(paleta)
plt.axis('off')
plt.show()
print('success')
