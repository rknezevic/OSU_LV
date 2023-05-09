import numpy as np
import matplotlib.pyplot as plt
#Skripta zadatak_3.py ucitava sliku  ˇ road.jpg’. Manipulacijom odgovarajuce´
#numpy matrice pokušajte:
# a) posvijetliti sliku,
# b) prikazati samo drugu cetvrtinu slike po širini, ˇ
# c) zarotirati sliku za 90 stupnjeva u smjeru kazaljke na satu,
# d) zrcaliti sliku.

ind = (data[:,0] == 1)
img = plt.imread('LV2/road.jpg')
img = img[:, :, 0].copy()
plt.imshow(img, cmap="gray")
plt.title("Cesta")
plt.show()

lighterImg = img+150
lighterImg[lighterImg < 150] = 255
plt.imshow(lighterImg, cmap="gray")
plt.title("Light cesta")
plt.show()

quarterImg = img[:, int(img.shape[1]/4):int(img.shape[1]/2)]
plt.imshow(quarterImg, cmap="gray")
plt.title("Sliceana cesta")
plt.show()

rotatedImg = np.rot90(img, 3)
plt.imshow(rotatedImg, cmap="gray")
plt.title("Cesta zarotirana za 90")
plt.show()

mirrorImg = np.flip(img, axis=1)
plt.imshow(mirrorImg, cmap="gray")
plt.title("Zrcaljena cesta")
plt.show()
