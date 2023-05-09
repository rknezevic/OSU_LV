import numpy as np
import matplotlib.pyplot as plt

# Napišite program koji ce kreirati sliku koja sadrži ´ cetiri kvadrata crne odnosno ˇ
# bijele boje (vidi primjer slike 2.4 ispod). Za kreiranje ove funkcije koristite numpy funkcije
# zeros i ones kako biste kreirali crna i bijela polja dimenzija 50x50 piksela. Kako biste ih složili
# u odgovarajuci oblik koristite numpy funkcije ´ hstack i vstack.

firstLight = np.ones((50, 50))*255
secondLight = firstLight.copy()
firstDark = np.zeros((50, 50))
secondDark = firstDark.copy()

firstRow = np.hstack((firstDark, firstLight))
secondRow = np.hstack((secondLight, secondDark))
fullSquare = np.vstack((firstRow, secondRow))
plt.imshow(fullSquare, cmap="gray")
plt.title("Crno bijeli kvadrat")
plt.show()
