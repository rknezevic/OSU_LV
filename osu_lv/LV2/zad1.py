import numpy as np
import matplotlib.pyplot as plt

#  Pomocu funkcija  numpy.array i matplotlib.pyplot pokušajte nacrtati sliku
# 2.3 u okviru skripte zadatak_1.py. Igrajte se sa slikom, promijenite boju linija, debljinu linije i
# sl. **geometrijski element neki**

x = np.array([1, 2, 3, 3, 1])
y = np.array([1, 2, 2, 1, 1])

plt.plot(x, y, 'r', linewidth=3, marker="o", markersize=10)
plt.axis([0.0, 4.0, 0.0, 4.0])
plt.xlabel("x-os")
plt.ylabel("y-os")
plt.title("Primjer")
plt.show()
