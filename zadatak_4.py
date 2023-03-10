import numpy as np
import matplotlib.pyplot as plt

mat = np.array([[0, 1], [1, 0]])

# crtanje šahovske ploče
plt.imshow(mat, cmap='binary')

# prikazivanje šahovske ploče
plt.show()
