import numpy as np
import matplotlib . pyplot as plt

x = np.array([2, 3, 3, 1, 2])
y = np.array([2, 2, 1, 1, 2])

plt . plot (x , y , 'b', linewidth =2 , marker ="*", markersize =8 , color = 'green')
plt . axis ([0 ,4 ,0 , 4])
plt . xlabel ('x os ')
plt . ylabel (' y os ')
plt . title ( ' primjer ')
plt . show ()