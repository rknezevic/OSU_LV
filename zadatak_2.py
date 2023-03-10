import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt("Data.csv", skiprows=1, delimiter=',')
print (len(data))

plt.scatter(data[:, 1], data [:, 2])
plt.xlabel("Visina (cm): ")
plt.ylabel("Masa (kg): ")
plt.title("Odnos visine i mase: ")
plt.show()

plt.scatter(data[::50, 1], data[::50, 2])
plt.xlabel("Visina (cm): ")
plt.ylabel("Masa (kg): ")
plt.title("Odnos visine i mase svake pedesete osobe: ")
plt.show()

print(np.min(data[:, 1]))
print(np.max(data[:, 1]))
print(np.mean(data[:, 1]))

ind = (data[:, 0] == 1)
print(np.max(data[ind, 1]))
ind = (data[:, 0] == 0)
print(np.max(data[ind, 1]))
