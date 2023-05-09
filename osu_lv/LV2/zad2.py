import numpy as np
import matplotlib.pyplot as plt

# Datoteka data.csv sadrži mjerenja visine i mase provedena na muškarcima i
#ženama. Skripta zadatak_2.py ucitava dane podatke u obliku numpy polja  data pri cemu je u 
#prvom stupcu polja oznaka spola (1 muško, 0 žensko), drugi stupac polja je visina u cm, a treci
#stupac polja je masa u kg.

data = np.loadtxt('LV2\data.csv', delimiter=',', skiprows=1)

print(f"Broj ljudi: {data.shape[0]}")


visina = data[:, 1] #pohranjivanje drugog stupca 
masa = data[:, 2] #pohranjivanje treceg stupca

plt.scatter(visina, masa, c='b', s=1, marker=".")
plt.xlabel('Visina(cm)')
plt.ylabel('Masa(kg)')
plt.title('Odnos visine i mase')
plt.show()

visina50 = data[::50, 1]
masa50 = data[::50, 2]
plt.scatter(visina50, masa50, c='b', s=1, marker=".")
plt.xlabel('Visina(cm)')
plt.ylabel('Masa(kg)')
plt.title('Odnos visine i mase (svaka 50-a osoba)')
plt.show()

print(f'Najveca vrijednost visine: {visina.max()}cm')
print(f'Najmanja vrijednost visine: {visina.min()}cm')
print(f'Srednja vrijednost visine: {visina.mean()}cm')

visinaM = data[:, 1][data[:, 0] == 1]
visinaZ = data[:, 1][data[:, 0] == 0]

print(f'Najveca vrijednost visine muskaraca: {visinaM.max()}cm')
print(f'Najmanja vrijednost visine muskaraca: {visinaM.min()}cm')
print(f'Srednja vrijednost visine muskaraca: {visinaM.mean()}cm')

print(f'Najveca vrijednost visine zena: {visinaZ.max()}cm')
print(f'Najmanja vrijednost visine zena : {visinaZ.min()}cm')
print(f'Srednja vrijednost visine zena: {visinaZ.mean()}cm')


