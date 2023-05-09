import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering

""" Zadatak 7.5.2. Skripta zadatak_1.py sadrži funkciju generate_data koja služi za generiranje
umjetnih podatkovnih primjera kako bi se demonstriralo grupiranje. Funkcija prima cijeli broj
koji definira željeni broju uzoraka u skupu i cijeli broj od 1 do 5 koji definira na koji nacin ˇ ce´
se generirati podaci, a vraca generirani skup podataka u obliku numpy polja pri ´ cemu su prvi i ˇ
drugi stupac vrijednosti prve odnosno druge ulazne velicine za svaki podatak. Skripta generira ˇ
500 podatkovnih primjera i prikazuje ih u obliku dijagrama raspršenja.

1. Pokrenite skriptu. Prepoznajete li koliko ima grupa u generiranim podacima? Mijenjajte
nacin generiranja podataka. ˇ
2. Primijenite metodu K srednjih vrijednosti te ponovo prikažite primjere, ali svaki primjer
obojite ovisno o njegovoj pripadnosti pojedinoj grupi. Nekoliko puta pokrenite programski
kod. Mijenjate broj K. Što primjecujete? 
3. Mijenjajte nacin definiranja umjetnih primjera te promatrajte rezultate grupiranja podataka
(koristite optimalni broj grupa). Kako komentirate dobivene rezultate? """

def generate_data(n_samples, flagc):
    # 3 grupe
    if flagc == 1:
        random_state = 365
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
    
    # 3 grupe
    elif flagc == 2:
        random_state = 148
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    # 4 grupe 
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                        centers = 4,
                        cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
                        random_state=random_state)
    # 2 grupe
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    
    # 2 grupe  
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X

# generiranje podatkovnih primjera
X = generate_data(500, 1)
# prikazi primjere u obliku dijagrama rasprsenja
plt.figure()
plt.scatter(X[:,0],X[:,1])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri')
plt.show()
#broj grupa u podacima se lako moze prepoznati uz pomoc vizualizacije (dijagram rasprsenja) za svaki od nacina generiranja podataka (1-5)

kmeans = KMeans(n_clusters=3, init ='random')
kmeans.fit(X)
labels = kmeans.predict(X)
plt.figure()
plt.scatter(X[:,0],X[:,1], c=labels, cmap='viridis')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Grupirani podatkovni primjeri')
plt.show()
#neispravnim postavljanjem broja k dobija se previše ili premalo grupa
#kmeans kod nekih primjera ne grupira kako treba jer pretpostavlja da su grupe sferične, podjednake velicine i slicne gustoce,
#ne radi dobro s grupama nepravilnih oblika (jer radi na principu udaljenosti) (uz primjenu optimalnih vrijednosti k)
#kada flagc=1, radi dobro jer su grupe sfericne

