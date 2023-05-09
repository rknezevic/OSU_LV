from tensorflow import keras
from tensorflow.keras import layers         #bug, radi
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
print(f'Broj primjera za ucenje: {len(X_train)}')
print(f'Broj primjera za testiranje: {len(X_test)}')
#moze se i skalirat sa X_train_s = X_train.astype("float32")/255
#ulazni podaci imaju oblik (broj primjera,28,28)(svaka slika je 28x28 piksela), svaki piksel predstavljen brojem 0-255
#izlazna velicina kodirana na nacin da su znamenke predstavljene brojevima 0-9
#svaka slika(primjer)-2d matrica, 28x28

#prikaz random slike i oznake
X_train_reshaped = np.reshape(X_train,(len(X_train),X_train.shape[1]*X_train.shape[2])) #umjesto len(X_train) moze i X_train.shape[0]
X_test_reshaped = np.reshape(X_test,(len(X_test),X_test.shape[1]*X_test.shape[2]))      #umjesto len(X_test) moze i X_test.shape[0]
plt.imshow(X_train[7])   #slike se prikazuju normalnim 2d poljem
plt.title(f'Slika broja {y_train[7]}')
plt.show()

#izrada mreze i ispis detalja
model = keras.Sequential()
model.add(layers.Input(shape=(784,)))
model.add(layers.Dense(units=100, activation="relu"))
model.add(layers.Dense(units=50, activation="relu"))
model.add(layers.Dense(units=10, activation="softmax"))
model.summary()
#oneHotEncoding izlaza, da sve bude prema skici u predlosku, za ovo u kerasu postoji isto funkcija y_train = keras.utils.to_categorical(y_train, num_classes=10)
oh=OneHotEncoder()
y_train_encoded = oh.fit_transform(np.reshape(y_train,(-1,1))).toarray() #OneHotEncoder trazi 2d array, pa treba reshape (-1,1), tj (n,1),
y_test_encoded = oh.transform(np.reshape(y_test,(-1,1))).toarray() #-1 znaci sam skontaj koliko, mora toarray() obavezno kod onehotencodera

#podesavanje parametara treninga
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy",])
history = model.fit(X_train_reshaped , y_train_encoded, batch_size=32, epochs=20, validation_split=0.1)

#evaluacija i ispis 
score = model.evaluate(X_test_reshaped, y_test_encoded, verbose=0)
for i in range(len(model.metrics_names)):
    print(f'{model.metrics_names[i]} = {score[i]}')

#predict i matrica zabune
y_predictions = model.predict(X_test_reshaped)  #vraca za svaki primjer vektor vjerojatnosti pripadanja svakoj od 10 klasa (softmax) (10 000,10)
y_predictions = np.argmax(y_predictions, axis=1)  #vraća polje indeksa najvecih elemenata u svakom pojedinom retku (1d polju) (0-9) (10 000,) - 1d polje
cm = confusion_matrix(y_test, y_predictions)    #zbog prethodnog koraka, usporedba s y_test, a ne encoded
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#spremanje modela
model.save('Model/')



