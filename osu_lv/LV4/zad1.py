import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

#Skripta zadatak_1.py ucitava podatkovni skup iz ˇ data_C02_emission.csv.
#Potrebno je izgraditi i vrednovati model koji procjenjuje emisiju C02 plinova na temelju ostalih numerickih ulaznih veli ˇ cina. 
#Detalje oko ovog podatkovnog skupa mogu se pronaci u 3. ´
#laboratorijskoj vježbi.



#




#a) Odaberite željene numericke velicine specificiranjem liste s nazivima stupaca. Podijelite
#podatke na skup za ucenje i skup za testiranje u omjeru 80%-20%.
data = pd.read_csv('data_C02_emission.csv')
y=data['CO2 Emissions (g/km)'].copy()
X=data[['Engine Size (L)','Cylinders','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)']]
X_train , X_test , y_train , y_test = train_test_split (X, y, test_size = 0.2, random_state =1) #vraca isti tip kao i predani, podjela na 20% i 80%
#inace je lakse raditi s numpy arrayevima nego s dataframeom jer stalno sve vraca numpy

#b) Pomocu matplotlib biblioteke i dijagrama raspršenja prikažite ovisnost emisije C02 plinova ´
#o jednoj numerickoj velicini. Pri tome podatke koji pripadaju skupu za ucenje oznacite ˇ
#plavom bojom, a podatke koji pripadaju skupu za testiranje oznacite crvenom bojom.
for col in X_train.columns:
    plt.scatter(X_train[col],y_train, c='b', label='Train', s=5)
    plt.scatter(X_test[col],y_test, c='r', label='Test', s=5)
    plt.xlabel(col)
    plt.ylabel('CO2 Emissions (g/km)')
    plt.legend()
    plt.show()

# c) Izvršite standardizaciju ulaznih velicina skupa za ucenje. Prikažite histogram vrijednosti ˇ
#jedne ulazne velicine prije i nakon skaliranja. Na temelju dobivenih parametara skaliranja ˇ
#transformirajte ulazne velicine skupa podataka za testiranje. 
sc = MinMaxScaler () #transform vraca numpy array, zato se mora nazad u dataframe (da se sve radilo s numpy, ne bi bilo potrebe za time)
X_train_n = pd.DataFrame(sc.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
for col in X_train.columns:
    fig,axs = plt.subplots(2,figsize=(8, 8))
    axs[0].hist(X_train[col])
    axs[0].set_title('Before scaler')
    axs[1].hist(X_train_n[col])
    axs[1].set_xlabel(col)
    axs[1].set_title('After scaler')
    plt.show()
X_test_n = pd.DataFrame(sc.transform(X_test), columns=X_test.columns, index=X_test.index)

#d) Izgradite linearni regresijski modeli. Ispišite u terminal dobivene parametre modela i
#povežite ih s izrazom 4.6.
linearModel=lm.LinearRegression()
linearModel.fit(X_train_n, y_train)
print(f'Parametri modela: {linearModel.coef_}')
print(f'Intercept parametar: {linearModel.intercept_}')

#e) Izvršite procjenu izlazne velicine na temelju ulaznih veli ˇ cina skupa za testiranje. Prikažite ˇ
#pomocu dijagrama raspršenja odnos izme ´ du stvarnih vrijednosti izlazne veli ¯ cine i procjene ˇ
#dobivene modelom.
y_prediction = linearModel.predict(X_test_n) #vraca numpy array
plt.scatter(X_test_n['Fuel Consumption City (L/100km)'],y_test, c='b', label='Real values', s=5)
plt.scatter(X_test_n['Fuel Consumption City (L/100km)'],y_prediction, c='r', label='Prediction', s=5)
plt.xlabel('Fuel Consumption City (L/100km)')
plt.ylabel('CO2 Emissions (g/km)')
plt.legend()
plt.show()

#f) Izvršite vrednovanje modela na nacin da izra ˇ cunate vrijednosti regresijskih metrika na ˇ
#skupu podataka za testiranje.
#g) Što se dogada s vrijednostima evaluacijskih metrika na testnom skupu kada mijenjate broj ¯
#ulaznih velicina?
print(f'Mean squared error: {mean_squared_error(y_test, y_prediction)}')
print(f'Mean absolute error: {mean_absolute_error(y_test, y_prediction)}')
print(f'Mean absolute percentage error: {mean_absolute_percentage_error(y_test, y_prediction)}%')
print(f'R2 score: {r2_score(y_test, y_prediction)}')












