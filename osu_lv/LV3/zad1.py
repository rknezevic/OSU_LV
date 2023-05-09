import pandas as pd
import numpy as np

data = pd.read_csv(r"C:\Users\rober\OneDrive\Radna površina\osu_lv-main\LV3\emisije.csv")
data_np = np.array(data)
#data_np = np.genfromtxt(r"C:\Users\rober\OneDrive\Radna površina\osu_lv-main\LV3\emisije.csv", delimiter = ',', skip_header = 1)
 
#Skripta zadatak_1.py ucitava podatkovni skup iz ˇ data_C02_emission.csv.
#Dodajte programski kod u skriptu pomocu kojeg možete odgovoriti na sljedeca pitanja: ´
"""#a) Koliko mjerenja sadrži DataFrame? Kojeg je tipa svaka velicina? Postoje li izostale ili 
#duplicirane vrijednosti? Obrišite ih ako postoje. Kategoricke veli ˇ cine konvertirajte u tip 
#category.
print("Valjda poceinjemo")

print(f'Broj mjerenja: {len(data)}')
print(f'Broj mjerenja: {data_np.shape[0]}')
print("askjlfhlasgfls")
print('Tipovi velicina:')
print(data.dtypes)
print(f'Broj dupliciranih vrijednosti:{data.duplicated().sum()}')
print('Broj izostalih vrijednosti po stupcima:')
print(data.isnull().sum())
data['Make']=pd.Categorical(data['Make'])
data['Vehicle Class']=pd.Categorical(data['Vehicle Class'])
data['Transmission']=pd.Categorical(data['Transmission'])
data['Fuel Type']=pd.Categorical(data['Fuel Type'])
print(data.dtypes)"""


#b) Koja tri automobila ima najvecu odnosno najmanju gradsku potrošnju? Ispišite u terminal: 
#ime proizvodaca, model vozila i kolika je gradska potrošnja. ˇ
data_new = pd.DataFrame(
    data, columns=["Make", "Model", "Fuel Consumption City (L/100km)"]
).sort_values("Fuel Consumption City (L/100km)", ascending=False)
print(data_new.head(3))
print(data_new.tail(3))
print("mojeee")
data_b = np.array(data[["Make", "Model", "Fuel Consumption City (L/100km)"]])
data_np_sorted = data_b[data_b[:, 2].argsort()[::-1]]

print(data_np_sorted[:3])
print(data_np_sorted[-3:])

"""import numpy as np

# Pretvaranje podataka u numpy array
data_np = np.array(data[["Make", "Model", "Fuel Consumption City (L/100km)"]])

# Sortiranje po stupcu koji sadrži potrošnju goriva po silaznom redoslijedu ( [::-1] )
data_np_sorted = data_np[data_np[:, 2].argsort()[::-1]]

# Ispis tri najveće i tri najmanje potrošnje goriva
print(data_np_sorted[:3])
print(data_np_sorted[-3:])

#Dizelski motor s 4 cilindra s najvecom potrosnjom?

print(f"Dizelski motor s najvecom potrosnjom koji ima 4 cilindra je: {data_np_sorted[(data_np_sorted[:, 4] == '4') & (data_np_sorted[:, 6] == 'D')][-1]}")
"""
#c) Koliko vozila ima velicinu motora izmedu 2.5 i 3.5 L? Kolika je prosjecna C02 emisija ˇ
#plinova za ova vozila?

engine_data = data[(data["Engine Size (L)"] >= 2.5) & (data["Engine Size (L)"] <= 3.5)]
print(len(engine_data))
print("CO2: ", engine_data.loc[:, "CO2 Emissions (g/km)"].mean())

print("MASJFIOASJ")
engine_extracted = data_np[(data_np[:, 3] >= 2.5) & (data_np[:, 3] <= 3.5)]
print(len(engine_extracted))
print("CO2 ", engine_extracted[:, 11].mean())

#d) Koliko mjerenja se odnosi na vozila proizvodaca Audi? Kolika je prosjecna emisija C02 ˇ
#plinova automobila proizvodaca Audi koji imaju 4 cilindara? ˇ


print(f"Broj Audija:{len(data[data['Make']=='Audi'])}")
num_audi = np.where(data_np[:,0].astype(str) == 'Audi')[0].size
print(f"MOJ Broj Audija: {num_audi}")
print(f"Prosjecna emisija CO2 audija s 4 cilindra: {(data[(data['Make']=='Audi') & (data['Cylinders']==4)])['CO2 Emissions (g/km)'].mean()} (g/km)")

#e) Koliko je vozila s 4,6,8. . . cilindara? Kolika je prosjecna emisija C02 plinova s obzirom na 
#broj cilindara?
print(data.groupby('Cylinders').agg({'Make':'count', 'CO2 Emissions (g/km)':'mean'}).rename(columns={'Make': 'Car count'}))
"""
parni_cilindri=data[(data['Cylinders'])%2==0]
print(len(parni_cilindri))
print(parni_cilindri.groupby(by='Cylinders')['CO2 Emissions (g/km)'].mean())
"""
#f) Kolika je prosjecna gradska potrošnja u slucaju vozila koja koriste dizel, a kolika za vozila 
#koja koriste regularni benzin? Koliko iznose medijalne vrijednosti?
print('Mean i median za dizel:')
print(data[data['Fuel Type']=='D']['Fuel Consumption City (L/100km)'].agg(['mean','median']))
print('Mean i median za benzin:')
print(data[data['Fuel Type']=='X']['Fuel Consumption City (L/100km)'].agg(['mean','median']))
# gas_cars = data[data[:, 9] == 'X')
    # print(gas_cars.mean())
    # city_consumption = gas_cars[:,8].astype(float)

#g) Koje vozilo s 4 cilindra koje koristi dizelski motor ima najvecu gradsku potrošnju goriva? ´
print(data[(data['Cylinders']==4) & (data['Fuel Type']=='D')].sort_values(by='Fuel Consumption City (L/100km)').tail(1))

#h) Koliko ima vozila ima rucni tip mjenjaca (bez obzira na broj brzina)? ˇ
print(f"Broj vozila s manualnim mjenjačem: {len(data[data['Transmission'].str.startswith('M')])}")
print(f"BROJ MANUALNIH VOZILA: { np.sum(np.char.startswith(data_np[:, 4].astype(str), 'M'))}")
# first_letter = np.chararray.capitalize(arr.astype(str))[:, 0]
# num_m = len(arr[np.char.startswith(arr, 'M')])
#i) Izracunajte korelaciju izmedu numerickih velicina. Komentirajte dobiveni rezultat.
#print(data.corr(numeric_only=True))







