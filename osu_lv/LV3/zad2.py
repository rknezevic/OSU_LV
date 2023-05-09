import pandas as pd
import matplotlib.pyplot as plt

#Zadatak 3.4.2 Napišite programski kod koji ce prikazati sljedece vizualizacije: ´





data = pd.read_csv('data_C02_emission.csv')

# a) Pomocu histograma prikažite emisiju C02 plinova. Komentirajte dobiveni prikaz. ´
def zad2a():
    data['CO2 Emissions (g/km)'].plot(kind='hist', bins=20) #numpy -- plt.hist(data[:, 7], bins=20)
    plt.xlabel('CO2 Emission (g/km)')
    plt.show()

# b) Pomocu dijagrama raspršenja prikažite odnos izmedu gradske potrošnje goriva i emisije ¯
#C02 plinova. Komentirajte dobiveni prikaz. Kako biste bolje razumjeli odnose izmedu¯
#velicina, obojite tockice na dijagramu raspršenja s obzirom na tip goriva. ˇ
def zad2b():
    data['Make']=pd.Categorical(data['Make'])
    data['Vehicle Class']=pd.Categorical(data['Vehicle Class'])
    data['Transmission']=pd.Categorical(data['Transmission'])
    data['Fuel Type']=pd.Categorical(data['Fuel Type'])
    data.plot.scatter(x='Fuel Consumption City (L/100km)', y='CO2 Emissions (g/km)', c='Fuel Type', cmap = 'viridis', s=20)
    #plt.scatter(x=data[:, 6], y= data[:, 11], c = data[:, 6], cmap= 'viridis', s=20)
    plt.show()

# c) Pomocu kutijastog dijagrama prikažite razdiobu izvangradske potrošnje s obzirom na tip
#   goriva. Primjecujete li grubu mjernu pogrešku u podacima? 
def zad2c():
    data.groupby('Fuel Type').boxplot(column='Fuel Consumption Hwy (L/100km)')
    plt.show()

# d) Pomocu stupcastog dijagrama prikažite broj vozila po tipu goriva. Koristite metodu ˇ
#groupby.
def zad2d():
    data.groupby('Fuel Type').agg({'Make':'count'}).rename(columns={'Make':'Car number'}).plot(kind="bar")
    plt.show()


#grouped_fuel_type = np.unique(data_np[:, 6], return_counts=True)
# prikaz broja vozila po Fuel Type-u
#plt.bar(grouped_fuel_type[0], grouped_fuel_type[1])
#plt.xlabel('Fuel Type')
#plt.ylabel('Car number')
#plt.show() 


# e) Pomocu stupcastog grafa prikažite na istoj slici prosjecnu C02 emisiju vozila s obzirom na ˇ
# broj cilindara.
def zad2e(): 
    data.groupby('Fuel Type').agg({'CO2 Emissions (g/km)':'mean'}).plot(kind='bar')
    plt.show()

# 
