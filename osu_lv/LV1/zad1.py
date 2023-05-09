#funkcija
def total_euro(workHours,hourPrice):
    return workHours*hourPrice


workHours = float(input("Unesite broj radnih sati: "))
hourPrice = float(input("Unesite satnicu: "))
print(f"{total_euro(workHours,hourPrice)} eura")



 