# import numpy as np

# emp = np.empty((0, 3))

# rand = np.arange(15).reshape(5, 3)
# print(rand)

# for i in range(rand.shape[0]):
#     emp = np.concatenate((emp, rand[i, :].reshape(1, 3)), axis=0)
#     # print(rand[i, :].shape)

# print(emp)


# print(np.argmax([1, 5, 6]))

string = "01-Jan"
respo = string.split('-')
print(respo)
num, name = respo
print(num)
print(name)










"""
# Import classes
from datetime import datetime as dt
from dateutil import relativedelta as rd

# Import main model class
from model import Modeler

# App class
class App():
    def __init__(self):
        self.modeler = Modeler()
        self.modeler()

    def run(self):
        date_string = input('\n[INPUT]: Enter a month and year to obtain a prediction for, in the form MM/YYYY: ')
        # date_string = "05/2022"
        date = dt.strptime(date_string, "%m/%Y")
        timedelta = rd.relativedelta(date, dt(2021, 1, 1))
        month_delta = timedelta.months + 12*timedelta.years + 1
        self.modeler.predict(month_delta)
    
app = App()
app.run()
"""