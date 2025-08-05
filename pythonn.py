# myfamily = {
#   "child1" : {
#     "name" : "Emil",
#     "year" : 2004
#   },
#   "child2" : {
#     "name" : "Tobias",
#     "year" : 2007
#   },
#   "child3" : {
#     "name" : "Linus",
#     "year" : 2011
#   }
# }

# for x, x1 in myfamily.items():
#   print(x)
	
#   for y, y1 in x1.items():
#     print(y, ":", y1)

# class Cars():
#     def __init__(self, brand, cost):
#         self.brand = brand
#         self.cost = cost

#     def __str__(self):
#         return f'Это {self.brand} и он стоит {self.cost} доларов'

# mercedes = Cars('mercedes', '1 500 000')
# bmw = Cars('BMW', '500 000')

# print(bmw)

# bmw.cost = '600 000'
# print(bmw)


# del bmw.brand
# print(mercedes)
# print(bmw)


# class Person:
#   def __init__(self, fname, lname):
#     self.firstname = fname
#     self.lastname = lname

#   def __str__(self):
#     return f"{self.firstname}({self.lastname})"

# class Student(Person):
#   pass

# x = Student("Mike", "Olsen")
# print(x)




# class Person:
#   def __init__(self, fname, lname):
#     self.firstname = fname
#     self.lastname = lname

#   def __str__(self):
#     return f"{self.firstname} {self.lastname}"
    
# y = Person('Anton', 'Shklyar')
# print(y)



# class Student(Person):
#   def __init__(self, fname, lname, year):
#     super().__init__(fname, lname)
#     self.graduationyear = year

#   def __str__(self):
#     return f"Welcome {self.firstname} {self.lastname} to the class of {self.graduationyear}"

# x = Student("Mike", "Olsen", 2024)
# print(x)



# from datetime import *

# print(datetime.now())

# x = '4343'
# while type(x) is not int:
# 	try:
# 	  x = int(input("Enter a number:"))
# 	except:
# 		print('Wrong input, please try again.')




# x = 20
# print(f'{20:.3f}')


# speed = [99,86,87,88,111,86,103,87,94,78,77,85,86, 2,2,2]

# x = [[speed.count(i), i] for i in speed if speed.count(i) > 1]

# f = []
# for i in range(len(x)):
#     print(x[i])
#     if ((x[i])[0]) == ((max(x))[0]):
#         f.append(x[i])
        
# print()		
# print(f)

# mnoj = {tuple(i) for i in f}
# print(mnoj)

# print(min(mnoj)[1])



# from numpy import *
# from matplotlib import pyplot as plt

# x = random.uniform(0.0, 5.0, 100_000)

# plt.hist(x, 100)
# plt.show() 


# from math import *
# from scipy.stats import *
# from scipy.optimize import root

# from numpy import *
# from numpy.random import *

# from matplotlib import *
# from matplotlib.pyplot import *

# def f(x):
#     return x + cos(x)

# func = root(f, 0)
# print(func)


# x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
# y = [99,86,87,88,111,86,103,87,94,78,77,85,86]


# scatter(x,y)
# show()





# speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]

# speed = randint(1,10, size = (1_000_000))

# print(mean(speed))#5
# print(median(speed))#+-5
# print(mode(speed))#5

# print(std(speed))#1
# print(var(speed))#1

# print(percentile(speed, 100))#+-6


# hist(speed,100)

# show()




# subplot(2,2,1)
# x = array([5,7,8,7,2,17,2,9,4,11,12,9,6])
# y = array([99,86,87,88,111,86,103,87,94,78,77,85,86])
# scatter(x, y)

# subplot(2,2,2)
# x = array([2,2,8,1,15,8,12,9,7,3,11,4,7,14,12])
# y = array([100,105,84,105,90,99,90,95,94,100,79,112,91,80,85])
# scatter(x, y, c = 'r')

# subplot(2,2,3)
# x = array(["A", "B", "C", "D"])
# y = array([3, 8, 1, 10])
# bar(x,y)


# subplot(2,2,4)
# x = array([2,2,8,1,15,8,12,9,7,3,11,4,7,14,12])
# y = array([100,105,84,105,90,99,90,95,94,100,79,112,91,80,85])
# plot(x, y, 'o:g')


# show()


# class Movies():
#     def __init__(self, name, rating, year):
#         self.name = name
#         self.rating = rating
#         self. year = year

#     def __str__(self):
#         return f'Фильм {self.year} года, под названием {self.name}, пользователи нашего кинотеатра в среднем оценили  на {self.rating}'
    
# inception = Movies('Inception', '9.1', '2013')
# print(inception)




# class Person:
#   def __init__(self, fname, lname):
#     self.firstname = fname
#     self.lastname = lname

#   def __str__(self):
#     return f"{self.firstname} {self.lastname}"
    
# y = Person('Anton', 'Shklyar')
# print(y)






# from csv import *

##чтение
# with open('x.txt','r', encoding='utf-8', ) as f:
#     o_reader = reader(f)
#     for i in o_reader:
#         print(i)


##запись
# header = ['model', 'color', 'year']
# cars = [
#     ['Tesla Model S', 'Red', 2021],
#     ['Ford Mustang', 'Black', 2019]
# ]

# with open('x.txt', 'w', encoding='utf-8', newline='') as f:
#     o_writer = writer(f)
#     o_writer.writerow(header)
#     o_writer.writerows(cars)
    

# import json


# library = {
#     "name": "Alex",
#     "is_admin": True,
#     "courses": ["History", "Math"]
# }

# with open('x.json', 'w', encoding='utf-8') as f:
#     json.dump(library, f, indent=4, ensure_ascii=False)


















# import csv

# x = []

# with open('movies.csv', 'r', encoding='utf-8') as file:
#     o_reader = csv.reader(file)
#     for i in o_reader:
#         x.append(i)



# x.sort(key=lambda x: (-float(x[1]), float(x[2])))

# movies_list_top_5 = []
# y = 0
# for i in x:
#     movies_list_top_5.append(i)
#     y+=1
#     if y == 5:
#         break



# import json

# with open('movies_top_5.json', 'w', encoding='utf-8', newline='') as file:
#     json.dump(movies_list_top_5, file, indent=4)











# Создадим DataFrame из словаря
# movie_data = {
#     'title': ['The Shawshank Redemption', 'The Godfather', 'Pulp Fiction'],
#     'rating': [9.3, 9.2, 8.9],
#     'year': [1994, 1972, 1994]
# }

# df = pd.DataFrame(movie_data)
# import pandas as pd

# data = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35]}
# df = pd.DataFrame(data)

# print(df)


# data = [
#     ['Name', 'Age'],
#         ['Alice', 25],
#         ['Bob', 30],
#         ['Charlie', 35]
#         ]
# import csv
# with open('new_df.csv', 'w', encoding='utf-8', newline='') as file:
#     o_writer = csv.writer(file)
#     o_writer = o_writer.writerows(data)


# header = data[0]
# ost = data[1:]

# df = pd.read_csv('movies.csv')

# # Создадим колонку 'Age in 10 years'
# print(df.isnull().sum())
# # print(df)
# # print()
# # print()
# # print(df.head())
# # print()
# # print()
# # print(df.tail())
# # print()
# # print()
# # print(df.shape)
# # print()
# # print()
# # print(df.info())
# # print()
# # print()
# # print(df.describe())





# import pandas as pd

# df = pd.read_csv('Titanic.csv')
# print(df)
# ll = len(df.columns)


# print()
# for i in range(0,ll):
#     print( df.columns[i],    (df.iloc[:,i]).unique()       )
#     print()


# import numpy as np

# vector_a = np.array([2,3])
# vector_b = np.array([4,5])

# scal_proiz = vector_a @ vector_b

# print(scal_proiz)


# matrix_1 = np.array([[2,4],
#                      [3,9]])

# matrix_2 = np.array([[4,2],
#                      [1,2]])

# matrix_result = matrix_1 @ matrix_2

# print(matrix_result)

# import numpy as np
# from scipy import stats

# ds_ages = np.array([170, 180, 190])

# print(np.mean(ds_ages))
# print(np.median(ds_ages))
# print(stats.mode(ds_ages))
# print(np.std(ds_ages))
# print(np.var(ds_ages))


# import numpy as np
# import matplotlib.pyplot as plt

# ds_ages = np.array([175, 182, 168, 190, 177, 185, 179, 170, 183, 195, 172, 176, 188, 171, 180, 174, 165, 192, 178, 181])


# plt.hist(ds_ages, 10)

# plt.show()



# import numpy as np
# from numpy import random
# k = 1_000
# game = np.random.randint(0,2, size = (k))

# heads = np.sum(game == 1)
# tails = np.sum(game == 0)


# print(f'\nКоличество выпавших орлов:{heads}, {(heads/k*100):.3f}%\n\nКоличество выпавших решек:{tails}, {(tails/k*100):.3f}%')


# numpy, scipy, sympy, pandas, matplotlib, turtle

import sympy
import numpy as np
from matplotlib import pyplot as plt

x = sympy.symbols('x')

f = 3*x**2 + 2*x + 5

diff = sympy.diff(f, x)
# diff = 6*x+2

m_linear = sympy.lambdify(x, f, 'numpy')
m_diff = sympy.lambdify(x, diff, 'numpy')

x_values = np.linspace(-10,10, 100)

y_values_linear = m_linear(x_values) 
y_values_diff = m_diff(x_values)

plt.figure(figsize=(10, 4))

plt.subplot(1,2,1)
plt.plot(x_values, y_values_linear)
plt.title('Функция $f(x)=3x^2+2x+5$')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(x_values, y_values_diff, 'r')
plt.title('Производная $f\'(x)=6x+2$')
plt.xlabel('x')
plt.ylabel('f\'(x)')
plt.grid(True)


plt.tight_layout()
plt.show()

