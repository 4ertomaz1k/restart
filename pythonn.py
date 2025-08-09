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

# import sympy
# import numpy as np
# from matplotlib import pyplot as plt

# x = sympy.symbols('x')

# f = 3*x**2 + 2*x + 5

# diff = sympy.diff(f, x)
# # diff = 6*x+2

# m_linear = sympy.lambdify(x, f, 'numpy')
# m_diff = sympy.lambdify(x, diff, 'numpy')

# x_values = np.linspace(-10,10, 100)

# y_values_linear = m_linear(x_values) 
# y_values_diff = m_diff(x_values)

# plt.figure(figsize=(10, 4))

# plt.subplot(1,2,1)
# plt.plot(x_values, y_values_linear)
# plt.title('Функция $f(x)=3x^2+2x+5$')
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.grid(True)

# plt.subplot(1,2,2)
# plt.plot(x_values, y_values_diff, 'r')
# plt.title('Производная $f\'(x)=6x+2$')
# plt.xlabel('x')
# plt.ylabel('f\'(x)')
# plt.grid(True)


# plt.tight_layout()
# plt.show()

# import pandas as pd

# data = {'Имя': ['Аня', 'Борис', 'Вика', 'Гриша', 'Даша'],
#         'Возраст': [25, 32, 28, 45, 22],
#         'Город': ['Москва', 'Санкт-Петербург', 'Москва', 'Казань', 'Москва'],
#         'Зарплата': [70000, 95000, 80000, 120000, 65000]}

# df = pd.DataFrame(data)
# print(df)
# Vika = df[(df['Зарплата'] > 75_000) & (df['Город'] == 'Москва')]
# avg_cash = df.groupby('Город')['Зарплата'].mean()
# avg_age = df.groupby(['Город', 'Возраст'])['Возраст'].mean()




#      Имя  Возраст            Город  Зарплата
# 0    Аня       25           Москва     70000
# 1  Борис       32  Санкт-Петербург     95000
# 2   Вика       28           Москва     80000
# 3  Гриша       45           Казань    120000
# 4   Даша       22           Москва     65000

# quest_1_1 = df[(df['Зарплата'] > 80_000) | (df['Возраст'] < 25)]
# quest_1_2 = df[(df['Город'] != 'Москва')]

# quest_2_1 = df.groupby('Город')['Имя'].count()
# quest_2_2 = df.groupby('Город')['Зарплата'].agg(['min', 'max'])


# print(quest_2_2)


# users_df = pd.DataFrame({
#     'User_id': [1, 2, 3, 4],
#     'Имя': ['Аня', 'Борис', 'Вика', 'Гриша']
# })

# orders_df = pd.DataFrame({
#     'Order_id': [101, 102, 103, 104],
#     'User_id': [1, 3, 2, 5],
#     'Товар': ['Ноутбук', 'Телефон', 'Клавиатура', 'Мышь']
# })

# print("DataFrame с пользователями:")
# print(users_df)
# print("\nDataFrame с заказами:")
# print(orders_df)

# merged_df = pd.merge(users_df, orders_df, on='User_id')
# print("\nОбъединенный DataFrame:")
# print(merged_df)


# Создаем DataFrame с пропусками
# data = {'A': [1, 2, None, 4], 'B': [5, None, 7, 8]}
# df_nan = pd.DataFrame(data)

# print("DataFrame с пропусками:")
# print(df_nan)
# print("\nКоличество пропусков в каждом столбце:")
# print(df_nan.isnull().sum())

# import pandas as pd

# # DataFrame с данными о городах
# cities_df = pd.DataFrame({
#     'city_id': [1, 2, 3, 4, 5],
#     'название_города': ['Москва', 'Санкт-Петербург', 'Казань', 'Новосибирск', 'Екатеринбург']
# })

# # DataFrame с данными о населении (с пропуском для Новосибирска)
# population_df = pd.DataFrame({
#     'city_id': [1, 2, 3, 5],
#     'население': [13_010_000, 5_600_000, 1_300_000, 1_500_000]
# })

# print("DataFrame с городами:")
# print(cities_df)
# print("\nDataFrame с населением:")
# print(population_df)
# print()

# merged_df = pd.merge(cities_df, population_df, on='city_id', how = 'outer')
# print(merged_df)



# import pandas as pd
# import numpy as np

# # DataFrame с пропусками
# data_with_nan = pd.DataFrame({
#     'столбец_A': [10, 20, np.nan, 40, 50],
#     'столбец_B': [100, np.nan, 300, 400, 500],
#     'столбец_C': [1, 2, 3, 4, np.nan]
# })

# print("DataFrame с пропусками:")
# print(data_with_nan)
# print("\nКоличество пропусков в каждом столбце:")
# print(data_with_nan.isnull().sum())

# data_with_nan['столбец_A'] = data_with_nan['столбец_A'].fillna(data_with_nan['столбец_A'].mean())

# print(data_with_nan)

# data_with_nan = data_with_nan.dropna()

# print()
# print(data_with_nan)

# import pandas as pd


# # Создаём DataFrame
# data = {
#     'title': ['The Shawshank Redemption', 'The Godfather', 'The Dark Knight', 'Pulp Fiction', 
#               'Forrest Gump', 'Inception', 'The Matrix', 'Fight Club', 'Goodfellas', 'The Lord of the Rings'],
#     'genre': ['Drama', 'Crime', 'Action', 'Crime', 'Drama', 
#               'Sci-Fi', 'Sci-Fi', 'Drama', 'Crime', 'Fantasy'],
#     'rating': [9.3, 9.2, 9.0, 8.9, 8.8, 8.8, 8.7, 8.8, 8.7, 8.9]
# }

# movies_df = pd.DataFrame(data)

# print("Созданный DataFrame:")
# print(movies_df)


# avg_of_genre = movies_df.groupby('genre')['rating'].mean()
# print()
# print(avg_of_genre)


# import pandas as pd

# movies_df = pd.DataFrame({
#     'movie_id': [1, 2, 3, 4, 5],
#     'title': ['Inception', 'The Matrix', 'The Dark Knight', 'Pulp Fiction', 'Forrest Gump']
# })

# print("Датафрейм 'movies_df':")
# print(movies_df)



# genres_df = pd.DataFrame({
#     'movie_id': [1, 2, 3, 4, 5],
#     'genre': ['Sci-Fi', 'Sci-Fi', 'Action', 'Crime', 'Drama']
# })

# print("\nДатафрейм 'genres_df':")
# print(genres_df)

# merged_movies_df = pd.merge(movies_df, genres_df, on='movie_id', how='left')
# print()
# print(merged_movies_df)

# import pandas as pd

# df = pd.DataFrame({'дата': pd.to_datetime(['2024-01-15', '2024-02-20'])})


# df['год'] = df['дата'].dt.year
# df['месяц'] = df['дата'].dt.month_name()
# df['день'] = df['дата'].dt.day
# df['день_недели'] = df['дата'].dt.day_name()

# import pandas as pd

# # Создаём DataFrame
# data = {
#     'дата': ['10-01-2023', '15-01-2023', '20-02-2023', '25-02-2023',
#              '01-03-2024', '05-03-2024', '10-04-2024', '15-04-2024'],
#     'температура': [12, 14, 16, 17, 15, 18, 20, 19]
# }

# df_temp = pd.DataFrame(data)

# print("Созданный DataFrame:")
# print(df_temp)
# print("\nИнформация о DataFrame до преобразования:")
# print(df_temp.info())

# df_temp['дата'] = pd.to_datetime(df_temp['дата'], format='%d-%m-%Y')


# df_temp['год'] = df_temp['дата'].dt.year
# df_temp['месяц'] = df_temp['дата'].dt.month
# df_temp['день'] = df_temp['дата'].dt.day

# quest_1 = df_temp[df_temp['температура'] > 15]
# print(quest_1)

# quest_2 = df_temp[df_temp['год'] == 2023]

# print(quest_2)



# import pandas as pd
# import matplotlib.pyplot as plt

# data = {
#     'Месяц': ['Янв', 'Фев', 'Мар', 'Апр', 'Май'],
#     'Продажи': [100, 150, 120, 180, 200]
# }
# df = pd.DataFrame(data)
# plt.plot(df['Месяц'], df['Продажи'])
# plt.title('Динамика продаж')
# plt.xlabel('Месяц')
# plt.ylabel('Продажи')
# plt.show()

# import pandas as pd
# from matplotlib import pyplot as plt

# data = {
#     'дата' : ['01-01-2024', '02-01-2024', '03-01-2024'],
#     'температура' : [0, -2, 5]
# }

# df = pd.DataFrame(data)

# plt.plot(df['дата'], df['температура'])
# plt.xlabel('Дата')
# plt.ylabel('Температура')
# plt.title('Динамика температур')
# plt.show()

# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt

# data = {
#     'Категория': ['A', 'B', 'A', 'B', 'A'],
#     'Значение': [10, 20, 15, 25, 12]
# }
# df = pd.DataFrame(data)


# sns.boxplot(x='Категория', y='Значение', data=df)
# plt.title('Сравнение значений по категориям')
# plt.show()








# Данные из таблицы
# data = {
#     '№': range(1, 21),
#     'id': [
#         4032772, 4527913, 4403127, 4223573, 3947019, 4608508, 4242850,
#         4395206, 4338865, 3873844, 3952704, 3784978, 4604163, 4449304,
#         3943602, 4469305, 4334991, 3820487, 4013792, 4392377
#     ],
#     'Р': [94, 81, 89, 91, 86, 86, 89, 89, 86, 86, 75, 67, 73, 89, 86, 86, 86, 66, 81, 70],
#     'М': [97, 88, 78, 76, 78, 72, 78, 76, 78, 76, 78, 84, 82, 74, 70, 80, 80, 82, 78, 80],
#     'Ф': [82, np.nan, np.nan, np.nan, 94, 67, np.nan, 51, np.nan, np.nan, 64, 79, np.nan, np.nan, np.nan, np.nan, 74, np.nan, np.nan, np.nan],
#     'И': [np.nan, 95, 83, 88, np.nan, 88, 85, 85, 78, 85, 83, 90, 88, 70, 80, 75, np.nan, 85, 75, 80],
#     'О': [np.nan] * 20,
#     'ИЯ': [np.nan] * 20,
#     'Х': [np.nan] * 20,
#     'ПМ': [np.nan] * 20,
#     'ПИ': [np.nan] * 20,
#     'Мех': [np.nan] * 20,
#     'ОЭиЭ': [np.nan] * 20,
#     'ОЭ': [np.nan] * 20,
#     'ИД': [5, 5, 10, 5, np.nan, 7, np.nan, np.nan, 5, np.nan, 8, 2, np.nan, 10, 7, np.nan, np.nan, 5, 2, 5],
#     'Σ': [278, 269, 260, 260, 258, 253, 252, 250, 247, 247, 244, 243, 243, 243, 243, 241, 240, 238, 236, 235]
# }

# # Создание DataFrame
# df = pd.DataFrame(data)

# # Установка '№ п/п' в качестве индекса (по желанию, для соответствия с таблицей)
# df.set_index('№', inplace=True)

# # Проверка и вывод результата
# print("Датасет успешно создан. Ниже представлена таблица для проверки:")
# print(df.to_string())



# print(f'Русский: {df['Р'].quantile(0.75)} ')
# print(f'Математика: {df['М'].quantile(0.75)} ')
# print(f'Физика: {df['Ф'].quantile(0.75)} ')
# print(f'Информатика: {df['И'].quantile(0.75)} ')


# plt.figure(figsize=(20,10))

# plt.subplot(1,4,1)
# sns.boxplot(y=df['Р'])
# plt.title('Русский')

# plt.subplot(1,4,2)
# sns.boxplot(y=df['М'])
# plt.title('Математика')

# plt.subplot(1,4,3)
# sns.boxplot(y=df['Ф'])
# plt.title('Физика')

# plt.subplot(1,4,4)
# sns.boxplot(y=df['И'])
# plt.title('Информатика')

# plt.show()





# movies_df = pd.DataFrame({
#     'title': ['Inception', 'The Matrix', 'The Dark Knight', 'Pulp Fiction', 'Forrest Gump'],
#     'genre': ['Sci-Fi', 'Sci-Fi', 'Action', 'Crime', 'Drama'],
#     'rating': [8.8, 8.7, 9.0, 8.9, 8.8]
# })


# # sns.barplot(x='genre', y='rating', data=movies_df)
# plt.bar(movies_df['genre'], movies_df['rating'])


# plt.title('avg rating by genre')
# plt.show()

# print(movies_df['rating'])
# print(np.percentile(movies_df['rating'], 25))

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

data = {
    'температура': [25, 28, 30, 22, 26, 29],
    'влажность': [60, 55, 50, 70, 65, 58],
    'продажи_мороженого': [100, 150, 200, 70, 120, 180]
}
df = pd.DataFrame(data)

correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, cmap='coolwarm')

plt.show()