### HAPPINESS UP ###
### Экономический проект, способный поднять экономику Российской Федерации и СНГ в целом. ###

# Датасеты:
# https://docs.google.com/spreadsheets/d/1HDshC3DOggWEd-goJxOZRaMK8hnC8mYNf2sOjXmSz7Y/edit#gid=302619806
#https://docs.google.com/spreadsheets/d/1gR7oprLt0FHbdz2S-A4ZMP5MKyiw1EQLZrD_-ydYY2s/edit#gid=381670079
#https://docs.google.com/spreadsheets/d/1bu6xrHNdMyCz2Smp6DmM4sYMsR-EwGONlVoT5WgGra8/edit#gid=1720320222
#https://docs.google.com/spreadsheets/d/1FDbth0OmC4yX6w0qieKXo8no9FHgaIgDnakqgc2CeN4/edit#gid=1830665464
#https://docs.google.com/spreadsheets/d/1kG2m6FQPkOJDEz9NKm25ewZfqbMOwe8Eu7FkV0OgSPs/edit#gid=432236967
#https://docs.google.com/spreadsheets/d/1EtCkhDhptEGKD4nYBoiuwJrff48SvM0uK2DMrC3QWmw/edit#gid=1286559583
#https://docs.google.com/spreadsheets/d/1-w4l8Xjo99ku80sKMXfv8FE3hzfDUVnB/edit?dls=true#gid=1981302677

%matplotlib inline
 
# Импортирование необходимых библиотек.

pip install pytelegrambotapi

import sklearn
from fbprophet import Prophet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import telebot

# Подключение Google Drive.
 
from google.colab import drive
drive.mount('/content/drive/')

# Загрузка необходимой директории с Google Drive.

PATH = '/content/drive/My Drive/'

# Новые датасеты:
gdp_all = pd.read_excel('/content/drive/Shared drives/Campus_K2/Money_History/ProjectedRealPerCapitaGDPValues.xls', skiprows=10)
happiness_2015 = pd.read_csv('/content/drive/Shared drives/Campus_K2/Money_History/hp_2015.csv') # Список стран по индексу счастья — 2015г.
happiness_2016 = pd.read_csv('/content/drive/Shared drives/Campus_K2/Money_History/hp_2016.csv') # Список стран по индексу счастья — 2016г.
happiness_2017 = pd.read_csv('/content/drive/Shared drives/Campus_K2/Money_History/hp_2017.csv') # Список стран по индексу счастья — 2017г.
happiness_2018 = pd.read_csv('/content/drive/Shared drives/Campus_K2/Money_History/hp_2018.csv') # Список стран по индексу счастья — 2018г.
happiness_2019 = pd.read_csv('/content/drive/Shared drives/Campus_K2/Money_History/hp_2019.csv') # Список стран по индексу счастья — 2019г.
happiness_2020 = pd.read_csv('/content/drive/Shared drives/Campus_K2/Money_History/hp_2020.csv') # Список стран по индексу счастья — 2020г.

# Разделяем страны на группы по географическому положению (СНГ, Евросоюз, Азия, Американский континент) (ВВП и уровень счастья)
 
country_list_cis = ['Russia', 'Ukraine', 'Belarus', 'Armenia', 'Azerbaijan', 'Kazakhstan', 'Kyrgyzstan', 'Moldova', 'Tajikistan', 'Uzbekistan']
country_list_eu = ['Austria', 'Belgium', 'United Kingdom', 'Bulgaria', 'Hungary', 'Germany', 'Greece', 'Egypt', 'Ireland', 'Spain', 'North Cyprus', 'Latvia']
country_list_asia = ['China', 'Japan', 'South Korea']
country_list_america = ['United States']
 
country_gdp_cis = ['Russia', 'Ukraine', 'Belarus', 'Armenia', 'Azerbaijan', 'Kazakhstan', 'Kyrgyzstan', 'Moldova', 'Tajikistan', 'Uzbekistan']
country_gdp_eu = ['Austria', 'Belgium', 'United Kingdom', 'Bulgaria', 'Hungary', 'Germany', 'Greece', 'Egypt, Arab Rep.', 'Ireland', 'Spain', 'Cyprus', 'Latvia']
country_gdp_asia = ['China', 'Japan', 'Korea, Rep.']
country_gdp_america = ['United States']

# Индекс счастья в СНГ — 2015г.
 
print("Страны «Содружества Независимых Государств»:")
country_list_cis = ['Russia', 'Ukraine', 'Belarus', 'Armenia', 'Azerbaijan', 'Kazakhstan', 'Kyrgyzstan', 'Moldova', 'Tajikistan', 'Uzbekistan']
cis_analytics15 = happiness_2015[happiness_2015['Country'].isin(country_list_cis)][['Country','Happiness Score']]
cis_analytics15

# Индекс счастья в Евросоюзе — 2015г.
 
print("Страны «Евросоюза»:")
country_list_eu = ['Austria', 'Belgium', 'United Kingdom', 'Bulgaria', 'Hungary', 'Germany', 'Greece', 'Egypt', 'Ireland', 'Spain', 'North Cyprus', 'Latvia']
eu_analytics15 = happiness_2015[happiness_2015['Country'].isin(country_list_eu)][['Country','Happiness Score']]
eu_analytics15

# Индекс счастья в Азиатских государствах — 2015г.
 
print('Азиатские государства:')
country_list_asia = ['China', 'Japan', 'South Korea']
asia_analytics15 = happiness_2015[happiness_2015['Country'].isin(country_list_asia)][['Country','Happiness Score']]
asia_analytics15

# Индекс счастья в СНГ — 2016г.
 
print("Страны «Содружества Независимых Государств»:")
cis_analytics16 = happiness_2016[happiness_2016['Country'].isin(country_list_cis)][['Country','Happiness Score']]
cis_analytics16

# Индекс счастья в Евросоюзе — 2016г.
 
print("Страны «Евросоюза»:")
eu_analytics16 = happiness_2016[happiness_2016['Country'].isin(country_list_eu)][['Country','Happiness Score']]
eu_analytics16

# Индекс счастья в Азиатских государствах — 2016г.
 
print('Азиатские государства:')
asia_analytics16 = happiness_2016[happiness_2016['Country'].isin(country_list_asia)][['Country','Happiness Score']]
asia_analytics16

# Индекс счастья на Американском континенте — 2016г.
 
print('Американский континент:')
america_analytics16 = happiness_2016[happiness_2016['Country'].isin(country_list_america)][['Country','Happiness Score']]
america_analytics16

# Индекс счастья в СНГ — 2017г.
 
print("Страны «Содружества Независимых Государств»:")
cis_analytics17 = happiness_2017[happiness_2017['Country'].isin(country_list_cis)][['Country','Happiness.Score']]
cis_analytics17

# Индекс счастья в Евросоюзе — 2017г.
 
print("Страны «Евросоюза»:")
eu_analytics17 = happiness_2017[happiness_2017['Country'].isin(country_list_eu)][['Country','Happiness.Score']]
eu_analytics17

# Индекс счастья в Азиатских государствах — 2017г.
 
print('Азиатские государства:')
asia_analytics17 = happiness_2017[happiness_2017['Country'].isin(country_list_asia)][['Country','Happiness.Score']]
asia_analytics17

# Индекс счастья на Американском континенте — 2017г.
 
print('Американский континент:')
america_analytics17 = happiness_2017[happiness_2017['Country'].isin(country_list_america)][['Country','Happiness.Score']]
america_analytics17

# Индекс счастья в СНГ — 2018г.

print("Страны «Содружества Независимых Государств»:")
cis_analytics18 = happiness_2018[happiness_2018['Country or region'].isin(country_list_cis)][['Country or region','Score']]
cis_analytics18

# Индекс счастья в Евросоюзе — 2018г.

print("Страны «Евросоюза»:")
eu_analytics18 = happiness_2018[happiness_2018['Country or region'].isin(country_list_eu)][['Country or region','Score']]
eu_analytics18

# Индекс счастья в Азиатских государствах — 2018г.

print('Азиатские государства:')
asia_analytics18 = happiness_2018[happiness_2018['Country or region'].isin(country_list_asia)][['Country or region','Score']]
asia_analytics18

# Индекс счастья на Американском континенте — 2018г.

print('Американский континент:')
america_analytics18 = happiness_2018[happiness_2018['Country or region'].isin(country_list_america)][['Country or region','Score']]
america_analytics18

# Индекс счастья в СНГ — 2019г.

print("Страны «Содружества Независимых Государств»:")
cis_analytics19 = happiness_2019[happiness_2019['Country or region'].isin(country_list_cis)][['Country or region','Score']]
cis_analytics19

# Индекс счастья в Евросоюзе — 2019г.

print("Страны «Евросоюза»:")
eu_analytics19 = happiness_2019[happiness_2019['Country or region'].isin(country_list_eu)][['Country or region','Score']]
eu_analytics19

# Индекс счастья в Азиатских государствах — 2019г.

print('Азиатские государства:')
asia_analytics19 = happiness_2019[happiness_2019['Country or region'].isin(country_list_asia)][['Country or region','Score']]
asia_analytics19

# Индекс счастья на Американском континнте — 2019г.

print('Американский континент:')
america_analytics19 = happiness_2019[happiness_2019['Country or region'].isin(country_list_america)][['Country or region','Score']]
america_analytics19

# Индекс счастья в СНГ — 2020г.

print("Страны «Содружества Независимых Государств»:")
cis_analytics20 = happiness_2020[happiness_2020['Country or region'].isin(country_list_cis)][['Country or region','Score']]
cis_analytics20

# Индекс счастья в Евросоюзе — 2020г.

print("Страны «Евросоюза»:")
eu_analytics20 = happiness_2020[happiness_2020['Country or region'].isin(country_list_eu)][['Country or region','Score']]
eu_analytics20

# Индекс счастья в Азиатских государствах — 2020г.

print('Азиатские государства:')
asia_analytics20 = happiness_2020[happiness_2020['Country or region'].isin(country_list_asia)][['Country or region','Score']]
asia_analytics20

# Индекс счастья на Американском континенте — 2020г.

print('Американский континент:')
america_analytics20 = happiness_2020[happiness_2020['Country or region'].isin(country_list_america)][['Country or region','Score']]
america_analytics20

# ВВП стран СНГ

print("Страны СНГ:")
GDP_cis = gdp_all[gdp_all['Country'].isin(country_gdp_cis)][['Country', 2015, 2016, 2017, 2018, 2019, 2020]]
GDP_cis

# ВВП стран Евросоюза

print("Страны «Евросоюза»:")
GDP_eu = gdp_all[gdp_all['Country'].isin(country_gdp_eu)][['Country', 2015, 2016, 2017, 2018, 2019, 2020]]
GDP_eu

# ВВП Азиатских государств

print("Азиатские государства:")
GDP_asia = gdp_all[gdp_all['Country'].isin(country_gdp_asia)][['Country', 2015, 2016, 2017, 2018, 2019, 2020]]
GDP_asia

# ВВП стран Американского континента

print("Американский континент:")
GDP_america = gdp_all[gdp_all['Country'].isin(country_gdp_america)][['Country', 2015, 2016, 2017, 2018, 2019, 2020]]
GDP_america

print(happiness_2017.loc[48]) # На 2017-й год индекс счастья Российской Федерации был равен 5.963.

print(happiness_2018.loc[58]) # На 2018-й год индекс счастья Российской Федерации был равен 5.81.

print(happiness_2019.loc[67]) # На 2019-й год индекс счастья Российской Федерации был равен 5.648.

print(happiness_2020.loc[72]) # На 2020-й год индекс счастья Российской Федерации равен 5.546.

gdp_all.describe()
# Вывод стандартных отклонений набора данных.

happiness_2015.describe()
# Вывод стандартных отклонений набора данных.

happiness_2016.describe()
# Вывод стандартных отклонений набора данных.

happiness_2017.describe()
# Вывод стандартных отклонений набора данных.

happiness_2018.describe()
# Вывод стандартных отклонений набора данных.

happiness_2019.describe()
# Вывод стандартных отклонений набора данных.

happiness_2020.describe()
# Вывод стандартных отклонений набора данных.

# Диаграмма

fig, ax = plt.subplots(figsize=(20, 10))
plt.bar(GDP_america['Country'].values, GDP_america[2015].values)
ax.set_xlabel('Название страны')
ax.set_ylabel('Уровень счастья')
plt.show()

# Диаграмма

fig, ax = plt.subplots(figsize=(20, 10))
plt.bar(GDP_cis['Country'].values, GDP_cis[2015].values)
ax.set_xlabel('Название страны')
ax.set_ylabel('Уровень счастья')
plt.show()

# Диаграмма

fig, ax = plt.subplots(figsize=(20, 10))
plt.bar(eu_analytics15['Country'].values, eu_analytics15['Happiness Score'].values)
ax.set_xlabel('Название страны')
ax.set_ylabel('Уровень счастья')
plt.show()

# Диаграмма

fig, ax = plt.subplots(figsize=(20, 10))
plt.bar(GDP_eu['Country'].values, GDP_eu[2015].values)
ax.set_xlabel('Название страны')
ax.set_ylabel('Уровень счастья')
plt.show()

sns.heatmap(happiness_2015.corr(), annot=True,cmap='RdYlGn',linewidths=0.2) # Кореляции 
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()

# Корреляция

sns.heatmap(happiness_2016.corr(), annot=True,cmap='RdYlGn',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()

# Корреляция

sns.heatmap(happiness_2017.corr(), annot=True,cmap='RdYlGn',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()

# Корреляция

sns.heatmap(happiness_2018.corr(), annot=True,cmap='RdYlGn',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()

# Корреляция

sns.heatmap(happiness_2019.corr(), annot=True,cmap='RdYlGn',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(30,20)
plt.show()

# Создание датафрейма для обучения модели уровня счастья

russia_all_happiness_score = [5.716, 5.856, 5.963, 5.81]
buf = pd.DataFrame()
buf['y'] = russia_all_happiness_score
buf['ds'] = ['2015', '2016', '2017', '2018']
buf['ds'] = pd.to_datetime(buf['ds'])

# Создание датафрейма для обучения улучшенной модели прогноза уровня счастья

russia_all_happiness_score = [5.716, 5.856, 5.963, 5.81]
buf2 = pd.DataFrame()
buf2['y'] = russia_all_happiness_score
buf2['ds'] = ['2015', '2016', '2017', '2018']
buf2['ds'] = pd.to_datetime(buf2['ds'])

# Обучение модели

model = Prophet(interval_width=0.95) # обучаем модель предсказывать ВВП с процентом неопределенности в будущем 95%
model.fit(buf)

# Обучение модели

model2 = Prophet(interval_width=0.95) # обучаем модель предсказывать уровень счастья с процентом неопределенности в будущем 95%
model2.fit(buf2)

# Вывод значения ВВП для обучения модели

buf

# Вывод значения уровня счастья для обучения модели

buf2

# Предсказание значений ВВП Российской Федерации на ближайшие 10 лет

forecast1 = model.make_future_dataframe(periods=10, freq='Y')
forecast1 = model.predict(forecast1)
forecast1 = forecast1.loc[forecast1['ds'] != '2018-12-31']
plt.figure(figsize=(18, 6))
model.plot(forecast1, xlabel = 'Год', ylabel = 'Значение ВВП')
plt.title('Предсказание значений ВВП Российской Федерации на ближайшие 10 лет')

# Метрики

metric_df = forecast1.set_index('ds')[['yhat']][:6].reset_index()
metric_df['y'] = GDP_cis[GDP_cis['Country'].isin(['Russia'])].melt()[1:]['value'].values.tolist()
metric_df

# Предсказание значений уровня счастья на ближайшие 10 лет

forecast2 = model2.make_future_dataframe(periods=10, freq='Y')
forecast2 = model2.predict(forecast2)

plt.figure(figsize=(18, 6))
model2.plot(forecast2, xlabel = 'Год', ylabel = 'Уровень счастья')
plt.title('Предсказание значений уровня счастья на ближайшие 10 лет');

# Разделяем данные на train и test 

from sklearn.model_selection import train_test_split
happiness_2019.head()
target = 'Score'
train_x, test_x, train_y, test_y = train_test_split(happiness_2019.drop(target, axis = 1), happiness_2019[target])

# код с комментариями, что он делает
metric_df

# Оценка качества метрики (пригодится позже)

#from sklearn.metrics import r2_score
#sklearn.metrics.r2_score(metric_df.y, metric_df.yhat)

#from sklearn.metrics import mean_absolute_error
#sklearn.metrics.mean_absolute_error(metric_df.y, metric_df.yhat)

# Разбиваем модели на компоненты

model.plot_components(forecast1)

# Разбиваем модели на компоненты

model.plot_components(forecast2)

# Создание датафрейма для обучения модели прогноза ВВП

russia_all_gdp_modernizated = [11642.461914, 11621.854492,	11821.383789,	12099.396484, 12319.634766, 12574.315430]
buf3 = pd.DataFrame()
buf3['y'] = russia_all_gdp_modernizated
buf3['ds'] = ['2015', '2016', '2017', '2018', '2019', '2020']
buf3['ds'] = pd.to_datetime(buf3['ds'])

# Создание датафрейма для обучения модели уровня счастья

russia_all_happiness_score_modernizted = [5.716, 5.856, 5.963, 5.81, 5.648, 5.5460]
buf4 = pd.DataFrame()
buf4['y'] = russia_all_happiness_score_modernizted
buf4['ds'] = ['2015', '2016', '2017', '2018', '2019', '2020']
buf4['ds'] = pd.to_datetime(buf4['ds'])

# Обучение улучшенной модели

model3 = Prophet(interval_width=0.95) # обучаем модель предсказывать ВВП с процентом неопределенности в будущем 95%
model3.fit(buf3)

# Обучение улучшенной модели

model4 = Prophet(interval_width=0.95) # обучаем модель предсказывать ВВП с процентом неопределенности в будущем 95%
model4.fit(buf4)

# Предсказание значений уровня счастья на ближайшие 10 лет

forecast3 = model3.make_future_dataframe(periods=10, freq='Y')
forecast3 = model3.predict(forecast3)

plt.figure(figsize=(18, 6))
model3.plot(forecast3, xlabel = 'Год', ylabel = 'Уровень счастья')
plt.title('Предсказание значений уровня счастья на ближайшие 10 лет');

# Предсказание значений уровня счастья на ближайшие 10 лет

forecast4 = model4.make_future_dataframe(periods=10, freq='Y')
forecast4 = model4.predict(forecast4)

plt.figure(figsize=(18, 6))
model4.plot(forecast4, xlabel = 'Год', ylabel = 'Уровень счастья')
plt.title('Предсказание значений уровня счастья на ближайшие 10 лет');
