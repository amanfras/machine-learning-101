# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 01:29:54 2021

@author: Adela
"""


from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
from sklearn import preprocessing
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy  as np  
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error


#%% Nos quedamos con los datos de la ciudad de madrid ya que hemos decidido hacer un estudio centrado en esta ciudad
 
dataframe = pd.read_csv("D:/keepcoding/machine-learning-101/project/airbnb-listings.csv", sep=';')

dataframe = dataframe[dataframe['City']=='Madrid']

#%% Arreglar los códigos postales debido a que varios de ellos se han introducido erróneamenrte en el dataset y es preferible solucionarlo antes de hacer nada

dataframe['Zipcode'][dataframe['Zipcode'] == ""] = math.nan
dataframe['Zipcode'][dataframe['Zipcode'] == "-"] = math.nan
dataframe['Zipcode'][dataframe['Zipcode'] == '28'] = math.nan

dataframe['Zipcode'][dataframe['Zipcode'] == 'Madrid 28004'] = '28004'

dataframe['Zipcode'][dataframe['Zipcode'] == '28002\n28002'] = '28002'
dataframe['Zipcode'][dataframe['Zipcode'] == '28051\n28051'] = '28051'

dataframe['Zipcode'][dataframe['Zipcode'] == '280013'] = '28013'
dataframe['Zipcode'][dataframe['Zipcode'] == '2015'] = '28015'
dataframe['Zipcode'][dataframe['Zipcode'] == '2815'] = '28015'
dataframe['Zipcode'][dataframe['Zipcode'] == '2805'] = '28005'
dataframe['Zipcode'][dataframe['Zipcode'] == '2804'] = '28004'

#%% Vamos a ver si hay alguna variable que nos interesa eliminar desde el principio por su tipo y que no nos aportan información

tipos = dataframe.dtypes

dataframe = dataframe.drop(['ID', 'Listing Url', 'Scrape ID', 'Last Scraped', 'Name', 'Summary', 'Space', 'Description', 'Experiences Offered', 'Neighborhood Overview', 'Notes', 'Transit', 'Access', 'Interaction', 'House Rules', 'Thumbnail Url', 'Medium Url', 'Picture Url', 'XL Picture Url', 'Host ID', 'Host URL', 'Host Name', 'Host Since', 'Host Location', 'Host About', 'Host Response Time', 'Host Response Rate', 'Host Acceptance Rate', 'Host Thumbnail Url', 'Host Picture Url', 'Host Total Listings Count', 'Host Verifications', 'Street', 'Host Neighbourhood', 'Neighbourhood Cleansed', 'Neighbourhood Group Cleansed', 'City', 'State', 'Market', 'Smart Location', 'Country Code', 'Country', 'Weekly Price', 'Monthly Price', 'Calendar Updated', 'Has Availability', 'Availability 30', 'Availability 60', 'Availability 90', 'Availability 365', 'Calendar last Scraped', 'First Review', 'Last Review', 'License', 'Jurisdiction Names', 'Calculated host listings count', 'Geolocation', 'Features'], axis=1)

#De este modo se ha decidido eliminar todas las variables en las que se describe el airbnb a través de frases, las url, los ID que no aportan datos, los datos de localización de ciudad y país.. debido a que ya se ha preseleccionado la ciudad de Madrid y no tendría sentido emplear estos campos, los de precios ya que no tiene sentido emplearlos para predecirlo, etc.

#%% Dividimos en train y test

dataframe_train, dataframe_test = train_test_split(dataframe, test_size = 0.3, shuffle = True, random_state = 0)

# Codificamos las variables de tipo string

dataframe_train['Neighbourhood'].fillna('Other', inplace=True)
le1 = preprocessing.LabelEncoder()
le1.fit(dataframe_train['Neighbourhood'])
dataframe_train['Neighbourhood'] = le1.transform(dataframe_train['Neighbourhood'])

dataframe_train['Zipcode'].fillna(dataframe_train['Zipcode'].mode()[0], inplace=True)
dataframe_train['Zipcode'] = dataframe_train['Zipcode'].astype(int)

le2 = preprocessing.LabelEncoder()
le2.fit(dataframe_train['Property Type'])
dataframe_train['Property Type'] = le2.transform(dataframe_train['Property Type'])

le3 = preprocessing.LabelEncoder()
le3.fit(dataframe_train['Room Type'])
dataframe_train['Room Type'] = le3.transform(dataframe_train['Room Type'])

le4 = preprocessing.LabelEncoder()
le4.fit(dataframe_train['Bed Type'])
dataframe_train['Bed Type'] = le4.transform(dataframe_train['Bed Type'])

dataframe_train['Amenities'].fillna('', inplace=True)
dataframe_train['Amenities'] = dataframe_train['Amenities'].apply(lambda x: len(x.split(',')))

le5 = preprocessing.LabelEncoder()
le5.fit(dataframe_train['Cancellation Policy'])
dataframe_train['Cancellation Policy'] = le5.transform(dataframe_train['Cancellation Policy'])

tipos = dataframe_train.dtypes # Ya son todos int/float

# Vemos el número de nans por si nos interesa quitarnos más variables o rellenar

nans = dataframe_train.isna().sum() 

dataframe_train['Host Listings Count'].fillna(dataframe_train['Host Listings Count'].mode()[0], inplace=True)
dataframe_train['Bathrooms'].fillna(dataframe_train['Bathrooms'].mode()[0], inplace=True)
dataframe_train['Bedrooms'].fillna(dataframe_train['Bedrooms'].mode()[0], inplace=True)
dataframe_train['Beds'].fillna(dataframe_train['Beds'].mode()[0], inplace=True)
dataframe_train = dataframe_train.drop(['Square Feet'], axis=1) # Porcentaje muy alto de valores nan no nos interesa
dataframe_train['Price'].fillna(dataframe_train['Price'].mode()[0], inplace=True)
dataframe_train = dataframe_train.drop(['Security Deposit'], axis=1)
dataframe_train['Cleaning Fee'].fillna(0, inplace=True) # Porcentaje muy alto de valores nan no nos interesa
dataframe_train['Review Scores Rating'].fillna(dataframe_train['Review Scores Rating'].mode()[0], inplace=True)
dataframe_train['Review Scores Accuracy'].fillna(dataframe_train['Review Scores Accuracy'].mode()[0], inplace=True)
dataframe_train['Review Scores Cleanliness'].fillna(dataframe_train['Review Scores Cleanliness'].mode()[0], inplace=True)
dataframe_train['Review Scores Checkin'].fillna(dataframe_train['Review Scores Checkin'].mode()[0], inplace=True)
dataframe_train['Review Scores Communication'].fillna(dataframe_train['Review Scores Communication'].mode()[0], inplace=True)
dataframe_train['Review Scores Location'].fillna(dataframe_train['Review Scores Location'].mode()[0], inplace=True)
dataframe_train['Review Scores Value'].fillna(dataframe_train['Review Scores Value'].mode()[0], inplace=True)
dataframe_train['Reviews per Month'].fillna(dataframe_train['Reviews per Month'].mode()[0], inplace=True)


#%% Comenzamos el análisis exploratorio

descripcion = dataframe_train.describe().T

#Plots de las variables que tiene sentido mirar si hay outliers basándonos en la descripción anterior

dataframe_train['Host Listings Count'].plot.hist(alpha=0.5, bins=25, grid = True)
plt.xlabel('Host Listings Count')
dataframe_train_2 = dataframe_train[dataframe_train['Host Listings Count'] < 210] # Quitamos outliers

dataframe_train_2['Latitude'].plot.hist(alpha=0.5, bins=25, grid = True)
plt.xlabel('Latitude')

dataframe_train_2['Longitude'].plot.hist(alpha=0.5, bins=25, grid = True)
plt.xlabel('Longitude')

dataframe_train_2['Accommodates'].plot.hist(alpha=0.5, bins=25, grid = True)
plt.xlabel('Accommodates')
dataframe_train_2['Accommodates'].value_counts()
dataframe_train_2 = dataframe_train_2[dataframe_train_2['Accommodates'] < 13] # Quitamos outliers

dataframe_train_2['Bathrooms'].plot.hist(alpha=0.5, bins=25, grid = True)
plt.xlabel('Bathrooms')
dataframe_train_2['Bathrooms'].value_counts()
dataframe_train_2 = dataframe_train_2[dataframe_train_2['Bathrooms'] < 6.1] # Quitamos outliers

dataframe_train_2['Bedrooms'].plot.hist(alpha=0.5, bins=25, grid = True)
plt.xlabel('Bedrooms')
dataframe_train_2['Bedrooms'].value_counts()
dataframe_train_2 = dataframe_train_2[dataframe_train_2['Bedrooms'] < 6.1] # Quitamos outliers

dataframe_train_2['Beds'].plot.hist(alpha=0.5, bins=25, grid = True)
plt.xlabel('Beds')
dataframe_train_2['Beds'].value_counts()
dataframe_train_2 = dataframe_train_2[dataframe_train_2['Beds'] < 11] # Quitamos outliers

dataframe_train_2['Guests Included'].plot.hist(alpha=0.5, bins=25, grid = True)
plt.xlabel('Guests Included')
dataframe_train_2['Guests Included'].value_counts()
dataframe_train_2 = dataframe_train_2[dataframe_train_2['Guests Included'] < 10] # Quitamos outliers

dataframe_train_2['Extra People'].plot.hist(alpha=0.5, bins=25, grid = True)
plt.xlabel('Extra People')
dataframe_train_2['Extra People'].value_counts()
dataframe_train_2 = dataframe_train_2[dataframe_train_2['Extra People'] < 50] # Quitamos outliers

dataframe_train_2['Price'].plot.hist(alpha=0.5, bins=25, grid = True)
plt.xlabel('Price')
dataframe_train_2['Price'].value_counts()
dataframe_train_2 = dataframe_train_2[dataframe_train_2['Price'] < 400] # Quitamos outliers

# Scatter plot de la variable objetivo definida y como variable dependiente y algunas de las variables explicativas como independientes, en el caso de algunas de las codificadas usaremos waterfront

dataframe_train_2.plot(kind = 'scatter',x='Host Listings Count',y = 'Price')
dataframe_train_2.boxplot(by='Room Type',column = 'Price')
dataframe_train_2.plot(kind = 'scatter',x='Accommodates',y = 'Price')
dataframe_train_2.plot(kind = 'scatter',x='Bathrooms',y = 'Price')
dataframe_train_2.plot(kind = 'scatter',x='Bedrooms',y = 'Price')
dataframe_train_2.plot(kind = 'scatter',x='Beds',y = 'Price')
dataframe_train_2.boxplot(by='Bed Type',column = 'Price')
dataframe_train_2.plot(kind = 'scatter',x='Amenities',y = 'Price')
dataframe_train_2.plot(kind = 'scatter',x='Guests Included',y = 'Price')
dataframe_train_2.plot(kind = 'scatter',x='Extra People',y = 'Price')
dataframe_train_2.plot(kind = 'scatter',x='Review Scores Rating',y = 'Price')
dataframe_train_2.boxplot(by='Cancellation Policy',column = 'Price')
dataframe_train_2.plot(kind = 'scatter',x='Minimum Nights',y = 'Price')
dataframe_train_2['Minimum Nights'].value_counts()
dataframe_train_2 = dataframe_train_2[dataframe_train_2['Minimum Nights'] < 50] # Quitamos outliers

print(f'Porcentaje de registros eliminados: {((dataframe_train.shape[0] - dataframe_train_2.shape[0])/dataframe_train.shape[0])*100}%')


# Vamos a ver si hay colinealidad 

corr = np.abs(dataframe_train_2.drop(['Price'], axis=1).corr())
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, mask=mask,vmin = 0.0, vmax=1.0, center=0.5,
            linewidths=.1, cmap="YlGnBu", cbar_kws={"shrink": .8})

# Alta colinealidad entre beds y accommodates, eliminamos accommodates, vemos que hay relación entre las review scores, en concreto:
# Review Scores Rating = (Review Scores Accuracy + Review Scores Cleanliness + Review Scores Checkin + Review Scores Communication + Review Scores Location + Review Scores Value)*100/60
# Nos quedaremos con Review Scores Rating que es la combinación de las demás

dataframe_train_final = dataframe_train_2.drop(['Accommodates', 'Review Scores Accuracy', 'Review Scores Cleanliness', 'Review Scores Checkin', 'Review Scores Communication', 'Review Scores Location', 'Review Scores Value'], axis=1)
pd.plotting.scatter_matrix(dataframe_train_final, alpha=0.2, figsize=(20, 20), diagonal = 'kde')

# Vamos a ver la relación con la variable dependiente, dividimos en variable dependiente e independientes

Y_train = dataframe_train_final['Price']
X_train = dataframe_train_final.drop(['Price'], axis=1)

feature_names = X_train.columns
f_test, _ = f_regression(X_train, Y_train)
f_test /= np.max(f_test)
mi = mutual_info_regression(X_train, Y_train)
mi /= np.max(mi)

plt.figure(figsize=(20, 5))

plt.subplot(1,2,1)
plt.bar(range(X_train.shape[1]),f_test,  align="center")
plt.xticks(range(X_train.shape[1]),feature_names, rotation = 90)
plt.xlabel('features')
plt.ylabel('Ranking')
plt.title('$FTest$ score')

plt.subplot(1,2,2)
plt.bar(range(X_train.shape[1]),mi, align="center")
plt.xticks(range(X_train.shape[1]),feature_names, rotation = 90)
plt.xlabel('features')
plt.ylabel('Ranking')
plt.title('Mutual information score')

# Con estos resultados vamos a eliminar un par de variables más debido a su puntuación baja en el Ftest y mutual information score

X_train = X_train.drop(['Property Type', 'Bed Type', 'Maximum Nights'], axis=1)

#%% Preparamos para modelado, para ello vamos a realizar en el test los fillna y codificaciones que hicimos en el train

dataframe_test['Price'].fillna(dataframe_test['Price'].mode()[0], inplace=True)
dataframe_test = dataframe_test[dataframe_test['Price'] < 400]

dataframe_test = dataframe_test.drop(['Square Feet', 'Security Deposit', 'Property Type', 'Bed Type', 'Maximum Nights', 'Accommodates', 'Review Scores Accuracy', 'Review Scores Cleanliness', 'Review Scores Checkin', 'Review Scores Communication', 'Review Scores Location', 'Review Scores Value'], axis=1)

dataframe_test['Host Listings Count'].fillna(dataframe_test['Host Listings Count'].mode()[0], inplace=True)
dataframe_test = dataframe_test[dataframe_test['Host Listings Count'] < 210]

dataframe_test['Neighbourhood'].fillna('Other', inplace=True)
dataframe_test['Neighbourhood'] = le1.transform(dataframe_test['Neighbourhood'])

dataframe_test['Zipcode'].fillna(dataframe_test['Zipcode'].mode()[0], inplace=True)
dataframe_test['Zipcode'] = dataframe_test['Zipcode'].astype(int)

dataframe_test['Room Type'] = le3.transform(dataframe_test['Room Type'])

dataframe_test['Bathrooms'].fillna(dataframe_test['Bathrooms'].mode()[0], inplace=True)
dataframe_test = dataframe_test[dataframe_test['Bathrooms'] < 6.1]

dataframe_test['Bedrooms'].fillna(dataframe_test['Bedrooms'].mode()[0], inplace=True)
dataframe_test = dataframe_test[dataframe_test['Bedrooms'] < 6.1]

dataframe_test['Beds'].fillna(dataframe_test['Beds'].mode()[0], inplace=True)
dataframe_test = dataframe_test[dataframe_test['Beds'] < 11]

dataframe_test['Amenities'].fillna('', inplace=True)
dataframe_test['Amenities'] = dataframe_test['Amenities'].apply(lambda x: len(x.split(',')))

dataframe_test['Cleaning Fee'].fillna(0, inplace=True) 

dataframe_test = dataframe_test[dataframe_test['Guests Included'] < 10]

dataframe_test = dataframe_test[dataframe_test['Extra People'] < 50]

dataframe_test = dataframe_test[dataframe_test['Minimum Nights'] < 50]

dataframe_test['Review Scores Rating'].fillna(dataframe_test['Review Scores Rating'].mode()[0], inplace=True)

dataframe_test['Cancellation Policy'] = le5.transform(dataframe_test['Cancellation Policy'])

dataframe_test['Reviews per Month'].fillna(dataframe_test['Reviews per Month'].mode()[0], inplace=True)

# Dividimos en en variable dependiente e independientes

Y_test = dataframe_test['Price']
X_test = dataframe_test.drop(['Price'],  axis=1)

# Escalamos con los datos de train

scaler = preprocessing.StandardScaler().fit(X_train)
XtrainScaled = scaler.transform(X_train)
XtestScaled = scaler.transform(X_test)

#Vamos a hacer el histograma de la variable dependiente para ver si una transformación en ella nos sería más óptima

plt.subplot(1,2,1)
plt.hist(Y_train, bins=30)
plt.subplot(1,2,2)
plt.hist(np.log10(Y_train), bins=30)

#Comprobamos que nos es más óptimo trabajar con el logaritmo en base 10 de dicha variable 

#%% Primero utilizamos la capacidad de Lasso para seleccionar variable

alpha_vector_lasso = np.logspace(-5,5,50)
param_grid_lasso = {'alpha': alpha_vector_lasso}
grid_lasso = GridSearchCV(Lasso(), scoring= 'neg_mean_squared_error', param_grid=param_grid_lasso, cv = 5)
grid_lasso.fit(XtrainScaled, np.log10(Y_train))
print("best mean cross-validation score: {:.3f}".format(grid_lasso.best_score_))
print("best parameters: {}".format(grid_lasso.best_params_))
scores = -1*np.array(grid_lasso.cv_results_['mean_test_score'])
plt.semilogx(alpha_vector_lasso,scores,'-o')
plt.xlabel('alpha',fontsize=16)
plt.ylabel('5-Fold MSE')

alpha_vector_lasso = np.logspace(-3,-2,50)  # hacemos zoom
param_grid_lasso = {'alpha': alpha_vector_lasso}
grid_lasso = GridSearchCV(Lasso(), scoring= 'neg_mean_squared_error', param_grid=param_grid_lasso, cv = 5)
grid_lasso.fit(XtrainScaled, np.log10(Y_train))
print("best mean cross-validation score: {:.3f}".format(grid_lasso.best_score_))
print("best parameters: {}".format(grid_lasso.best_params_))
scores = -1*np.array(grid_lasso.cv_results_['mean_test_score'])
plt.semilogx(alpha_vector_lasso,scores,'-o')
plt.xlabel('alpha',fontsize=16)
plt.ylabel('5-Fold MSE')

alpha_optimo_lasso = grid_lasso.best_params_['alpha']
lasso = Lasso(alpha = alpha_optimo_lasso).fit(XtrainScaled,np.log10(Y_train))
ytrainLasso = lasso.predict(XtrainScaled)
ytestLasso  = lasso.predict(XtestScaled)
mseTrainModelLasso = mean_squared_error(Y_train,pow(10,ytrainLasso))
mseTestModelLasso = mean_squared_error(Y_test, pow(10, ytestLasso))
print('MSE Modelo Lasso (train): %0.3g' % mseTrainModelLasso)
print('MSE Modelo Lasso (test) : %0.3g' % mseTestModelLasso)
print('RMSE Modelo Lasso (train): %0.3g' % np.sqrt(mseTrainModelLasso))
print('RMSE Modelo Lasso (test) : %0.3g' % np.sqrt(mseTestModelLasso))
w_lasso = lasso.coef_
for f,wi in zip(feature_names,w_lasso):
    print(f,wi)


#%% Ahora vamos a emplear el modelo de Ridge regression

alpha_vector_ridge = np.logspace(1,7,50)
param_grid_ridge = {'alpha': alpha_vector_ridge}
grid_ridge = GridSearchCV(Ridge(), param_grid=param_grid_ridge, cv = 5)
grid_ridge.fit(XtrainScaled, np.log10(Y_train))
print("best mean cross-validation score: {:.3f}".format(grid_ridge.best_score_))
print("best parameters: {}".format(grid_ridge.best_params_))
scores = np.array(grid_ridge.cv_results_['mean_test_score'])
plt.semilogx(alpha_vector_ridge,scores,'-o')
plt.xlabel('alpha',fontsize=16)
plt.ylabel('5-Fold MSE')

alpha_vector_ridge = np.logspace(1,3,50) # hacemos zoom
param_grid_ridge = {'alpha': alpha_vector_ridge}
grid_ridge = GridSearchCV(Ridge(), param_grid=param_grid_ridge, cv = 5)
grid_ridge.fit(XtrainScaled, np.log10(Y_train))
print("best mean cross-validation score: {:.3f}".format(grid_ridge.best_score_))
print("best parameters: {}".format(grid_ridge.best_params_))
scores = np.array(grid_ridge.cv_results_['mean_test_score'])
plt.semilogx(alpha_vector_ridge,scores,'-o')
plt.xlabel('alpha',fontsize=16)
plt.ylabel('5-Fold MSE')

alpha_optimo_ridge = grid_ridge.best_params_['alpha']
ridge = Ridge(alpha = alpha_optimo_ridge).fit(XtrainScaled,np.log10(Y_train))
ytrainRidge = ridge.predict(XtrainScaled)
ytestRidge  = ridge.predict(XtestScaled)
mseTrainModelRidge = mean_squared_error(Y_train,pow(10,ytrainRidge))
mseTestModelRidge = mean_squared_error(Y_test, pow(10, ytestRidge))
print('MSE Modelo Ridge (train): %0.3g' % mseTrainModelRidge)
print('MSE Modelo Ridge (test) : %0.3g' % mseTestModelRidge)
print('RMSE Modelo Ridge (train): %0.3g' % np.sqrt(mseTrainModelRidge))
print('RMSE Modelo Ridge (test) : %0.3g' % np.sqrt(mseTestModelRidge))
w_ridge = ridge.coef_
for f,wi in zip(feature_names,w_ridge):
    print(f,wi)
    
#%% Observando los MSE y RMSE vemos que son muy similares con ambos métodos, pero mediante Lasso hemos descartado dos variables independientes por lo que se convierte 
#en un modelo algo más simple y con unos errores muy similares (se puede comprobar en la gráfica que hay a continuación) por lo que sería el modelo más óptimo.

plt.plot(range(len(Y_train),len(Y_train)+len(Y_test)),Y_test,label='test')
plt.plot(range(len(Y_train),len(Y_train)+len(Y_test)),pow(10, ytestRidge),label='prediccion Ridge')
plt.plot(range(len(Y_train),len(Y_train)+len(Y_test)),pow(10, ytestLasso),label='prediccion Lasso')
plt.legend()