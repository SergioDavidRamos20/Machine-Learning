import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import seaborn as sb

#matplotlib inline
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer 

hepatitisknn = pd.read_csv("HepatitisCdata.csv")


x = hepatitisknn.iloc[:,2:].values
y = hepatitisknn.iloc[:,1].values

#preprocesamiento NA

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:,2:])
x[:,2:]=imputer.transform(x[:,2:])


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_neighbors = 7

knn = KNeighborsClassifier(n_neighbors)
knn.fit(x_train, y_train)

print('Precision del clasificador KNN en el conjunto de entrenamiento \n'
      ,format(knn.score(x_train, y_train)))

print('Precision del clasificador KNN en el conjunto de prueba \n'
      ,format(knn.score(x_test, y_test)))




"""
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x, y)

KNeighborsClassifier(n_neighbors=3)

prediction = knn.predict()
"""
    


 
