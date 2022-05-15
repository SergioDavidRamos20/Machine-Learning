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
from yellowbrick.classifier import ConfusionMatrix 

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

n_neighbors = 10

knn = KNeighborsClassifier(n_neighbors)
knn.fit(x_train, y_train)

#Evaluacion
print('Precision del clasificador KNN en el conjunto de entrenamiento \n'
      ,format(knn.score(x_train, y_train)))

print('Precision del clasificador KNN en el conjunto de prueba \n'
      ,format(knn.score(x_test, y_test)))

pred = knn.predict(x_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))


confusion=ConfusionMatrix(knn)
confusion.fit(x_train, y_train)
confusion.score(x_test, y_test)
confusion.poof()

#Obtener mejor K
k_range = range(1, 20)
scores = []

for k in k_range:
    knn =  KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train, y_train)
    scores.append(knn.score(x_test, y_test))
    
plt.figure()
plt.xlabel("k")
plt.ylabel('Presición')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20])




 
