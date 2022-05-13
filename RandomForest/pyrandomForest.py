import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer 
from yellowbrick.classifier import ConfusionMatrix 

hepatitis = pd.read_csv("HepatitisCdata.csv")

x = hepatitis.iloc[:,2:12].values
y = hepatitis.iloc[:,1].values

labelencoder = LabelEncoder()

#preprocesamiento para NA

imputer= SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:,2:])
x[:,2:]=imputer.transform(x[:,2:])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=0)

#entrenamiento
randomforest = RandomForestClassifier(n_estimators=100, random_state=0)
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_test)

#Matriz de confución
confusion1=ConfusionMatrix(randomforest)
confusion1.fit(x_train, y_train)
confusion1.score(x_test, y_test)
confusion1.poof()

#Evaluacion del algoritmo
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

hepatitis.head()









