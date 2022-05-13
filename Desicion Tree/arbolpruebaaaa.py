import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix 
from sklearn import tree

base=pd.read_csv('HepatitisCdata.csv')
X=base.iloc[:,2:12].values
#print(X)
y=base.iloc[:,1].values
#print(y)
labelencoder=LabelEncoder()
"""

X[:,0]=labelencoder.fit_transform(X[:,0])
X[:,1]=labelencoder.fit_transform(X[:,1])
X[:,2]=labelencoder.fit_transform(X[:,2])
"""
X[:,1]=labelencoder.fit_transform(X[:,1])
#print(X[:,1])
print('---------------------------------------------')
imputer= SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:,2:])
X[:,2:]=imputer.transform(X[:,2:])
print(X[:,2:])





X_entrenar, X_probar, y_entrenar, y_probar=train_test_split (X,y,test_size = 0.7,random_state=0) #escogemos 30% de los datos para pruebas

#modelo 1
modelo1=DecisionTreeClassifier(criterion='entropy')
modelo1.fit(X_entrenar,y_entrenar)
export_graphviz(modelo1, out_file= 'modelo1.dot')
predicciones1=modelo1.predict(X_probar)
print(accuracy_score(y_probar,predicciones1))

#Generar Matrix de Confusion
confusion1=ConfusionMatrix(modelo1)
confusion1.fit(X_entrenar,y_entrenar)
confusion1.score(X_probar,y_probar)
confusion1.poof()

#evaluacion 


#modelo 2

modelo2=DecisionTreeClassifier(criterion='entropy',min_samples_split=100)
modelo2.fit(X_entrenar,y_entrenar)
export_graphviz(modelo2,out_file='modelo2.dot')
predicciones2=modelo2.predict(X_probar)
accuracy_score(y_probar,predicciones2)


confusion2 = ConfusionMatrix(modelo2)
confusion2.fit(X_entrenar,y_entrenar)
confusion2.score(X_probar,y_probar)
confusion2.poof()

#modelo3

modelo3=DecisionTreeClassifier(criterion='entropy',min_samples_leaf=5,min_samples_split=20)
modelo3.fit(X_entrenar,y_entrenar)
export_graphviz(modelo3, out_file='modelo3.dot')
predicciones3 =  modelo3.predict(X_probar)
accuracy_score(y_probar, predicciones3)


confusion3=ConfusionMatrix(modelo3)
confusion3.fit(X_entrenar,y_entrenar)
confusion3.score(X_probar,y_probar)
confusion3.poof()

fig1, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (8,4), dpi=500)
tree.plot_tree(modelo1);
fig1.savefig('tree1.png')
plt.show()





