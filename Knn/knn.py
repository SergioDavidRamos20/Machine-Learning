import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from tkinter.filedialog import askopenfilename
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer 



hepatitisknn = pd.read_csv("HepatitisCdata.csv")


x = hepatitisknn.iloc[:,2:].values
y = hepatitisknn.iloc[:,1].values


#preprocesamiento NA

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:,2:])
x[:,2:]=imputer.transform(x[:,2:])

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x, y)

KNeighborsClassifier(n_neighbors=3)

prediction = knn.predict()

    


 
