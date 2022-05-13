library(caret)
library(randomForest)

hepatitis <- read.csv("E:/Sergio Ramos/UDLA/Semestre 9/Inteligencia Computacional/Machine Learning/RandomForest/HepatitisCdata.csv")
str(hepatitis)
hepatitis$class <- factor(hepatitis$class)

