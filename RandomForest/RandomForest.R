install.packages("randomForest")
library(caret)
library(randomForest)

datos <- read.csv(file.choose(), sep = ",", header = T)


modelo <- randomForest(class ~ ., data = datos, ntree = 100)



