## si alguna librería falla, recordar instalarla. Por ejemplo: install.packages("rpart.plot")

library(rpart)
library(rpart.plot)

v <- iris$Species


set.seed(23)

# preparar para dividir en 60% de entrenamiento y 50% de test
iris[, 'train'] <- ifelse(runif(nrow(iris)) < 0.75, 1, 0)

trainSet <- iris[iris$train == 1,]
testSet <- iris[iris$train == 0, ]

trainColNum <- grep('train', names(trainSet))

trainSet <- trainSet[, -trainColNum]
testSet <- testSet[, -trainColNum]

treeFit <- rpart(Species~.,data=trainSet,method = 'class')
print(treeFit)

rpart.plot(treeFit, box.col=c("red", "green"))

Prediction1 <- predict(treeFit,newdata=testSet[-5],type = 'class')


## matriz de confusión
library(caret)

confusionMatrix(Prediction1,testSet$Species)

