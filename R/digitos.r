###################
## Clasificación de grafía de dígitos a partir de su representación como píxeles utilizando SVM
## este script debe ejecutarse en la misma carpeta que los ficheros "mnist_train.csv" y "mnist_test.csv"

############################################## Loading libraries ###########################################

library(ggplot2)
library(kernlab)
library(caret)
library(caTools)
library(gridExtra)



### 1) Carga de datos

mnist_train <- read.csv("mnist_train.csv", stringsAsFactors = F, header = F)
mnist_test <- read.csv("mnist_test.csv", stringsAsFactors = F, header = F)


# ponemos a la columna con la etiqueta (el dígito al que corresponde la imagen) el nombre label
names(mnist_test)[1] <- "label"
names(mnist_train)[1] <- "label"



# 2 Preparación

# 2.1 Convertimos label a tipo factor

mnist_train$label <- factor(mnist_train$label)
summary(mnist_train$label)

mnist_test$label <- factor(mnist_test$label)


# 2.2 El conjunto de entrenamiento es demasiado grande, tomamos una muestra


set.seed(23)
sample_indices <- sample(1: nrow(mnist_train), 10000) # nos quedamos con 10000 valores
train <- mnist_train[sample_indices, ]

# 2.3 Escalado. Tomamos 255 como valor máximo y 0 como mínimo y aplicamos MinMax, tanto al train como al test

train[ , 2:ncol(train)] <- train[ , 2:ncol(train)]/255
test <- cbind(label = mnist_test[ ,1], mnist_test[ , 2:ncol(mnist_test)]/255)


### Entrenamiento y evaluación

model_linear <- ksvm(label ~ ., data = train, scaled = FALSE, kernel = "vanilladot", C = 10)
print(model_linear) 

eval <- predict(model_linear, newdata = test, type = "response")
confusionMatrix(eval, test$label) 


##validación cruzada para optimizar  C

grid_linear <- expand.grid(C= c(0.001, 0.1 ,1 ,10 ,100)) 

# ojo, puede tardar
fit.linear <- train(label ~ ., data = train, metric = "Accuracy", method = "svmLinear",
                    tuneGrid = grid_linear, preProcess = NULL,
                    trControl = trainControl(method = "cv", number = 5))


print(fit.linear) 
plot(fit.linear)


eval_cv_linear <- predict(fit.linear, newdata = test)
confusionMatrix(eval_cv_linear, test$label)

