df <- data(iris) ## cargar datos
head(iris) 
nor <-function(x) { (x -min(x))/(max(x)-min(x))   }
library(class)
mayor_acierto = 0

accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}

for(k in seq(2,16)){
    suma = 0
    N<- 100
    for (i in seq(1,N)) {
        ran <- sample(1:nrow(iris), 0.6 * nrow(iris)) 
        iris_norm <- as.data.frame(lapply(iris[,c(1,2,3,4)], nor))
        iris_train <- iris_norm[ran,] 
        iris_test <- iris_norm[-ran,]
        iris_target_category <- iris[ran,5]
        iris_test_category <- iris[-ran,5]
        pr <- knn(iris_train,iris_test,cl=iris_target_category,k=k)
        tab <- table(pr,iris_test_category)
        suma <- suma + accuracy(tab)
    }
    aciertos <- suma/N
    print(paste(k,aciertos))
    if (aciertos>mayor_acierto) {
        mayor_acierto = aciertos
        mejor_k = k
    }
}
print(paste(mejor_k,mayor_acierto))

