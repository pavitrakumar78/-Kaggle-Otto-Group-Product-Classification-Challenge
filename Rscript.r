setwd("E:/Kaggle/Otto product classification")
library(h2o)
library(caret)
library(mlbench)
library(ggplot2)
library(reshape2)
library(DEEPR)

#max cpus
localH2O = h2o.init(ip = "localhost", port = 54321, startH2O = TRUE,nthreads=-1)
#only 2 cpus
localH2O = h2o.init(ip = "localhost", port = 54321, startH2O = TRUE)

train = read.csv("train.csv", header = TRUE, stringsAsFactors = FALSE)

#http://en.wikipedia.org/wiki/Anscombe_transform

for(i in 2:94){
  train[,i] <- as.numeric(train[,i])
  train[,i] <- sqrt(train[,i]+(3/8))
}

test = read.csv("test.csv", header = TRUE, stringsAsFactors = FALSE)

for(i in 2:94){
  test[,i] <- as.numeric(test[,i])
  test[,i] <- sqrt(test[,i]+(3/8))
}

test$id <- NULL
train$id = NULL
train$target <- as.factor(train$target)

data_h2o <- as.h2o(localH2O, train, key = 'train')
test_h2o <- as.h2o(localH2O, test, key = 'test')

pos.train <- sample(1:nrow(train),nrow(train)*0.80)
data_train <- data_h2o[pos.train,]
data_test <- data_h2o[-pos.train,]

submission <- read.csv("sampleSubmission (1).csv")
submission[,2:10] <- 0


#------------------running model-------------------

#model@model$params$epochs

for(i in 1:10){
  print(i)
  model <- h2o.deeplearning(x=1:93, y=94,
                            data=data_h2o,
                            classification=T,
                            activation="RectifierWithDropout",
                            hidden=c(1024,512,256),
                            hidden_dropout_ratio=c(0.5,0.5,0.5),
                            input_dropout_ratio=0.05,
                            epochs=1500,
                            l1=1e-5,
                            l2=1e-5,
                            rho=0.99,
                            epsilon=1e-8,
                            train_samples_per_iteration=750,
                            max_w2=10,
                            seed=1)
  submission[,2:10] <- submission[,2:10] + as.data.frame(h2o.predict(model,test_h2o))[,2:10]
  print(i)
  write.csv(submission,file="submission.csv",row.names=FALSE) 
} 
