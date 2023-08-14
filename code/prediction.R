# Loading the required libraries
library(corrplot)
library(ggvis)
library(caTools)
library(caret)
library(MLmetrics)
library(ROCR)
library(pROC)
library(rpart)
library(e1071)
library(randomForest)
library(xgboost)
library(tidyverse)
#detach(package:neuralnet)
### Data Loading and structure
data = read.csv("D:\\6th_winter_sem_22_23\\cse3506_eda\\project\\source_code\\data\\diabetes.csv")
head(data)
summary(data)
str(data)
sum(is.na(data))

### Correlations
correlations <- cor(data)
correlations
corrplot(correlations, method="color")

## Visualization
par(mar=c(1,1,1,1))
pairs(data, col=data$Outcome)
data %>% ggvis(~Glucose,~Insulin,fill =~Outcome) %>% layer_points()
data %>% ggvis(~BMI,~DiabetesPedigreeFunction,fill =~Outcome) %>% layer_points()
data %>% ggvis(~Age,~Pregnancies,fill =~Outcome) %>% layer_points()
chisq.test(data)

### Preparing the data
set.seed(8)
split <- sample.split(data$Outcome, SplitRatio = 0.75)
data_train <- subset(data, split == TRUE)
data_test <- subset(data, split == FALSE)

### Feature Scaling
#data_train[1:(length(data_train)-1)] = scale(data_train[1:(length(data_train)-1)])
#data_test[1:(length(data_test)-1)] = scale(data_test[1:(length(data_test)-1)])

### Decision Tree Classification
model_dt = rpart(formula = Outcome ~ .-Insulin-SkinThickness,data = data_train)
model_dt
predict_train_dt = predict(model_dt, newdata = data_train[1:(length(data_train)-1)])
predict_test_dt <- predict(model_dt, newdata = data_test[1:(length(data_test)-1)])
predict_test_dt <- round(predict_test_dt)
cf_dt=confusionMatrix(as.factor(data_test$Outcome), as.factor(predict_test_dt))
cf_dt
F1_Score(predict_test_dt,data_test$Outcome)
cf_dt$table[1,1]/sum(cf_dt$table[1,1:2]) #precision
cf_dt$table[1,1]/sum(cf_dt$table[1:2,1]) #recall
ROCRpred_dt <- prediction(predict_test_dt, data_test$Outcome)
ROCRperf_dt <- performance(ROCRpred_dt, 'tpr','fpr')
plot(ROCRperf_dt, colorize = TRUE, text.adj = c(-0.2,1.7))
roc_object_dt <- roc( data_test$Outcome, predict_test_dt)
auc(roc_object_dt)

### naive_bayes
model_nb = naiveBayes(x = data_train[c(-1,-4,-9)], y = data_train$Outcome)
model_nb
predict_train_nb = predict(model_nb, newdata = data_train[1:(length(data_train)-1)])
predict_test_nb <- predict(model_nb, newdata = data_test[1:(length(data_test)-1)])
cf_nb=confusionMatrix(as.factor(data_test$Outcome), as.factor(predict_test_nb))
cf_nb
F1_Score(predict_test_nb,data_test$Outcome)
cf_nb$table[1,1]/sum(cf_nb$table[1,1:2]) #precision
cf_nb$table[1,1]/sum(cf_nb$table[1:2,1]) #recall
ROCRpred_nb <- prediction(as.numeric(predict_test_nb), data_test$Outcome)
ROCRperf_nb <- performance(ROCRpred_nb, 'tpr','fpr')
plot(ROCRperf_nb, colorize = TRUE, text.adj = c(-0.2,1.7))
roc_object_nb <- roc( data_test$Outcome, as.numeric(predict_test_nb))
auc(roc_object_nb)

### random_forest
model_rf = randomForest(x = data_train[c(-1,-4,-9)], y = data_train$Outcome, ntree = 500)
model_rf
predict_train_rf = predict(model_rf, newdata = data_train[1:(length(data_train)-1)])
predict_test_rf <- predict(model_rf, newdata = data_test[1:(length(data_test)-1)])
predict_test_rf<-round(predict_test_rf)
cf_rf=confusionMatrix(as.factor(data_test$Outcome), as.factor(predict_test_rf))
cf_rf
F1_Score(predict_test_rf,data_test$Outcome)
cf_rf$table[1,1]/sum(cf_rf$table[1,1:2]) #precision
cf_rf$table[1,1]/sum(cf_rf$table[1:2,1]) #recall
predict_train_rf<-as.numeric(predict_train_rf)
predict_test_rf<-as.numeric(predict_test_rf)
ROCRpred_rf <- prediction(predict_test_rf, data_test$Outcome)
ROCRperf_rf <- performance(ROCRpred_rf, 'tpr','fpr')
plot(ROCRperf_rf, colorize = TRUE, text.adj = c(-0.2,1.7))
roc_object_rf <- roc( data_test$Outcome, predict_test_rf)
auc( roc_object_rf )

### kernal svm
model_svm = svm(formula = Outcome ~ .-SkinThickness -Pregnancies ,
             data = data_train,
             type = 'C-classification',
             kernel = 'radial')
model_svm
predict_train_svm = predict(model_svm, newdata = data_train[1:(length(data_train)-1)])
predict_test_svm <- predict(model_svm, newdata = data_test[1:(length(data_test)-1)])
cf_svm = confusionMatrix(as.factor(data_test$Outcome), as.factor(predict_test_svm))
cf_svm
F1_Score(predict_test_svm,data_test$Outcome)
cf_svm$table[1,1]/sum(cf_svm$table[1,1:2]) #precision
cf_svm$table[1,1]/sum(cf_svm$table[1:2,1]) #recall
predict_train_svm<-as.numeric(predict_train_svm)
predict_test_svm<-as.numeric(predict_test_svm)
ROCRpred_svm <- prediction(predict_test_svm, data_test$Outcome)
ROCRperf_svm <- performance(ROCRpred_svm, 'tpr','fpr')
plot(ROCRperf_svm, colorize = TRUE, text.adj = c(-0.2,1.7))
roc_object_svm <- roc( data_test$Outcome, predict_test_svm)
auc( roc_object_svm )

### Logistic regression
model_lr <- glm (formula = Outcome ~.-Pregnancies -SkinThickness , data = data_train, family = binomial)
summary(model_lr)
predict_train_lr <- predict(model_lr, type = 'response')
predict_test_lr <- predict(model_lr, newdata = data_test, type = 'response')
cf_lr = confusionMatrix(as.factor(data_test$Outcome), as.factor(round(predict_test_lr)))
cf_lr
F1_Score(as.factor(round(predict_test_lr)),data_test$Outcome)
cf_lr$table[1,1]/sum(cf_lr$table[1,1:2]) #precision
cf_lr$table[1,1]/sum(cf_lr$table[1:2,1]) #recall
ROCRpred <- prediction(predict_test_lr, data_test$Outcome)
ROCRperf <- performance(ROCRpred, 'tpr','fpr')
plot(ROCRperf, colorize = TRUE, text.adj = c(-0.2,1.7))
roc_object_lr <- roc( data_test$Outcome, predict_test_lr)
auc( roc_object_lr )

### XGBoost
model_xgb <- xgboost(data = as.matrix(data_train[c(-1,-4,-9)]), label = data_train$Outcome, nrounds = 10)
summary(model_xgb)
predict_train_xgb <- predict(model_xgb, newdata = as.matrix(data_train[c(-1,-4,-9)]))
predict_test_xgb <- predict(model_xgb, newdata = as.matrix(data_test[c(-1,-4,-9)]))
cf_xgb = confusionMatrix(as.factor(data_test$Outcome), as.factor(round(predict_test_xgb)))
cf_xgb
F1_Score(as.factor(round(predict_test_xgb)),data_test$Outcome)
cf_xgb$table[1,1]/sum(cf_lr$table[1,1:2]) #precision
cf_xgb$table[1,1]/sum(cf_lr$table[1:2,1]) #recall
ROCRpred <- prediction(predict_test_xgb, data_test$Outcome)
ROCRperf <- performance(ROCRpred, 'tpr','fpr')
plot(ROCRperf, colorize = TRUE, text.adj = c(-0.2,1.7))
roc_object_lr <- roc( data_test$Outcome, predict_test_lr)
auc( roc_object_lr )

library(neuralnet)
### Neural Network
model_nn = neuralnet(
  Outcome ~.-Pregnancies -SkinThickness,
  data=data_train,
  hidden=c(4,2),
  act.fct="tanh"
)
plot(model_nn,rep = "best")
predict_test_nn <- predict(model_nn, data_test)
F1_Score(as.factor(round(predict_test_xgb)),data_test$Outcome)
cf_xgb$table[1,1]/sum(cf_lr$table[1,1:2]) #precision
cf_xgb$table[1,1]/sum(cf_lr$table[1:2,1]) #recall
detach(package:neuralnet)
ROCRpred <- prediction(predict_test_nn, data_test$Outcome)
ROCRperf <- performance(ROCRpred, 'tpr','fpr')
plot(ROCRperf, colorize = TRUE, text.adj = c(-0.2,1.7))
roc_object_lr <- roc( data_test$Outcome, predict_test_lr)
auc( roc_object_lr )
