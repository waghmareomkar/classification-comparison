#Reading dataset
hdata <- read.csv(file="/home/rishabh/Downloads/DMW/heartdisease.csv",header=TRUE,sep=",")
nrow(hdata)
ncol(hdata)
colnames(hdata)
summary(hdata)
hdata$num[hdata$num > 1 ]  <- 1



# Technique 1: Linear regression
plot(hdata$age,hdata$num, main = "NUM PLOTTED AGAINST AGE", xlab = "AGE", ylab = "NUM")
train_hdata = hdata[1:212,]
test_hdata = hdata[213:303,]
dim(train_hdata)
dim(test_hdata)

library(caTools)
regressor=lm(formula = num~age, data=train_hdata)
#predicting the test set result using regressor
hd_age_predict=predict(regressor, newdata=test_hdata)
hd_age_predict
# As the result is not whole number, rounding the result
round_age=hd_age_predict
rage=round(round_age)
table(rage)
table(test_hdata$num)
# Displaying the accuracy using confusion Matrix
library(e1071)
library(caret)
df=confusionMatrix(factor(rage),factor(test_hdata$num))
df

# Technique 2: Multiple regression prediction using multiple linear regression
regressor=lm(formula =
               num~age+sex+cp+trestbps+chol+fbs+restecg+thalach+exang+oldpeak+slope,
             data=train_hdata)
# predicting the test set result
hd_age_predict=predict(regressor, newdata=test_hdata)
# As the result is not whole number, rounding the result
round_age=hd_age_predict
rage=round(round_age)
library(e1071)
library(caret)
df=confusionMatrix(rage,test_hdata$num)


# Technique 3: k-Nearest Neighbour claasifer
# Prediction using KNN
# Use data transformation technique such as scaling and normalization for normalizing dataset
# Writting the function for normalizing the values of all variables
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x))) }
h1data<-hdata
h1data_n <- as.data.frame(lapply(h1data[1:11], normalize))
traink_hdata=h1data_n[1:212,]
testk_hdata=h1data_n[213:303,]

library(class)
h1data_train_labels <- hdata[1:212, 14]
h1data_test_labels <- hdata[213:303, 14]
# Applying Knn function on dataset
h1data_test_pred <- knn(train = traink_hdata, test = testk_hdata,cl =
                          h1data_train_labels, k=17)
# Another method of getting confusion matrix using CrossTable
library(gmodels)
CrossTable(x=h1data_test_labels,y=h1data_test_pred,prop.chisq = FALSE)
table(h1data_test_labels,h1data_test_pred)

library(e1071)
library(caret)
df=confusionMatrix(factor(h1data_test_labels),factor(h1data_test_pred))


# Technique 4: Naive bayes classifier
# Naive bayes classifyer needs catagorical data for prediction
# Preparing data for Naive Bayesh1data<-hdata
h1data$age=factor(h1data$age)
h1data$sex=factor(h1data$sex)
h1data$cp=factor(h1data$cp)
h1data$trestbps=factor(h1data$trestbps)
h1data$chol=factor(h1data$chol)
h1data$fbs=factor(h1data$fbs)
h1data$restecg=factor(h1data$restecg)
h1data$thalach=factor(h1data$thalach)
h1data$exang=factor$exang
h1data$exang=factor(h1data$exang)
h1data$oldpeak=factor(h1data$oldpeak)
h1data$slope=factor(h1data$slope)
h1data$num=factor(h1data$num)
# Dividing dataset into training and testing
trainnb_hdata=h1data[1:212,]
testnb_hdata=h1data[213:303,-14]
# Applying Naive Bayes claasifier on dataset
library(e1071)
classifier <- naiveBayes(num
                         ~age+sex+cp+trestbps+chol+fbs+restecg+thalach+exang+oldpeak+slope,trainnb_hdata)
prediction <- predict(classifier, testnb_hdata ,type="class")
prediction

table(prediction, h1data[213:303,14])

# Displaying the accuracy using confusion Matrix
library(e1071)
library(caret)
df=confusionMatrix(factor(h1data[213:303,14]),factor(prediction))