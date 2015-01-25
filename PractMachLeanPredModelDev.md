Practical Machine Learning Prediction Model Development
========================================================

This document will review the development and testing of my prediction model created for the Practical Machine Learning Coursera Course. The prediction model attempts to classify a weight lifting technique as correct or one of four common mistakes using data gathered by sensors attached to body parts and equipment. The model development is described in the following six steps:

1. Load and Clean Data
2. Partition Data
3. Select Features
4. Train Models
5. Estimate Out of Sample Error
6. Generate Submissions and Validate

## 1. Load and Clean Data

The training and testing data for this assignment were created by [Groupware@LES](groupware.les.inf.puc-rio.br); links to the data and their papers are available at their website, and we are using their [Weight Lifting Exercises Dataset](http://groupware.les.inf.puc-rio.br/har#wle_paper_section).

The specific data were obtained from the class provided links, [training](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv) and [testing](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv), and saved into the working directory. Blank observations are read as "NA" to make later cleaning easier. The dimensions of each data set are provided.


```r
fulltraining <- read.csv("pml-training.csv", stringsAsFactors = FALSE, na.strings = c("", "NA"))
validation <- read.csv("pml-testing.csv", stringsAsFactors = FALSE, na.strings = c("", "NA"))
dim(fulltraining)
```

```
## [1] 19622   160
```

```r
dim(validation)
```

```
## [1]  20 160
```

## 2. Partition Data

The full training data set needs to be separated into a training set and testing set because the testing set provided will be used for validation and grading the model prediction. So, before doing anything else to the data, it is partitioned. The "classe" variable is the outcome I am trying to predict. From here on out the data partitioned from the full training data will be referred to as the testing and training data respectively; the full testing data will be referred to as the validation data.


```r
suppressMessages(library(caret))
set.seed(1234)
inTrain <- createDataPartition(y=fulltraining$classe, p = 0.75, list = FALSE)
training <- fulltraining[inTrain, ]
testing <- fulltraining[-inTrain, ]
dim(training)
```

```
## [1] 14718   160
```

```r
dim(testing)
```

```
## [1] 4904  160
```


## 3. Select Features

The testing and training data sets contain 160 variables, but many of those are almost completely empty of useful data, so the data is condensed down to only those variables that contain less than 5 blank or NA observations. Other variables are not useful predictors. The remaining variables are retained as possible features.


```r
trainings <- training[ ,colSums(is.na(training)) < 5]
testings <- testing[ ,colSums(is.na(training)) <5]
exclude <- names(trainings) %in% c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "new_window", "num_window")
trainingsmall <- trainings[ , !exclude]
testingsmall <- testings[ , !exclude]
dim(trainingsmall)
```

```
## [1] 14718    53
```

```r
dim(testingsmall)
```

```
## [1] 4904   53
```


## 4. Train Models

Cross validation, using the K-Folds method stipulating 5 folds, is included directly in the training for each of the models.


```r
XVCtrl <- trainControl(method = "cv", number = 5)
```

The first model trained was a tree model. The "classe" variable needs to be set as a factor to avoid errors while training using the "rpart" method.


```r
suppressMessages(library(rpart))
set.seed(12340)
trainingsmall$classe <- as.factor(trainingsmall$classe)
testingsmall$classe <- as.factor(testingsmall$classe)
modFitTree <- train(classe ~ ., method = "rpart", data = trainingsmall, trControl = XVCtrl)
TreePredIS <- predict(modFitTree, newdata = trainingsmall)
TreeConfIS <- confusionMatrix(TreePredIS, trainingsmall$classe)
TreeISAK <- TreeConfIS$overall[1:2]
```

The tree model in sample accuracy and kappa (note kappa is included for error comparisons in step #5) are:


```
## Accuracy    Kappa 
##    0.495    0.341
```

The accuracy and kappa of this tree model are too low, so next a random forest model is trained.


```r
suppressMessages(library(randomForest))
set.seed(12341)
modFitRF <- train(classe ~ ., method = "rf", data = trainingsmall, trControl = XVCtrl)
RFPredIS <- predict(modFitRF, newdata = trainingsmall)
RFConfIS <- confusionMatrix(RFPredIS, trainingsmall$classe)
RFISAK <- RFConfIS$overall[1:2]
```

The random forest model in sample accuracy and kappa are:


```
## Accuracy    Kappa 
##        1        1
```

The accuracy and kappa of the random forest model are much higher, and this is the method used to generate submission answers.


## 5. Estimate Out of Sample Error

Because we are predicting categorical items, kappa is the selected measure of error for these models.  Kappa results range from -1 to 1, and higher kappa values indicate lower error. I would expect to see high out of sample error (i.e., low kappa) for the tree model given its low accuracy and low out of sample error (i.e., high kappa) from the random forest model given its high accuracy. Because cross validation is built into the model training, the out of sample error should be higher (i.e., lower kappa) than the observed in sample error, but not by much.

Both the tree and random forest models are used to generate predictions on the testing data. The comparison of these predictions and the true values are used to calculate kappa values, which is the metric I am using for error. The out of sample kappa values are compared with the in sample kappa values generated in step #4. The out of sample kappa values can also be compared to the kappa values generated when the model is validated in step #6.

Here is the comparison between predicted and true classes for the tree model using the testing data:


```r
suppressMessages(library(rpart))
TreePred <- predict(modFitTree, newdata = testingsmall)
TreeTrue <- testingsmall$classe
TreeTable <- table(TreePred, TreeTrue)
TreeConf <- confusionMatrix(TreePred, testingsmall$classe)
```

#### Figure 1. Tree Model Comparison Table

```
##         TreeTrue
## TreePred    A    B    C    D    E
##        A 1275  390  416  356  134
##        B   28  340   31  141  131
##        C   88  219  408  307  230
##        D    0    0    0    0    0
##        E    4    0    0    0  406
```

The tree model accuracy and kappa for the out of sample test data are:


```
## Accuracy    Kappa 
##    0.495    0.340
```

This out of sample kappa value is lower than, but similar to, the in sample kappa generated for the tree model in step #4.

Here is the same comparison for the random forest model:


```r
suppressMessages(library(randomForest))
RFPred <- predict(modFitRF, newdata = testingsmall)
RFTrue <- testingsmall$classe
RFTable <- table(RFPred, RFTrue)
RFConf <- confusionMatrix(RFPred, testingsmall$classe)
```

#### Figure 2. Random Forest Model Comparison Table

```
##       RFTrue
## RFPred    A    B    C    D    E
##      A 1395    3    0    0    0
##      B    0  946   15    0    0
##      C    0    0  839   10    0
##      D    0    0    1  794    0
##      E    0    0    0    0  901
```

The random forest model accuracy and kappa for the out of sample test data are:

```
## Accuracy    Kappa 
##    0.994    0.993
```

Again this out of sample kappa value is lower than, but similar to, the in sample kappa generated from the random forest model in step #4.

The confusion matrix was a helpful tool for digging into each model, so I have included the random forest confusion matrix for the predictions on the testing data. 

#### Figure 3. Random Forest Confusion Matrix

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    3    0    0    0
##          B    0  946   15    0    0
##          C    0    0  839   10    0
##          D    0    0    1  794    0
##          E    0    0    0    0  901
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9941         
##                  95% CI : (0.9915, 0.996)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9925         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9968   0.9813   0.9876   1.0000
## Specificity            0.9991   0.9962   0.9975   0.9998   1.0000
## Pos Pred Value         0.9979   0.9844   0.9882   0.9987   1.0000
## Neg Pred Value         1.0000   0.9992   0.9961   0.9976   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1929   0.1711   0.1619   0.1837
## Detection Prevalence   0.2851   0.1960   0.1731   0.1621   0.1837
## Balanced Accuracy      0.9996   0.9965   0.9894   0.9937   1.0000
```


## 6. Generate Submissions and Validate

To create a set of submission files for the assignment, I used the predict function to 
generate the vector of predictions and the instructor-suggested code to create the set of text files.


```r
suppressMessages(library(randomForest))
RFPredVal <- predict(modFitRF, newdata = validation)
TreePredVal <- predict(modFitTree, newdata = validation)
CompTable <- table(RFPredVal, TreePredVal)
```

#### Figure 4. Random Forest vs. Tree Model Predictions on Validation Data


```
##          TreePredVal
## RFPredVal A B C D E
##         A 7 0 0 0 0
##         B 3 0 5 0 0
##         C 0 0 1 0 0
##         D 0 0 1 0 0
##         E 1 0 2 0 0
```

I used the random forest model to generate my submissions and it correctly predicted 20 out of 20. Using these correct answers I created a new "classe" variable in the validation data to calculate accuracy and kappa values for the validation runs of both the tree and random forest models.


```r
validationKA <- validation
validationKA$classe <- RFPredVal
RFPredValKA <- predict(modFitRF, newdata = validationKA)
RFPKA <- confusionMatrix(RFPredValKA, validationKA$classe)
RandomForestModelValidation <- RFPKA$overall[1:2]
TreePredValKA <- predict(modFitTree, newdata = validationKA)
TreePKA <- confusionMatrix(TreePredValKA, validationKA$classe)
TreeModelValidation <- TreePKA$overall[1:2]
ValKA <- rbind(TreeModelValidation, RandomForestModelValidation)
```

The accuracy and kappa values for both models on the validation data are:


```
##                             Accuracy Kappa
## TreeModelValidation              0.4 0.236
## RandomForestModelValidation      1.0 1.000
```

These kappa values are similar to those observed from the out of sample data in step #5 for each model.

The following code generates the files I used to submit my predictions; it is supplied for information but is not evaluated in this document.


```r
RFSubmit <- as.character(predict(modFitRF, newdata = validation))
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)}}
pml_write_files(RFSubmit)
```
