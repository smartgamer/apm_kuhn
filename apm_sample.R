
#Applied Predictive Modeling by Kuhn

library(caret)
data(segmentationData)
dim(segmentationData)
str(segmentationData[, 1:9])

#The authors designated a training set (n = 1009) and a test set (n = 1010). We’ll use these:
## remove the cell identifier
segmentationData$Cell <- NULL
seg_train <- subset(segmentationData, Case == "Train")
seg_test <- subset(segmentationData, Case == "Test")
seg_train$Case <- NULL
seg_test$Case <- NULL

### Below is not used:
# If you wanted to do a random 50/50 split of the data, there is a function in caret that can be used:
## NOT executed...
## make a balanced random split
in_train <- createDataPartition(segmentationData$Class, p = 0.5, list = FALSE)
## �in_train� is set of row indices that are selected to go
## into the training set
train_data <- segmentationData[ in_train,]
test_data <- segmentationData[-in_train,]
###end

#Over–fitting occurs when a model inappropriately picks up on trends in the training set that do not generalize to new samples.

#Characterizing Over–Fitting Using the Training Set
# Examples are cross–validation (in many varieties) and the bootstrap. These procedures repeated split the training data into subsets used formodeling and performance evaluation.

#K–Fold Cross–Validation

##
# Typical Process for Model Building
# Now that we know how to evaluate models on the training set, we can try different techniques (including pre-processing) and try to optimize model performance.
# Performance might not be the only consideration. Others might include:
# . simplicty of prediction
# . redusing the number of predictors (aka features) in the model to
# . reduce cost or complexity
# . smoothness of the prediction equation
# . robustness of the solution
# Once we have 1-2 candidate models, we can evaluate the results on the test set.
##

# Linear Discriminant Analysis
library(MASS)
lda_fit <- lda(Class ~ ., data = seg_train, tol = 1.0e-15)
predict(lda_fit, newdata = seg_train[1:3,])

##Estimating Performance For Classification
#For classification models:
# .overall accuracy can be used, but this may be problematic when the classes are not balanced.
# .the Kappa statistic takes into account the expected error rate:
  k=(O-E)/(1-E_)
   # where O is the observed accuracy and E is the expected accuracy under chance agreement
# . For 2–class models, Receiver Operating Characteristic (ROC) curves can be used to characterize model performance (more later)

# A “ confusion matrix” is a cross–tabulation of the observed and predicted classes
# R functions for confusion matrices are in the e1071 package (the classAgreement function), the caret package ( confusionMatrix ), the mda ( confusion ) and others.
# ROC curve functions are found in the pROC package ( roc ) ROCR package   ( performance ), the verification package ( roc.area ) and others.
# We’ll use the confusionMatrix function and the pROC package later.

##Creating the ROC Curve
  # We’ll use the pROC function to compute and plot the ROC curve.
  # First, we need a set of predicted class probabilities and then we use the
  # roc function

lda_test_pred <- predict(lda_fit, newdata = seg_test)
library(pROC)
lda_roc <- roc(response = seg_test$Class,
                 predictor = lda_test_pred$posterior[, "PS"],
                 ## we need to tell the function that the _first_ level
                 ## is our event of interest
                 levels = rev(levels(seg_test$Class)))
lda_roc
# plot(exRoc print.thres = .5)
confusionMatrix(data = lda_test_pred$class, reference = seg_test$Class)

##Model Function Consistency
## only one way here:
rpart(y ~ ., data = dat)
## and both ways here:
lda(y ~ ., data = dat)
lda(x = predictors, y = outcome)


## setting the seed before calling �train� controls the resamples
set.seed(20792)
lda_mod <- train(Class ~ ., data = seg_train, method = "lda")

##The train Function
#To use five repeats of 10–fold cross–validation, we would use
ctrl <- trainControl(method = "repeatedcv", repeats = 5)
set.seed(20792)
lda_mod <- train(Class ~ ., data = seg_train, method = "lda", 
                     trControl = ctrl)
#By classification, the default performance metrics that are computed are accuracy and the kappa statistic. For regression, they are RMSE and R 2 . Instead, let’s measure the area under the ROC curve, sensitivity, and specificity.
twoClassSummary(fakeData)

ctrl <- trainControl(method = "repeatedcv", repeats = 5,
                       classProbs = TRUE,
                       summaryFunction = twoClassSummary)
set.seed(20792)
lda_mod <- train(Class ~ ., data = seg_train,
                     method = "lda",
                     ## Add the metric argument
                     trControl = ctrl, metric = "ROC",
                     ## Also pass in options to �lda� using �...�                 +
                   tol = 1.0e-15)

##Digression – Parallel Processing

#foreach and caret

library(doMC)
# on unix, linux or OS X
library(doParallel) # windows and others
registerDoMC(cores = 2)
lda_mod

#Other Models
#Tuning the Number of Neighbors
## The same resamples are used
set.seed(20792)
knn_mod <- train(Class ~ ., data = seg_train,
                     method = "knn",
                     trControl = ctrl,
                     ## tuning parameter values to evaluate
                     tuneGrid = data.frame(k = seq(1, 25, by = 2)),
                     preProc = c("center", "scale"),
                     metric = "ROC")
knn_mod
ggplot(knn_mod)

##Predicting New Samples
## to get the classes:
predict(knn_mod, newdata = head(seg_test))

## We choose �prob� to get class probabilities:
predict(knn_mod, newdata = head(seg_test), type = "prob")

##Comparing Models
## The same resamples are used
set.seed(20792)
knn_yj_mod <- train(Class ~ ., data = seg_train,
                    method = "knn",
                    trControl = ctrl,
                    tuneGrid = data.frame(k = seq(1, 25, by = 2)),
                    preProc = c("center", "scale", "YeoJohnson"),
                    metric = "ROC")
## What was the best area under the ROC curve?
getTrainPerf(knn_yj_mod)

## Comparing Models
## Conduct o a paired t-test on the resampled AUC values to control for
## resample-to-resample variability:
# compare_models(knn_yj_mod, knn_mod, metric = "ROC")
# ## Yes, but not by much

##Some Notes on Tuning
# . train attempts to fit as few models as possible. In quite a few places, we use a “sub–model trick” to get predictions for some sub–models without refitting
# . random search, where the tuning parameters are randomly selected, can be used when you want to try a larger range of parameters 
# . adaptive resampling (aka racing) can be used to reduce the time to tuning the models based on interim analyses to discard candidate sub–models
# the resamples class can be used to visualize and compare models on a larger scale than compare models function
# train allows for much more flexibility to customize the tuning process




