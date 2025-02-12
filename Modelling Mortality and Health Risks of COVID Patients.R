COVID <- read.csv("C:\\Users\\Leon\\Downloads\\CovidHospDataBrasil.csv")
COVID.kaggle <- read.csv("C:\\Users\\Leon\\Downloads\\CovidHospDataBrasil_test.csv")

# Installing required packages
library(glmnet)
library(MASS)
library(bestglm)
library(car)
library(pROC)
library(tidyverse)
library(MLmetrics)
library(mgcv)
library(caret)
library(DiagrammeR)
library(randomForest)

# Cleaning data & creating new variables
COVID['dateHosp'] <- lapply(COVID['dateHosp'], as.Date)

admission.conditions <- c('vaccine', 'fever', 'cough', 'sorethroat', 'dyspnoea', 'oxygensat', 
                          'diarrhea', 'vomit', 'hematologic', 'downsyn', 'asthma',
                          'diabetes', 'neurological', 'pneumopathy', 'obesity')
admission.conditions.symptoms <- c('fever', 'cough', 'sorethroat', 'dyspnoea', 'oxygensat', 
                                   'diarrhea', 'vomit')
admission.conditions.symptoms.common <- c('fever', 'cough', 'sorethroat', 'dyspnoea', 'oxygensat')
admission.conditions.diseases <- c('hematologic', 'downsyn', 'asthma',
                                   'diabetes', 'neurological', 'pneumopathy', 'obesity')

# Making binary variables for vaccine, icu and deaths
catergorical.cols <- c("vaccine", "fever", "cough", "sorethroat", "dyspnoea", "oxygensat", 
                       "diarrhea", "vomit", "hematologic", "downsyn", "asthma", "diabetes", "neurological",
                       "pneumopathy", "obesity", "covidDeath")

COVID[catergorical.cols] <- lapply(COVID[catergorical.cols], function(x) as.numeric(as.logical(x)))

# New variables that are used for analysis in 3.3.1
COVID <- COVID %>%
  mutate(dateAdmIcu = as.Date(dateAdmIcu)) %>% 
  mutate(dateDisIcu = as.Date(dateDisIcu)) %>% 
  mutate(durationICU = difftime(as.POSIXct(dateDisIcu), as.POSIXct(dateAdmIcu), units = "days")) %>% 
  mutate(durationHosp = difftime(as.POSIXct(dateEndObs), as.POSIXct(dateHosp), units = "days")) %>% 
  mutate(age.group = cut(age, 
                         breaks = c(0, 20, 40, 60, 80, 130),
                         labels = c('0-20', '21-40', '41-60', '61-80', '80+'),
                         right = FALSE))

COVID$durationICU[is.na(COVID$durationICU)] <- 0

COVID$icu <- ifelse(COVID$icu == 'True', 1, 0)

death.conditions <- c('respdistress', 'cardio', 'hepatic', 'immuno', 'renal')

COVID[death.conditions] <- lapply(COVID[death.conditions], function(x) as.numeric(as.logical(x)))
COVID <- COVID %>% 
  mutate(preAdmCond = rowSums(select(., all_of(admission.conditions)), na.rm = TRUE)) %>% 
  mutate(deathCond = rowSums(select(., all_of(death.conditions)), na.rm = TRUE)) %>% 
  mutate(preAdmCondSymp = rowSums(select(., all_of(admission.conditions.symptoms)), na.rm = TRUE)) %>% 
  mutate(preAdmCondDis = rowSums(select(., all_of(admission.conditions.diseases)), na.rm = TRUE)) %>% 
  mutate(preAdmCondSympComm = rowSums(select(., all_of(admission.conditions.symptoms.common)), na.rm = TRUE)) 

# Now for Kaggle data set: Cleaning data & creating new variables
COVID.kaggle['dateHosp'] <- lapply(COVID.kaggle['dateHosp'], as.Date)

COVID.kaggle[admission.conditions] <- lapply(COVID.kaggle[admission.conditions], function(x) as.numeric(as.logical(x)))

COVID.kaggle <- COVID.kaggle %>%
  mutate(age.group = cut(age, 
                         breaks = c(0, 20, 40, 60, 80, 130),
                         labels = c('0-20', '21-40', '41-60', '61-80', '80+'),
                         right = FALSE))

COVID.kaggle <- COVID.kaggle %>% 
  mutate(preAdmCond = rowSums(select(., all_of(admission.conditions)), na.rm = TRUE)) %>% 
  mutate(preAdmCondSymp = rowSums(select(., all_of(admission.conditions.symptoms)), na.rm = TRUE)) %>% 
  mutate(preAdmCondDis = rowSums(select(., all_of(admission.conditions.diseases)), na.rm = TRUE)) %>% 
  mutate(preAdmCondSympComm = rowSums(select(., all_of(admission.conditions.symptoms.common)), na.rm = TRUE)) 

# First create test data set
set.seed(9)
test.index <- sample(1:nrow(COVID), 0.2 * nrow(COVID))
test.COVID <- COVID[test.index, ]
train.COVID <- COVID[-test.index, ]

# Then from remaining training data, take another 20% for validation set
validation.index <- sample(1:nrow(train.COVID), 0.25 * nrow(train.COVID))
validation.COVID <- train.COVID[validation.index, ]
train.COVID <- train.COVID[-validation.index, ]

# 3.3.1
# Creating the best model for the COVID data set.
# Firstly we create a basic logistic regression model using stepAIC to determine
# which predictors are best suited for the model. 
# Using logistic regression

# Function that does CV, returning the averaged values for K-fold 
# Type 1 is for without regularisation, type 2 is with
CV <- function(data, k, cutoff, model.fit, type, model.matrix) {
  fold.size <- floor(nrow(data) / k)
  data <- data[sample(1:nrow(data)), ] # Shuffling the rows of the data 
  
  accuracies <- numeric(k)
  precisions <- numeric(k)
  sensitivities <- numeric(k)
  f1.scores <- numeric(k)
  
  for (i in 1:k) {
    val.indices <- ((i - 1) * fold.size + 1):(i * fold.size)
    val <- data[val.indices, ]
    train <- data[-val.indices, ]
    
    model <- model.fit(train)
    
    if (type == 1) {
      predicted.prob <- predict(model, newdata = val, type = "response")
      predicted <- ifelse(predicted.prob > cutoff, 1, 0)
    } else if (type == 2) {
      X.val <- model.matrix(val)
      predicted.prob <- predict(model, newx = X.val, type = 'response')
      predicted <- ifelse(predicted.prob > cutoff, 1, 0)
    } else {
      predicted <- predict(model, newdata = val)
    }
    
    confusion <- table(predicted, val$covidDeath)
    
    # Calculating all metrics
    TP <- confusion[2, 2]  
    TN <- confusion[1, 1]  
    FP <- confusion[2, 1]  
    FN <- confusion[1, 2]  
    
    accuracy <- (TP + TN) / (TP + TN + FP + FN)
    precision <- TP / (TP + FP)
    sensitivity <- TP / (TP + FN)
    specificity <- TN / (TN + FP)
    f1.score <- (2 * (precision * sensitivity)) / (precision + sensitivity)
    
    accuracies[i] <- accuracy
    precisions[i] <- precision
    sensitivities[i] <- sensitivity
    f1.scores[i] <- f1.score
    
  }
  return(list(
    Accuracy = mean(accuracies),
    Precision = mean(precisions),
    Sensitivity = mean(sensitivities),
    F1.Score = mean(f1.scores)
    ))
}

# Create the function for logistic regression
logistic.step.model <- glm(covidDeath ~ sex + vaccine + fever + cough + sorethroat + dyspnoea +
                             oxygensat + diarrhea + vomit + hematologic + downsyn + asthma + diabetes +
                             neurological + pneumopathy + obesity + age + 
                             preAdmCond + preAdmCondSymp + preAdmCondDis + preAdmCondSympComm +
                             deathCond + icu + durationHosp + respdistress + cardio + hepatic + immuno +
                             renal + durationICU
                           , data = train.COVID, family = binomial)

logistic.step.model.improv <- stepAIC(logistic.step.model, direction = "both")
summary(logistic.step.model.improv)

# Getting the graphs for analysis of the logistic regression model
vif(logistic.step.model.improv)
roc(train.COVID$covidDeath, logistic.step.model.improv$fitted.values, plot = TRUE, legacy.axes = TRUE,
    xlab = 'False Positive', ylab = 'True Positive', main = 'Logistic Model')
par(mfrow = c(2,2))
plot(logistic.step.model.improv)

# This model generated from step-wise selection is then used for k-fold CV to compare
# with other models.
logisticModel <- function(train) {
  glm(covidDeath ~ sex + vaccine + fever + cough + dyspnoea +
        oxygensat + diarrhea + downsyn + asthma + diabetes +
        neurological + pneumopathy + obesity + age + 
        deathCond + icu + durationHosp + respdistress + cardio + durationICU
      , data = train, family = binomial)
}

set.seed(11)
new.train.COVID <- rbind(validation.COVID, test.COVID)
logistic.measures <- CV(new.train.COVID, 10, 0.5, logisticModel, 1)




# Regularisation method
# Creating a model using a combination of both ridge and lasso regression
# Using the 60% training data to train model to determine approppriate alpha value
set.seed(99)
new.index <- sample(1:nrow(train.COVID), 0.2 * nrow(train.COVID))
train.val.COVID <- train.COVID[new.index, ]
train.train.COVID <- train.COVID[-new.index, ]

X <- model.matrix(covidDeath ~ sex + vaccine + fever + cough + sorethroat + dyspnoea +
                    oxygensat + diarrhea + vomit + hematologic + downsyn + asthma + diabetes +
                    neurological + pneumopathy + obesity + age + 
                    preAdmCond + preAdmCondSymp + preAdmCondDis + preAdmCondSympComm +
                    deathCond + icu + durationHosp + respdistress + cardio + hepatic + immuno +
                    renal + durationICU, data = train.train.COVID)[,-1]
y <- train.train.COVID$covidDeath

X.validation <- model.matrix(covidDeath ~ sex + vaccine + fever + cough + sorethroat + dyspnoea +
                               oxygensat + diarrhea + vomit + hematologic + downsyn + asthma + diabetes +
                               neurological + pneumopathy + obesity + age + 
                               preAdmCond + preAdmCondSymp + preAdmCondDis + preAdmCondSympComm +
                               deathCond + icu + durationHosp + respdistress + cardio + hepatic + immuno +
                               renal + durationICU, data = train.val.COVID)[,-1]
y.validation <- train.val.COVID$covidDeath

# Determining which alpha value creates the highest accuracy in predicting COVID deaths.

alphas <- seq(0, 1, by = 0.1)
model.accuracies <- numeric(length(alphas))
models <- list()
cutoff <- 0.5

set.seed(12)
for (i in seq_along(alphas)) {
  alpha <- alphas[i]
  cv.model <- cv.glmnet(X, y, alpha = alpha, family = "binomial", type.measure = "class")
  models[[i]] <- cv.model
  
  # Making predictions using validation set
  predicted.prob <- predict(cv.model, newx = X.validation, s = 'lambda.min', type = 'response')
  predicted.factor <- ifelse(predicted.prob > cutoff, 1, 0)
  
  # Determine accuracy of prediction
  accuracy <- sum(predicted.factor == y.validation) / length(y.validation)
  model.accuracies[i] <- accuracy
}

best.index <- which.max(model.accuracies)
best.alpha <- alphas[best.index]
best.model <- models[[best.index]]
best.lambda <- best.model$lambda.min

print(model.accuracies[best.index])
coef(best.model)
par(mfrow = c(1,1))
plot(best.model, xvar = 'lambda', label = TRUE)

cv.model.roc <- glmnet(X, y, alpha = best.alpha, family = "binomial", lambda = best.lambda)
cv.predicted.prob <- predict(cv.model.roc, X, type = "response")[, 1]
roc(y, cv.predicted.prob, plot = TRUE, legacy.axes = TRUE,
    xlab = 'False Positive', ylab = 'True Positive', main = 'Elastic Net Model')

# Creating a residual plot for elastic net model
enet.residuals <- (y - cv.predicted.prob) / sqrt(cv.predicted.prob * (1 - cv.predicted.prob))
plot(cv.predicted.prob, enet.residuals,
     main = "Pearson Residuals vs Predicted Probabilities", 
     xlab = "Predicted Probabilities",
     ylab = "Pearson Residuals")
abline(h = 0, col = "red", lty = 2)

# Conducting CV on tuned parameters to compare with other models
regularisationModel <- function(train) {
  X <- model.matrix(covidDeath ~ sex + vaccine + fever + cough + sorethroat + dyspnoea +
                      oxygensat + diarrhea + vomit + hematologic + downsyn + asthma + diabetes +
                      neurological + pneumopathy + obesity + age + 
                      preAdmCond + preAdmCondSymp + preAdmCondDis + preAdmCondSympComm +
                      deathCond + icu + durationHosp + respdistress + cardio + hepatic + immuno +
                      renal + durationICU, data = train)[,-1]
  y <- train$covidDeath
  cv.model <- glmnet(X, y, alpha = best.alpha, family = "binomial", lambda = best.lambda)
  return(cv.model)
}

X.model.matrix <- function(train) {
  X <- model.matrix(covidDeath ~ sex + vaccine + fever + cough + sorethroat + dyspnoea +
                      oxygensat + diarrhea + vomit + hematologic + downsyn + asthma + diabetes +
                      neurological + pneumopathy + obesity + age + 
                      preAdmCond + preAdmCondSymp + preAdmCondDis + preAdmCondSympComm +
                      deathCond + icu + durationHosp + respdistress + cardio + hepatic + immuno +
                      renal + durationICU, data = train)[,-1]
}

set.seed(13)
regularisation.measures <- CV(new.train.COVID, 10, 0.5, regularisationModel, 2, X.model.matrix)

# Attempting to use a General Additive Model, smoothing particular continous variables.
# Fitting this model to get plots
gam.spline.model <- gam(covidDeath ~ sex + vaccine + fever + cough + sorethroat + dyspnoea +
                          oxygensat + diarrhea + vomit + hematologic + downsyn + asthma + diabetes +
                          neurological + pneumopathy + obesity + s(age) + 
                          s(preAdmCond) + preAdmCondSymp + preAdmCondDis + preAdmCondSympComm +
                          deathCond + icu + durationHosp + respdistress + cardio + hepatic + immuno + renal
                          + durationICU, data = train.COVID, family = binomial)
summary(gam.spline.model)




# Analysis of polynomial model
roc(train.COVID$covidDeath, gam.spline.model$fitted.values, plot = TRUE, legacy.axes = TRUE,
    xlab = 'False Positive', ylab = 'True Positive', main = 'Spline Model')
par(mfrow = c(1,2))
plot(gam.spline.model, se = TRUE) # two plots 1 for age other for pre admission condition symptoms
par(mfrow = c(2,2))
gam.check(gam.spline.model, type = "pearson")


# Conducting CV on the gam with splines model
gamModel <- function(train) {
  gam(covidDeath ~ sex + vaccine + fever + cough + sorethroat + dyspnoea +
        oxygensat + diarrhea + vomit + hematologic + downsyn + asthma + diabetes +
        neurological + pneumopathy + obesity + s(age) + 
        s(preAdmCond) + preAdmCondSymp + preAdmCondDis + preAdmCondSympComm +
        deathCond + icu + durationHosp + respdistress + cardio + hepatic + immuno + renal
        + durationICU, data = train, family = binomial)
}
set.seed(21)
gam.measures <- CV(new.train.COVID, 10, 0.5, gamModel, 1)




# Making a Random Forest Model using randomForest()
forest.train.COVID <- train.COVID
forest.train.COVID$covidDeath <- factor(forest.train.COVID$covidDeath)
rfor.model <- randomForest(covidDeath ~ sex + vaccine + fever + cough + sorethroat + dyspnoea +
                           oxygensat + diarrhea + vomit + hematologic + downsyn + asthma + diabetes +
                           neurological + pneumopathy + obesity + age + 
                           preAdmCond + preAdmCondSymp + preAdmCondDis + preAdmCondSympComm +
                           deathCond + icu + durationHosp + respdistress + cardio + hepatic + immuno + renal
                         + durationICU, data = forest.train.COVID, importance = TRUE, ntree = 3, do.trace=TRUE)
par(mfrow = c(1,1))
plot(rfor.model)
varImpPlot(rfor.model)

# Getting measures for random forest model
rfModel <- function(train) {
  model <- randomForest(covidDeath ~ sex + vaccine + fever + cough + sorethroat + dyspnoea +
                               oxygensat + diarrhea + vomit + hematologic + downsyn + asthma + diabetes +
                               neurological + pneumopathy + obesity + age + 
                               preAdmCond + preAdmCondSymp + preAdmCondDis + preAdmCondSympComm +
                               deathCond + icu + durationHosp + respdistress + cardio + hepatic + immuno + renal
                             + durationICU, data = forest.train.COVID, importance = TRUE, ntree = 3, do.trace=TRUE)
}
set.seed(89)
rf.measures <- CV(new.train.COVID, 10, 0.5, rfModel, 3)


# 3.3.2 Making the best predictive model
logistic.step.model.2 <- glm(covidDeath ~ sex + vaccine + fever + cough + sorethroat + dyspnoea +
                               oxygensat + diarrhea + vomit + hematologic + downsyn + asthma + diabetes +
                               neurological + pneumopathy + obesity + age + 
                               age.group + preAdmCond + preAdmCondSymp + preAdmCondDis + preAdmCondSympComm
                           , data = train.COVID, family = binomial)

logistic.step.model.improv.2 <- stepAIC(logistic.step.model.2, direction = "both")
summary(logistic.step.model.improv.2)

predict.logistic.2 <- predict(logistic.step.model.improv.2, newdata = validation.COVID, type = "response")
cutoff <- 0.3
predicted.logistic.2 <- ifelse(predict.logistic.2 > cutoff, 1, 0)

confusion.logistic.2 <- table(Predicted = predicted.logistic.2, Actual = validation.COVID$covidDeath)

TP.l2 <- confusion.logistic.2[2, 2]  
TN.l2 <- confusion.logistic.2[1, 1]  
FP.l2 <- confusion.logistic.2[2, 1]  
FN.l2 <- confusion.logistic.2[1, 2]  

accuracy.l2 <- (TP.l2 + TN.l2) / (TP.l2 + TN.l2 + FP.l2 + FN.l2)
precision.l2 <- TP.l2 / (TP.l2 + FP.l2)
sensitivity.l2 <- TP.l2 / (TP.l2 + FN.l2)
f1.score.l2 <- (2 * (precision.l2 * sensitivity.l2)) / (precision.l2 + sensitivity.l2)
logistic.measures.2 <- list(
  Accuracy = accuracy.l2,
  Precision = precision.l2,
  Sensitivity = sensitivity.l2,
  F1.Score = f1.score.l2
)

# Trying the regularisation method.
set.seed(992)
new.index <- sample(1:nrow(train.COVID), 0.2 * nrow(train.COVID))
train.val.COVID <- train.COVID[new.index, ]
train.train.COVID <- train.COVID[-new.index, ]

X.2 <- model.matrix(covidDeath ~ sex + vaccine + fever + cough + sorethroat + dyspnoea +
                    oxygensat + diarrhea + vomit + hematologic + downsyn + asthma + diabetes +
                    neurological + pneumopathy + obesity + age + age.group +
                    preAdmCond + preAdmCondSymp + preAdmCondDis + preAdmCondSympComm, data = train.train.COVID)[,-1]
y.2 <- train.train.COVID$covidDeath

X.validation.2 <- model.matrix(covidDeath ~ sex + vaccine + fever + cough + sorethroat + dyspnoea +
                               oxygensat + diarrhea + vomit + hematologic + downsyn + asthma + diabetes +
                               neurological + pneumopathy + obesity + age + age.group +
                               preAdmCond + preAdmCondSymp + preAdmCondDis + preAdmCondSympComm, data = train.val.COVID)[,-1]
y.validation.2 <- train.val.COVID$covidDeath

# Determining which alpha value creates the highest accuracy in predicting COVID deaths.

alphas <- seq(0, 1, by = 0.1)
model.accuracies.2 <- numeric(length(alphas))
models.2 <- list()
cutoff <- 0.5

set.seed(12)
for (i in seq_along(alphas)) {
  alpha <- alphas[i]
  cv.model <- cv.glmnet(X.2, y.2, alpha = alpha, family = "binomial", type.measure = "class")
  models.2[[i]] <- cv.model
  
  # Making predictions using validation set
  predicted.prob <- predict(cv.model, newx = X.validation.2, s = 'lambda.min', type = 'response')
  predicted.factor <- ifelse(predicted.prob > cutoff, 1, 0)
  
  # Determine accuracy of prediction
  accuracy <- sum(predicted.factor == y.validation) / length(y.validation)
  model.accuracies[i] <- accuracy
}

best.index.2 <- which.max(model.accuracies.2)
best.alpha.2 <- alphas[best.index]
best.model.2 <- models.2[[best.index]]
best.lambda.2 <- best.model$lambda.min

coef(best.model.2)
# Conducting CV on tuned parameters to compare with other models
X.validation.21 <- model.matrix(covidDeath ~ sex + vaccine + fever + cough + sorethroat + dyspnoea +
                                 oxygensat + diarrhea + vomit + hematologic + downsyn + asthma + diabetes +
                                 neurological + pneumopathy + obesity + age + age.group +
                                 preAdmCond + preAdmCondSymp + preAdmCondDis + preAdmCondSympComm, data = validation.COVID)[,-1]
y.validation.21 <- validation.COVID$covidDeath

cv.model.2 <- glmnet(X.2, y.2, alpha = best.alpha, family = "binomial", lambda = best.lambda)

cv.predicted.prob.2 <- predict(cv.model.2, newx = X.validation.21, type = "response")[, 1]
cv.valid.factors.2 <- ifelse(cv.predicted.prob.2 > cutoff, 1, 0)

confusion.reg.2 <- table(Predicted = cv.valid.factors.2, Actual = validation.COVID$covidDeath)
TP.r2 <- confusion.reg.2[2, 2]  
TN.r2 <- confusion.reg.2[1, 1]  
FP.r2 <- confusion.reg.2[2, 1]  
FN.r2 <- confusion.reg.2[1, 2]  

accuracy.r2 <- (TP.r2 + TN.r2) / (TP.r2 + TN.r2 + FP.r2 + FN.r2)
precision.r2 <- TP.r2 / (TP.r2 + FP.r2)
sensitivity.r2 <- TP.r2 / (TP.r2 + FN.r2)
f1.score.r2 <- (2 * (precision.r2 * sensitivity.r2)) / (precision.r2 + sensitivity.r2)
regularisation.measures.2 <- list(
  Accuracy = accuracy.r2,
  Precision = precision.r2,
  Sensitivity = sensitivity.r2,
  F1.Score = f1.score.r2
)

# GAM model 
gam.spline.model.2 <- gam(covidDeath ~ sex + vaccine + fever + cough + sorethroat + dyspnoea +
                          oxygensat + diarrhea + vomit + hematologic + downsyn + asthma + diabetes +
                          neurological + pneumopathy + obesity + s(age) + age.group +
                          s(preAdmCond) + preAdmCondSymp + preAdmCondDis + preAdmCondSympComm, data = train.COVID, family = binomial)
predict.gam.2 <- predict(gam.spline.model.2, newdata = validation.COVID, type = "response")
predicted.gam.2 <- ifelse(predict.gam.2 > cutoff, 1, 0)

confusion.gam.2 <- table(Predicted = predicted.gam.2, Actual = validation.COVID$covidDeath)

TP.g2 <- confusion.gam.2[2, 2]  
TN.g2 <- confusion.gam.2[1, 1]  
FP.g2 <- confusion.gam.2[2, 1]  
FN.g2 <- confusion.gam.2[1, 2]  

accuracy.g2 <- (TP.g2 + TN.g2) / (TP.g2 + TN.g2 + FP.g2 + FN.g2)
precision.g2 <- TP.g2 / (TP.g2 + FP.g2)
sensitivity.g2 <- TP.g2 / (TP.g2 + FN.g2)
f1.score.g2 <- (2 * (precision.g2 * sensitivity.g2)) / (precision.g2 + sensitivity.g2)
gam.measures.2 <- list(
  Accuracy = accuracy.g2,
  Precision = precision.g2,
  Sensitivity = sensitivity.g2,
  F1.Score = f1.score.g2
)

forest.val.COVID <- validation.COVID
forest.val.COVID$covidDeath <- factor(forest.val.COVID$covidDeath)
rfor.model.2 <- randomForest(covidDeath ~ sex + vaccine + fever + cough + sorethroat + dyspnoea +
                             oxygensat + diarrhea + vomit + hematologic + downsyn + asthma + diabetes +
                             neurological + pneumopathy + obesity + age + age.group +
                             preAdmCond + preAdmCondSymp + preAdmCondDis + preAdmCondSympComm, data = forest.train.COVID, importance = TRUE, ntree = 3, do.trace=TRUE)
predict.rf.2 <- predict(rfor.model.2, newdata = forest.val.COVID, type = "prob")[, 2]
predicted.rf.2 <- ifelse(predict.rf.2 > cutoff, 1, 0)
confusion.rf.2 <- table(Predicted = predicted.rf.2, Actual = forest.val.COVID$covidDeath)

TP.rf2 <- confusion.rf.2[2, 2]  
TN.rf2 <- confusion.rf.2[1, 1]  
FP.rf2 <- confusion.rf.2[2, 1]  
FN.rf2 <- confusion.rf.2[1, 2]  

accuracy.rf2 <- (TP.rf2 + TN.rf2) / (TP.rf2 + TN.rf2 + FP.rf2 + FN.rf2)
precision.rf2 <- TP.rf2 / (TP.rf2 + FP.rf2)
sensitivity.rf2 <- TP.rf2 / (TP.rf2 + FN.rf2)
f1.score.rf2 <- (2 * (precision.rf2 * sensitivity.rf2)) / (precision.rf2 + sensitivity.rf2)
rf.measures.2 <- list(
  Accuracy = accuracy.rf2,
  Precision = precision.rf2,
  Sensitivity = sensitivity.rf2,
  F1.Score = f1.score.rf2
)

# Final test set measures for chosen model:
predict.gam.t <- predict(gam.spline.model.2, newdata = test.COVID, type = "response")
predicted.gam.t <- ifelse(predict.gam.t > cutoff, 1, 0)

confusion.gam.t <- table(Predicted = predicted.gam.t, Actual = test.COVID$covidDeath)

TP.gt <- confusion.gam.t[2, 2]  
TN.gt <- confusion.gam.t[1, 1]  
FP.gt <- confusion.gam.t[2, 1]  
FN.gt <- confusion.gam.t[1, 2]  

accuracy.gt <- (TP.gt + TN.gt) / (TP.gt + TN.gt + FP.gt + FN.gt)
precision.gt <- TP.gt / (TP.gt + FP.gt)
sensitivity.gt <- TP.gt / (TP.gt + FN.gt)
f1.score.gt <- (2 * (precision.gt * sensitivity.gt)) / (precision.gt + sensitivity.gt)
gam.measures.t <- list(
  Accuracy = accuracy.gt,
  Precision = precision.gt,
  Sensitivity = sensitivity.gt,
  F1.Score = f1.score.gt
)


# Making predictions on the Kaggle dataset
gam.model <- gam(covidDeath ~ sex + vaccine + fever + cough + sorethroat + dyspnoea +
                   oxygensat + diarrhea + vomit + hematologic + downsyn + asthma + diabetes +
                   neurological + pneumopathy + obesity + s(age) + 
                   age.group + s(preAdmCond) + preAdmCondSymp + preAdmCondDis + preAdmCondSympComm
                 , data = train.COVID, family = binomial)
summary(gam.model)
cutoff <- 0.30
predict.gam.kaggle <- predict(gam.model, newdata = COVID.kaggle, type = "response")
predicted.gam.kaggle <- ifelse(predict.gam.kaggle > cutoff, 1, 0)

COVID.kaggle$covidDeath <- predicted.gam.kaggle

results.gam.df <- data.frame(Patient_id = COVID.kaggle$Patient_id, covidDeath = predicted.gam.kaggle)
colnames(results.gam.df) <- c('Patient_id', 'covidDeath')
results.gam.df$covidDeath <- as.logical(results.gam.df$covidDeath)
write.csv(results.gam.df, file = "C:/Users/Leon/Downloads/submissionGAM.csv", row.names = FALSE)

# EDA for kaggle dataset
theme_set(theme_bw() +
            theme(text=element_text(size = 8, colour = "grey25", face = "bold"), 
                  plot.title = element_text(
                    size = 8, hjust = 0.5, colour = "steelblue", face = "bold")))

ggplot(COVID.kaggle, aes(fill = as.factor(covidDeath), x = age.group)) +
  geom_bar(position = 'dodge') + 
  labs(x = 'Age Group', y = 'Number of Patients', fill = 'COVID Death') +
  scale_fill_manual(values = c('0' = 'skyblue', '1' = 'orange'), labels = c('0' = 'False', '1' = 'True')) +
  ggtitle('Distribution of COVID Deaths across Ages')

age.model <- glm(covidDeath ~ age, data = COVID.kaggle, family = binomial())
summary(age.model)

ggplot(COVID.kaggle, aes(fill = as.factor(covidDeath), x = vaccine)) +
  geom_bar(position = 'dodge') + 
  labs(x = 'Vaccine Status (False - True)', y = 'Number of Patients', fill = 'COVID Death') +
  scale_fill_manual(values = c('0' = 'skyblue', '1' = 'orange'), labels = c('0' = 'False', '1' = 'True')) +
  scale_x_discrete(labels = c('0' = 'False', '1' = 'True')) +
  ggtitle('COVID Deaths relating to Vaccine status')

vaccine.model <- glm(covidDeath ~ vaccine, data = COVID.kaggle, family = binomial())
summary(vaccine.model)

ggplot(COVID.kaggle, aes(fill = as.factor(covidDeath), x = preAdmCondSymp)) +
  geom_bar(position = 'dodge') + 
  labs(x = 'Number of Admission Symptoms', y = 'Number of Patients', fill = 'COVID Death') +
  scale_fill_manual(values = c('0' = 'skyblue', '1' = 'orange'), labels = c('0' = 'False', '1' = 'True')) +
  ggtitle('COVID Deaths relating to Admission Symptoms')

admission.cond.model <- glm(covidDeath ~ preAdmCondSymp, data = COVID.kaggle, family = binomial())
summary(admission.cond.model)
