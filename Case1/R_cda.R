rm(list=ls()) 
print(utils::getSrcDirectory(function(){}))
print(utils::getSrcFilename(function(){}, full.names = TRUE))
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
# options(scipen=999)
options(scipen=0)
# dev.off()
Sys.setenv(LANG = "en")

library(glmnet)
library(MASS)
library(dplyr)
library(nlme)
library(caret)
library(ggplot2)
library(patchwork)

#####################3333
df <- read.table("preprocessed_data/df_imputed_train.csv", sep=",", header=TRUE)
df <- data.frame(df)
X = as.matrix(df[, 3:(dim(df)[2] - 1)])
y = df[, "y"]



### Modelling
# alphas = seq(0.1, 1, 0.1)
# nlambda = 200
# 
# results <- matrix(NA, nrow = nlambda, ncol = length(alphas))
# for(index in 1:length(alphas)){
#   cvfit <- cv.glmnet(X, y, nlambda = nlambda, alpha = alphas[index],
#                      type.measure = "mse")
#   results[,index] = cvfit$cvm
# }
# apply(results, 2, min)
# # 305.1648
# 
# 
# ### Second Try
# alphas = seq(0.4, 1, 0.01)
# nlambda = 200
# 
# results_2 <- array(NA, dim=c(nlambda, length(alphas), 3))
# # dim: 200, 10, 3
# for(index in 1:length(alphas)){
#   print(index)
#   cvfit <- cv.glmnet(X, y, nlambda = nlambda, alpha = alphas[index],
#                      type.measure = "mse", nfolds = 30)
# 
#   results_2[,index, 1] = cvfit$cvm
#   results_2[,index, 2] = cvfit$cvsd
#   results_2[,index, 3] = cvfit$lambda
# }
# 
# minimums = apply(results_2[, ,1], 2, min)
# indexes_minimums = apply(results_2[, ,1], 2, which.min)
# 
# dim(results_2[, ,2])
# 
# # apply(results_2[, ,2], 1, )
# se_list <- rep(0, length(indexes_minimums))
# for(i in 1:length(indexes_minimums)){
#   se_list[i] <- results_2[indexes_minimums[i],i, 2]
# }
# 
# 
# 
# plot(alphas, apply(results_2[, ,1], 2, min), ylim = c(200, 800))
# segments(x0=alphas, x1=alphas,
#          y0=minimums - qnorm(0.975)*se_list,
#          y1=minimums + qnorm(0.975)*se_list, col="red", lty = 2)
# 
# 
# 
# fit <- glmnet(X, y, nlambda = nlambda, alpha = alpha)
# 
# plot(fit)
# print(fit)
# 
# cvfit <- cv.glmnet(X, y, nlambda = 200, alpha = 0.8, type.measure = "mse")
# plot(cvfit)
# print(cvfit)
# cvfit$lambda.min
# 
# 
# coef(cvfit, s = "lambda.min")
# 
# wts <-  c(rep(1,50), rep(2,50))
# fit <- glmnet(x, y, alpha = 0.2, weights = wts, nlambda = 20)
# 
# cvfit <- cv.glmnet(x, y, type.measure = "mse", nfolds = 20)
# print(cvfit)


##################################3 crat

set.seed(123)
model <- train(
  y ~., data = df, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneLength = 50)

summary(model)

lambda_optimal <- model$bestTune


coef(model$finalModel, model$bestTune$lambda)

model$lambda
str(model)

model$finalModel

########################################
nlambda = 100
realizations = 10

lasso_results <- array(NA, dim=c(nlambda, realizations, 3))
# dim: 200, 10, 3
for(index in 1:realizations){
  print(index)
  cvfit <- cv.glmnet(X, y, nlambda = nlambda, alpha = 1,
                     type.measure = "mse", nfolds = nrow(X),
                     grouped = FALSE)
  lasso_results[,index, 1] = cvfit$cvm
  lasso_results[,index, 2] = cvfit$cvsd
  lasso_results[,index, 3] = cvfit$lambda
}

lasso_averaged <- apply(lasso_results[, , 1], 1, mean)
lasso_sd  <- apply(lasso_results[, , 1], 1, var)

# lasso_results[, , 3][106]
true_alphas = lasso_results[,1, 3]
  
plot(true_alphas, lasso_averaged)
segments(x0=true_alphas, x1=true_alphas,
         y0=lasso_averaged - qnorm(0.975)*lasso_sd,
         y1=lasso_averaged + qnorm(0.975)*lasso_sd, col="red", lty = 2)
abline(v = true_alphas[which.min(lasso_averaged)])

### 2.118259; 332.1121
### 
lasso_leave_one_out = lasso_results

#### repeated cross validation for Lasso
k_fold_outer = 50
nlambda = 100
lasso_results <- array(NA, dim=c(nlambda, realizations, 3))

for(fold in 1:k_fold_outer){
  print(fold)
  cvfit <- cv.glmnet(X, y, nlambda = nlambda, alpha = 1,
                     type.measure = "mse", nfolds = 20)
  lasso_results[,fold, 1] = cvfit$cvm
  lasso_results[,fold, 2] = cvfit$cvsd
  lasso_results[,fold, 3] = cvfit$lambda
}

lasso_averaged <- apply(lasso_results[, , 1], 1, mean)
lasso_sd  <- apply(lasso_results[, , 1], 1, sd)
true_alphas = lasso_results[,1, 3]

plot(true_alphas, lasso_averaged)
segments(x0=true_alphas, x1=true_alphas,
         y0=lasso_averaged - qnorm(0.975)*lasso_sd,
         y1=lasso_averaged + qnorm(0.975)*lasso_sd, col="red", lty = 2)
abline(v = true_alphas[which.min(lasso_averaged)])
min(lasso_averaged)

###### picking optimal lambda
cv1  <- cv.glmnet(x, y, foldid = foldid, alpha = 1)
cv.5 <- cv.glmnet(x, y, foldid = foldid, alpha = 0.5)
cv0  <- cv.glmnet(x, y, foldid = foldid, alpha = 0)

lambdas_list = seq(1, 4, 0.01)

cvfit <- cv.glmnet(X, y, lambda = lambdas_list, alpha = 1,
                   type.measure = "mse", nfolds = 10)
plot(cvfit$cvm)

cv_default <- cv.glmnet(X, y, nlambda = 200, alpha = 1,
                        type.measure = "mse", nfolds = 20)
cv_default

############## repeated cross validation - scheme
df_nestedcross <- data.frame(df[, 2:118], y= df[, "y"])
yourdata <- df_nestedcross[sample(nrow(df_nestedcross)),]

folds_number = 20
repeated_number = 1
nlambda = 300
# inner_fold = 10

folds <- cut(seq(1,nrow(yourdata)),breaks=folds_number,labels=FALSE)
rmse_list <- array(NA, dim = c(repeated_number, folds_number, 3))
final_results <- array(NA, dim=c(repeated_number, nlambda, folds_number, 3))

for(index in 1:repeated_number){
  print("repetaed number:", index)
  
  #Perform k fold cross validation
  for(fold in 1:folds_number){
    print(fold)
    testIndexes <- which(folds==fold,arr.ind=TRUE)
    testData <- yourdata[testIndexes, ]
    trainData <- yourdata[-testIndexes, ]
    
    X_train = as.matrix(trainData[,-dim(trainData)[2]])
    y_train = trainData[, "y"]
    X_test = as.matrix(testData[,-dim(testData)[2]])
    y_test = testData[, "y"]
    
    # inner loop
    cv_fit <- cv.glmnet(X_train, y_train, nlambda = nlambda, alpha = 1,
                        type.measure = "mse", nfolds = nrow(X_train),
                        grouped = F)
    
    final_results[index, ,fold, 1] = cv_fit$cvm
    final_results[index, ,fold, 2] = cv_fit$cvsd
    final_results[index, ,fold, 3] = cv_fit$lambda
    
    predictions = predict(cv_fit, newx = X_test, s = "lambda.min")
    
    rmse_list[index, fold, 1] = sqrt(mean((predictions - y_test)^2))
    rmse_list[index, fold, 2] = cv_fit$lambda.min
    
  }
}

##### 19.3197
# 20.54628 17.69697 11.96690 23.03625 22.81544
rmse_loov <- rmse_list

means_list = array(NA, dim = c(repeated_number, nlambda))

sd(apply(rmse_list[, , 1], 1, mean))
mean(apply(rmse_list[, , 1], 1, mean))

for(j in 1:dim(final_results)[1]){
  means_list[j, ] <- apply(final_results[j, , , 1], 1, mean)
}
final_means <- apply(means_list, 2, mean)
min(final_means)
which.min(final_means)

rmse_list_final = apply(rmse_list[, , 1], 1, mean)
min(rmse_list_final)

over_folds <- apply(final_results[1, , , 1], 1, mean)
final_results[, 1, 3][which.min(over_folds)]



mean(rmse_list[, 1])
sd(rmse_list[, 1])

########## final lambda #############

cv_final <- cv.glmnet(X, y_centered, nlambda = 300, alpha = 1,
                    type.measure = "mse", nfolds = nrow(X),
                    intercept = FALSE)
cv_final
plot(cv_final)

coeficients <- as.matrix(coef(cv_final, s = "lambda.min"))
dataframe <- data.frame(coeficients) 
dataframe %>% filter(s1 > 0) %>% arrange(desc(s1))


p1 = ggplot(df, aes(x = num__x_22, y = y, color = factor(cat__C_4_K))) + geom_point()
p2 = ggplot(df, aes(x = num__x_71, y = y, color = factor(cat__C_5_H))) + geom_point()
p3 = ggplot(df, aes(x = num__x_81, y = y, color = factor(cat__C_4_K))) + geom_point()
p4 = ggplot(df, aes(x = num__x_61, y = y, color = factor(cat__C_5_H))) + geom_point()
p5 = ggplot(df, aes(x = num__x_61, y = y, color = factor(cat__C_1_G))) + geom_point()

(p1 + p2 + p3) / (p4 + p5)
# num__x_22.cat__C_4_K   9.61110520
# num__x_71.cat__C_5_H   4.47666273
# num__x_81.cat__C_4_K   3.89358160
# num__x_61.cat__C_5_H   3.53662310
# num__x_61.cat__C_1_G



########## final lambda - 10 fold #############
cv_final <- cv.glmnet(X, y, nlambda = 300, alpha = 1,
                      type.measure = "mse", nfolds = nrow(X),
                      intercept = FALSE)
cv_final

###################### final stuff
X_small <- X[,1:117]
s <- svd(X_small)
V <- s$v
norm(t(s$v) %*% s$v, type = "F")

y_centered = y - mean(y)
A = s$u %*% diag(s$d)
y_new_domain = solve(A) %*% y_centered

cv_simple <- cv.glmnet(t(V), y_new_domain, nlambda = 300, alpha = 1,
                      type.measure = "mse", nfolds = 10,
                      intercept = FALSE)
predictions <- predict(cv_simple, t(V), s = "lambda.min")
sqrt(mean((A %*% predictions - y_centered)^2))

cbind(A %*% predictions, y_centered)

### original domain
cv_original <- cv.glmnet(X_small, y_centered, nlambda = 300, alpha = 1,
                       type.measure = "mse", nfolds = nrow(X_small),
                       intercept = FALSE)
cv_original
plot(cv_original)

############################################################
cv_fit <- cv.glmnet(X_small, y_centered, nlambda = 300, alpha = 1,
                         type.measure = "mse", nfolds = 10,
                         intercept = FALSE)
cv_fit

repeated_cv_final = 10
lambdas_final = array(NA, dim = c(repeated_cv_final, 2))
for (i in 1:repeated_cv_final){
  print(i)
  cv_fit_intercept <- cv.glmnet(X, y, nlambda = 300, alpha = 1,
                                type.measure = "mse", nfolds = 10)
  lambdas_final[i, 1] = cv_fit_intercept$lambda.min
  lambdas_final[i, 2] = cv_fit_intercept$lambda.1se
}

cv_fit_intercept

par(mfrow = c(1, 1))
hist(lambdas_final[, 1], breaks = 20)

# plot(cv_fit)
# 
# predict.glmnet
# 
# cv_fit <- cv.glmnet(X_small, y_centered, nlambda = 300, alpha = 1,
#                     type.measure = "mse", nfolds = 20,
#                     intercept = FALSE)
# cv_fit
# plot(cv_fit)

#### this gives consistent results !!!!!
coeficients <- as.matrix(coef(cv_fit, s = "lambda.min"))
dataframe <- data.frame(coeficients) 
dataframe %>% filter(s1 > 0) %>% arrange(desc(s1))



fit <- glmnet(X_small, y, lambda = mean(lambdas_final[, 1]))
fit
plot(fit)

residuals <- y - predict(fit, X_small)
plot(residuals)


options(scipen=999)
coeficients <- as.matrix(coef(fit))
dataframe <- data.frame(coeficients) 
dataframe %>% filter(s0 > 0) %>% arrange(desc(s0)) %>% round(., 4)
                         
                      




residuals <- y - fitted_values
plot(residuals)

df_fitted_values <- data.frame(y = as.numeric(y), fitted_values = as.numeric(fitted_values))



shapiro.test(residuals)
hist(residuals)
##### excluding interactions

X_small <- X[, 1:95]
cv_no_interactions <- cv.glmnet(X_small, y, nlambda = 300, alpha = 1,
                      type.measure = "mse", nfolds = 10,
                      grouped = F, intercept = FALSE)
cv_no_interactions


colnames(X)

summary(fit)
print(fit)

########################################################################
### predictions

df_test <- read.table("preprocessed_data/df_imputed_test.csv", sep=",", header=TRUE)
df_test <- as.matrix(data.frame(df_test[, 3:dim(df_test)[2]]))

predictions <- predict(fit, df_test)
write(predictions, "myFile.txt", ncolumns = 1)















