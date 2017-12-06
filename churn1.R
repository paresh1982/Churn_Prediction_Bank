
install.packages("xgboost")
install.packages("data.table")
install.packages("funModeling")
install.packages("Matrix")


library(xgboost)
library(data.table)
library(funModeling)
library(Matrix)

train <- fread('train.csv', stringsAsFactors = FALSE)
test <- fread('test.csv', stringsAsFactors = FALSE)
submission <- data.frame(UCIC_ID=test$UCIC_ID,Responders=0.5)
test$Responders <- 0

train$UCIC_ID <- NULL
test$UCIC_ID <- NULL

total <- rbind(train,test)
data_summary <- df_status(total,print_results = FALSE)

#Remove columns with more than 95% missing values
total <- total[,data_summary$p_na<99,with=FALSE]
total[is.na(total)] <- -9999
total <- data.frame(total)


for(i in 1:ncol(total)){
  if(class(total[,i])=='character'){
    total[,i] <- as.numeric(as.factor(total[,i]))
  }
}

train <- total[1:nrow(train),]
test <- total[-(1:nrow(train)),]
target <- train$Responders
train$Responders <- NULL
test$Responders <- NULL
trainSparse <- sparse.model.matrix(~.,data=train)
testSparse <- sparse.model.matrix(~.,data=test)
common <- intersect(colnames(trainSparse),colnames(testSparse))
trainSparse <- trainSparse[,common]
testSparse <- testSparse[,common]

dtrain <- xgb.DMatrix(data = trainSparse, label = target)
dtest <- xgb.DMatrix(data = testSparse)

max.depth = 6.0000	
min_child_weight = 1.0000	
subsample = 0.7742	
lambda = 0.2568	
alpha = 0.9799	
gamma = 5.0000	
colsample = 0.6696	

optimal_round <- 2100
model <- xgb.train(params = list(booster = "gbtree", 
                                 eta = 0.01,
                                 max_depth = max.depth,
                                 min_child_weight = min_child_weight,
                                 subsample = subsample, 
                                 colsample_bytree = colsample,
                                 objective = "binary:logistic",
                                 eval_metric = "auc"),
                   data = dtrain, 
                   nround = optimal_round,
                   maximize = TRUE,
                   lambda = lambda,
                   gamma = gamma,
                   alpha = alpha,
                   nthread = 10,
                   verbose = TRUE,
                   tree_method = 'auto'
)

pred <- predict(model,dtest)
submission$Responders <- pred
fwrite(submission,'predXGBTree_1.csv')
