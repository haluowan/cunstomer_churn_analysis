############################# 随机森林客户流失分类#################################
rm(list=ls())
# 1.读取数据
model_data=read.csv("D:\\CDA python work\\customer churn\\R_model_data.csv",sep=",")

# 检查数据
str(model_data) ##探寻数据集内部结构##
View(head(model_data,100)) ##弹出视窗查看数据##
table(model_data$label) ##生成频数表，若想看某个变量的频数可用'$x'，x为变量##

model_data$X<-NULL

# 2.从数据集中随机抽70%定义为训练数据集，30%为测试数据集
set.seed(123)
model_data$ind <- sample(2, nrow(model_data), replace=TRUE, prob=c(0.7, 0.3))  ##sample(分成几组，样本数量，重复抽样，权重） nrow(ma_resp_data)=数据行数##
head(model_data) ##展示前6行数据##
table(model_data$ind)

trainData <- model_data[model_data$ind==1,] ##训练组##
testData <- model_data[model_data$ind==2,] ##测试组##
trainData$ind <- NULL ##删除ind##
testData$ind <- NULL ##删除ind##
names(trainData) ##查看变量名##
names(testData)

table(trainData$label)
table(testData$label)

# 3.构建模型
library(rpart) ##调用rpart##
library(rpart.plot) ##调用rpart.plot##
tree_fit_ma <- rpart(label~.,data = trainData, method = "class", cp=0.0005)  # 误差的增益值cp=0.0005

names(tree_fit_ma)
print(tree_fit_ma) ##输出树的生成规则##
summary(tree_fit_ma) ##输出树的具体生成过程##

### 得到分类规则
library(rattle)
asRules(tree_fit_ma)

### 对决策树绘图
rpart.plot(tree_fit_ma, type=4, extra=104,under=F, clip.right.labs=F, fallen.leaves=T,
           branch=1, uniform=T, shadow.col="gray", box.col="green", border.col="blue",
           split.col="red", split.cex=1.2, main="CDA - Decision Tree - tree_fit_ma")

rpart.plot(tree_fit_ma, branch=1, branch.type=2, type=1, extra=104,  
           shadow.col="gray", box.col="green", border.col="blue", split.col="red",  
           split.cex=1.2, main="CDA - Decision Tree - tree_fit_ma")

prp(tree_fit_ma, type=2, extra=104, nn=TRUE, fallen.leaves=TRUE,
    varlen=0, faclen=0, cex=1, shadow.col="grey", branch.lty=2,
    main="CDA - Decision Tree - tree_fit_ma")

### 4.对决策树剪枝
##  通过上面的分析来确定cp的值  
## 我们可以用下面的办法选择具有最小xerror的cp的办法：  
best_cp= tree_fit_ma$cptable[which.min(tree_fit_ma$cptable[,"xerror"]),"CP"];best_cp

tree_fit_ma_pruned <- prune(tree_fit_ma, cp=best_cp); ##根据CP值对决策树进行剪枝，即剪去cp值较小的不重要分支##


summary(tree_fit_ma_pruned)

asRules(tree_fit_ma_pruned)

# 图1
rpart.plot(tree_fit_ma_pruned, branch=1, branch.type=2, type=1, extra=104,
           shadow.col="gray", box.col="green", border.col="blue", split.col="red",
           split.cex=1.2, main="Decision Tree - tree_fit_ma") 

# 图2
library(RColorBrewer)
pal = c("Greens",	"Purples")
fancyRpartPlot(tree_fit_ma_pruned, main="Decision Tree", sub='Author: ', palettes= pal)

# 5. 检验模型
pred_ma <- predict(tree_fit_ma_pruned, newdata = testData, type = "class")

# 混淆矩阵
conf.matrix_ma <- table(testData$label, pred_ma)
rownames(conf.matrix_ma) <- paste("Actual", rownames(conf.matrix_ma), sep = ":")
colnames(conf.matrix_ma) <- paste("Pred", colnames(conf.matrix_ma), sep = ":")
print(conf.matrix_ma)

# 准确率
CValue <- sum(diag(conf.matrix_ma))/sum(conf.matrix_ma);CValue # C value
# ErrorRate <- 1 - CValue
ErrorRate <- 1.0 - (conf.matrix_ma[1, 1] + conf.matrix_ma[2, 2])/sum(conf.matrix_ma);ErrorRate

#ROC
library(ROCR)
pred_ma_1 <- as.numeric(pred_ma)
pred <- prediction(pred_ma_1,testData$label)
perf <- performance(pred,'tpr','fpr')
plot(perf,col='black',lty=3,lwd=3,main='ROC Curve')

# 交叉验证
CrossVal <- testData
names(CrossVal)

n = nrow(testData)
K = 10
tail = n%/%K   ##求n除以K的商
set.seed(5)
alea = runif(n)  ##生成一个n X 1的向量，原始都是随机树
ran = rank(alea) ##将alea的元素先从小到大排序，然后返回元素所在的位置
bloc = (ran - 1)%/%tail + 1  ##bloc也是n X 1的向量，然后值的范围为1:10, %/%为整数除法
bloc = as.factor(bloc)
print(summary(bloc))

all.err = numeric(0)

for (k in 1:K) {
    arbre <- rpart(label ~., data = CrossVal[bloc!=k,], method = 'class')
    pred <- predict(arbre, newdata = CrossVal[bloc == k,], type = 'class')
    mc <- table(CrossVal$label[bloc == k], pred)
    err <- 1.0 - (mc[1, 1] + mc[2, 2])/sum(mc)
    all.err <- rbind(all.err, err)
}

print(all.err)
print(mean(all.err))

# 6.输出模型结果
summary(tree_fit_ma_pruned)

asRules(tree_fit_ma_pruned)

fancyRpartPlot(tree_fit_ma_pruned, main="Decision Tree", sub='Author: ', palettes= pal)



#######################################################################################################
#######################################################################################################
# 尝试一：删除重要性较小的cityuvs /commentnums_pre 
tree_fit_ma_pruned$variable.importance

# 1.划分数据集
trainData2=trainData
testData2=testData
trainData2$commentnums_pre <- NULL 
trainData2$cityuvs <- NULL 
testData2$commentnums_pre <- NULL 
testData2$cityuvs <- NULL 
names(trainData2) 
names(testData2)

# 2.构建模型
library(rpart) ##调用rpart##
library(rpart.plot) ##调用rpart.plot##
tree_fit_ma2 <- rpart(label~.,data = trainData2, method = "class", cp=0.0005)  # 误差的增益值cp=0.0005
names(tree_fit_ma2)
print(tree_fit_ma2) ##输出树的生成规则##
summary(tree_fit_ma2) ##输出树的具体生成过程##

# 3.修剪决策树
best_cp= tree_fit_ma2$cptable[which.min(tree_fit_ma2$cptable[,"xerror"]),"CP"];best_cp

tree_fit_ma_pruned2 <- prune(tree_fit_ma2, cp=best_cp); ##根据CP值对决策树进行剪枝，即剪去cp值较小的不重要分支##

summary(tree_fit_ma_pruned2)
tree_fit_ma_pruned2$variable.importance
asRules(tree_fit_ma_pruned2)

 # 4.评价模型
pred_ma2 <- predict(tree_fit_ma_pruned2, newdata = testData2, type = "class")

# 混淆矩阵
conf.matrix_ma2 <- table(testData2$label, pred_ma2)
rownames(conf.matrix_ma2) <- paste("Actual", rownames(conf.matrix_ma2), sep = ":")
colnames(conf.matrix_ma2) <- paste("Pred", colnames(conf.matrix_ma2), sep = ":")
print(conf.matrix_ma2)

# 准确率
CValue
CValue2 <- sum(diag(conf.matrix_ma2))/sum(conf.matrix_ma2);CValue2 # C value
# ErrorRate <- 1 - CValue
ErrorRate
ErrorRate2 <- 1.0 - (conf.matrix_ma2[1, 1] + conf.matrix_ma2[2, 2])/sum(conf.matrix_ma2);ErrorRate2


# 尝试二：删除重要性较小的lowestprice  /cr_pre /uv_pre2/businessrate_pre2/hotelcr 
tree_fit_ma_pruned2$variable.importance

# 1.划分数据集
trainData3=trainData2
testData3=testData2
trainData3$uv_pre2 <- NULL 
trainData3$cr_pre <- NULL
trainData3$lowestprice <- NULL
trainData3$businessrate_pre2 <- NULL
trainData3$hotelcr  <- NULL

testData3$uv_pre2 <- NULL 
testData3$cr_pre <- NULL 
testData3$lowestprice <- NULL
testData3$businessrate_pre2 <- NULL
testData3$hotelcr <- NULL
names(trainData3) 
names(testData3)

# 2.构建模型
library(rpart) ##调用rpart##
library(rpart.plot) ##调用rpart.plot##
tree_fit_ma3 <- rpart(label~.,data = trainData3, method = "class", cp=0.0005)  # 误差的增益值cp=0.0005
names(tree_fit_ma3)
print(tree_fit_ma3) ##输出树的生成规则##
summary(tree_fit_ma3) ##输出树的具体生成过程##

# 3.修剪决策树
best_cp= tree_fit_ma3$cptable[which.min(tree_fit_ma3$cptable[,"xerror"]),"CP"];best_cp

tree_fit_ma_pruned3 <- prune(tree_fit_ma3, cp=best_cp); ##根据CP值对决策树进行剪枝，即剪去cp值较小的不重要分支##

summary(tree_fit_ma_pruned3)
tree_fit_ma_pruned3$variable.importance
asRules(tree_fit_ma_pruned3)

# 4.评价模型
pred_ma3 <- predict(tree_fit_ma_pruned3, newdata = testData3, type = "class")

# 混淆矩阵
conf.matrix_ma3 <- table(testData3$label, pred_ma3)
rownames(conf.matrix_ma3) <- paste("Actual", rownames(conf.matrix_ma3), sep = ":")
colnames(conf.matrix_ma3) <- paste("Pred", colnames(conf.matrix_ma3), sep = ":")
print(conf.matrix_ma3)

# 准确率
CValue2
CValue3 <- sum(diag(conf.matrix_ma3))/sum(conf.matrix_ma3);CValue3 # C value
# ErrorRate <- 1 - CValue
ErrorRate2
ErrorRate3 <- 1.0 - (conf.matrix_ma3[1, 1] + conf.matrix_ma3[2, 2])/sum(conf.matrix_ma3);ErrorRate3

#ROC
library(ROCR)
pred_ma_3 <- as.numeric(pred_ma3)
pred <- prediction(pred_ma_3,testData3$label)
perf <- performance(pred,'tpr','fpr')
plot(perf,col='black',lty=3,lwd=3,main='ROC Curve')

# 交叉验证
CrossVal <- testData3
names(CrossVal)

n = nrow(testData3)
K = 10
tail = n%/%K   ##求n除以K的商
set.seed(5)
alea = runif(n)  ##生成一个n X 1的向量，原始都是随机树
ran = rank(alea) ##将alea的元素先从小到大排序，然后返回元素所在的位置
bloc = (ran - 1)%/%tail + 1  ##bloc也是n X 1的向量，然后值的范围为1:10, %/%为整数除法
bloc = as.factor(bloc)
print(summary(bloc))

all.err = numeric(0)

for (k in 1:K) {
    arbre <- rpart(label ~., data = CrossVal[bloc!=k,], method = 'class')
    pred <- predict(arbre, newdata = CrossVal[bloc == k,], type = 'class')
    mc <- table(CrossVal$label[bloc == k], pred)
    err <- 1.0 - (mc[1, 1] + mc[2, 2])/sum(mc)
    all.err <- rbind(all.err, err)
}

print(all.err)
print(mean(all.err))


# 5.输出结果
summary(tree_fit_ma_pruned3)
tree_fit_ma_pruned3$variable.importance
asRules(tree_fit_ma_pruned3)

fancyRpartPlot(tree_fit_ma_pruned3, main="customer churn classify", sub=" ",palettes= pal)


### 生成的规则，用来对客户进行画像

# 尝试三：删除重要性较小的h
tree_fit_ma_pruned3$variable.importance

# 1.划分数据集
trainData4=trainData3
testData4=testData3
trainData4$h <- NULL 

testData4$h<- NULL 

names(trainData4) 
names(testData4)

# 2.构建模型
library(rpart) ##调用rpart##
library(rpart.plot) ##调用rpart.plot##
tree_fit_ma4 <- rpart(label~.,data = trainData4, method = "class", cp=0.0005)  # 误差的增益值cp=0.0005
names(tree_fit_ma4)
print(tree_fit_ma4) ##输出树的生成规则##
summary(tree_fit_ma4) ##输出树的具体生成过程##

# 3.修剪决策树
best_cp= tree_fit_ma4$cptable[which.min(tree_fit_ma4$cptable[,"xerror"]),"CP"];best_cp

tree_fit_ma_pruned4 <- prune(tree_fit_ma4, cp=best_cp); ##根据CP值对决策树进行剪枝，即剪去cp值较小的不重要分支##

summary(tree_fit_ma_pruned4)
tree_fit_ma_pruned4$variable.importance
asRules(tree_fit_ma_pruned4)

# 4.评价模型
pred_ma4 <- predict(tree_fit_ma_pruned4, newdata = testData4, type = "class")

# 混淆矩阵
conf.matrix_ma4 <- table(testData4$label, pred_ma4)
rownames(conf.matrix_ma4) <- paste("Actual", rownames(conf.matrix_ma4), sep = ":")
colnames(conf.matrix_ma4) <- paste("Pred", colnames(conf.matrix_ma4), sep = ":")
print(conf.matrix_ma4)

# 准确率
CValue
CValue2
CValue3
CValue4 <- sum(diag(conf.matrix_ma4))/sum(conf.matrix_ma4);CValue4 # C value
# ErrorRate <- 1 - CValue
ErrorRate3
ErrorRate4 <- 1.0 - (conf.matrix_ma4[1, 1] + conf.matrix_ma4[2, 2])/sum(conf.matrix_ma4);ErrorRate3

# 最终选择tree_fit_ma3


