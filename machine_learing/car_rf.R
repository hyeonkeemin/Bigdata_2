#install.packages('randomForest')
#install.packages('ROCR')

rm(list=ls())

library(randomForest)
library(caret) # 성능평가시 confusion matrix를 그리기 위한 라이브러리
library(e1071) # 모델 튜닝
library(ROCR) # 모델 평가

car = read.csv('file:///C:/Users/USER/Desktop/R/homework2.csv', header = T)

car = car[complete.cases(car),] # 결측치 제거
sum(is.na(car)) # 결측치 있는지 확인


############### 변수정리 ###############

#str(car) # 변수 수치 확인. 하이브리드와 LPG 타입은 범주형 자료이므로 factor로 변환 필요
car$LPG = as.factor(car$LPG)
car$하이브리드 = as.factor(car$하이브리드)
# 년식도 범주화
car$년식 = as.factor(car$년식)

# 숫자형 데이터의 정규화
car$연비 = scale(car$연비)
car$마력 = scale(car$마력)
car$토크 = scale(car$토크)
car$배기량 = scale(car$배기량)
car$중량 = scale(car$중량)


################ 학습데이터와 검증데이터 셋 분리 #############
train_idx = createDataPartition(car$가격, p=0.6, list=FALSE)

train_df = car[train_idx,]
test_df = car[train_idx,]

# near-zero variance 확인 후 변수 제거
nearZeroVar(train_df, saveMetrics = TRUE)
train_df = train_df[-nearZeroVar(car)]

str(train_df)

