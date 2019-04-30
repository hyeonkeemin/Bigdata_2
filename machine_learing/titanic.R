rm(list=ls())

#install.packages('gridExtra')
#install.packages("stringr")
#install.packages('caret') 
#install.packages('e1071')
#install.packages('randomForest')
#install.packages('dplyr')
#install.packages('ggplot2')
#install.packages('ggmosaic')
#install.packages('ROCR')
#install.packages('pscl')

library(caret)
library(randomForest)
library(e1071)
library(dplyr)
library(ggplot2)
library(stringr)
library(ggmosaic)
library(gridExtra)
library(ROCR)
library(pscl)


# 데이터 읽기
test = read.csv('file:///C:/Users/hyeonkee/Desktop/빅데이터/R/test.csv', header=T, na="") # na='' : 빈값 NA로 바꾸기
train = read.csv('file:///C:/Users/hyeonkee/Desktop/빅데이터/R/train.csv', header=T, na="")

result = read.csv('file:///C:/Users/hyeonkee/Desktop/빅데이터/R/gender_submission.csv', header=T)

test = merge(result, test)

apply(train, 2, function(x) length(unique(x))) # 전체 변수 고유값 확인

colSums(is.na(train)) # 결측치 확인
colSums(is.na(test))
# 결측치 확인결과 train 데이터셋에서 1.Embarked, 2.Cabin, 4.Age / test 데이터 셋에서 3.Fare, 4.age 확인




########################### 데이터 전처리 ###########################
train = select(train, -PassengerId, -Ticket, -Name) # 영어 이름의 경우 중간 이름에서 정보를 얻을 수 있지만 한국과 조금 무관하다 판단하여 제거함.
train$Survived = as.factor(train$Survived)
train$Pclass = as.factor(train$Pclass)
str(train)

test$Pclass = as.factor(test$Pclass)
test$Survived = as.factor(test$Survived)
str(test)


## SibSp : 형제 수, Parch : 부모/자녀 수. 비슷한 변수를 줄이면 모델이 단순해짐
train = train %>% 
  mutate(Familiy = SibSp + Parch + 1,
         FamiliySize = case_when(Familiy == 1 ~ 'Single',
                                 Familiy > 1 & Familiy <= 4 ~ 'Normal',
                                 Familiy > 4 ~ 'Big'),
         FamiliySize = factor(FamiliySize, levels = c('Single', 'Normal', 'Big')))

train = select(train, -SibSp, -Parch, -Familiy)

# test 셋에도 동일 처리
test = test %>% 
  mutate(Familiy = SibSp + Parch + 1,
         FamiliySize = case_when(Familiy == 1 ~ 'Single',
                                 Familiy > 1 & Familiy <= 4 ~ 'Normal',
                                 Familiy > 4 ~ 'Big'),
         FamiliySize = factor(FamiliySize, levels = c('Single', 'Normal', 'Big')))


## cabin : 숫자보다 알파뱃의 의미가 있을 것이므로 문자열만 추출
train$Cabin = str_extract(train$Cabin, '[A-z]')
head(train$Cabin)
unique(train$Cabin) # 변수 내 고유값 확인 완료

# test 셋에도 동일 처리
test$Cabin = str_extract(test$Cabin, '[A-z]')
head(test$Cabin)
unique(test$Cabin)


########## 결측치 처리
##### 1. Embarked
na_em = train %>% filter(is.na(Embarked))
na_em
ggplot(train, aes(x=Embarked, y=Fare))+
  geom_boxplot() # 확인 결과 Embarked의 NA값을 C로 넣는게 좋겠다. 그래프 좀 더 예쁘게 만들어보기

train$Embarked[is.na(train$Embarked)] = 'C'
colSums(is.na(train)) # 결측치 처리 확인 완료


##### 2. Cabin
train$Cabin[is.na(train$Cabin)] = 'N'
test$Cabin[is.na(test$Cabin)] = 'N'

train$Cabin = as.factor(train$Cabin)
test$Cabin = as.factor(test$Cabin)

##### 3. Fare
filter(test, is.na(Fare)) # 확인 결과 Embarked 값이 S로 확인. S의 평균 값으로 대체
test$Fare[is.na(test$Fare)] = mean(filter(test, Embarked=='S')$Fare, na.rm=T)
colSums(is.na(test)) #결측치 확인 완료


##### 4. Age
# 나이의 경우 20% 정도가 데이터가 없으나 생존율에 영향을 미칠것으로 예상됨. 따라서 정확도를 높이기 위해 머신러닝을 통해 결측값 예측
age_traindf_train = filter(train, !is.na(Age))
age_traindf_test = filter(train, is.na(Age))

age_testdf_train = filter(test, !is.na(Age))
age_testdf_test = filter(test, is.na(Age))

fitControl = trainControl(method = 'repeatedcv', number=10, repeats=5)
# 10-fold-cross validation을 5번 반복하여 가장 좋게 평가된것을 후보채택방법인 fitControl객체로 저장

custom_grid = expand.grid(mtry=1:10)
age_rf = train(Age ~., data=age_traindf_train, method='rf', trControl=fitControl, tuneGrid=custom_grid)
# mtry = trControl
age_rf

predict_age = round(predict(age_rf, newdata=age_traindf_test))
age_traindf_test$Age = predict_age

train = rbind(age_traindf_train, age_traindf_test)

colSums(is.na(train)) # 결측값 변경 확인


predict_age = predict(age_rf, newdata=age_testdf_test)
age_testdf_test$Age = round(predict_age)

test = rbind(age_testdf_train, age_testdf_test)

colSums(is.na(test)) # 결측값 변경 확인




######################### 시각화 #########################
# 생존과 성별
sex1=ggplot(train)+
  geom_mosaic(aes(x=product(Sex), fill=Survived)) + # y변수를 범주형으로 설정해줘야 한다.
  xlab('Sex') + ylab('Survived')

sex2=ggplot(train, aes(Sex, fill=Survived))+
  geom_bar(position='dodge')

grid.arrange(sex1,sex2, ncol=2)


# 생존과 class
class1=ggplot(train, aes(Pclass, fill=Survived))+
  geom_bar(position='dodge')

class2=ggplot(train, aes(Pclass, fill=Survived))+
  geom_bar(position='fill')

grid.arrange(class1, class2, ncol=2)


# 생존과 나이
age1= ggplot(train, aes(Age, fill=Survived)) +
  geom_density(alpha=0.5)

age2 = ggplot(train, aes(Age, Survived))+
  geom_jitter(color='lightblue')+
  geom_point(color='violet')

grid.arrange(age1,age2)

# 생존과 Cabin
ggplot(train, aes(Cabin, fill=Survived))+
  geom_bar(position='fill')

# 생존과 요금
fare1 = ggplot(train, aes(Fare))+
  geom_histogram(col='black', fill='lightblue')

fare2 = ggplot(train, aes(Survived, Fare))+
  geom_jitter(col='gray')+ # 관측치를 회색점으로 찍되, 중복되는 부분은 퍼지게 그려준다.
  geom_boxplot(alpha=0.5)

grid.arrange(fare1, fare2, ncol=2) # 그래프 확인결과 요금이 500넘어가는 부분은 이상치 확인

train = filter(train, Fare<500) # 이상치 제거



##################### 분석 및 모델링, 평가 ########################
summary(train)

##### 로지스틱 회귀분석
glm = glm(Survived ~ ., data = train, family = 'binomial')
summary(glm)

anova(glm, test='Chisq') # y가 범주형 변수이므로 카이스퀘어 검정
pR2(glm) # 결정계수 0.37

glm1_p = predict(glm, newdata = train, type='response') # type='response'는 y가 0과 1 인 이산현 변수이므로 지정
glm1_pr = prediction(glm1_p, train$Survived)
glm1_prf = performance(glm1_pr, measure = 'tpr', x.measure = 'fpr')
plot(glm1_prf) # 나쁘지 않다.

glm1_auc = performance(glm1_pr, measure = 'auc')
glm1_auc = glm1_auc@y.values[[1]]
glm1_auc # AUC 판단기준 : great(0.9 ~ 1) / good(0.8 ~ 0.9) / fair(0.7~0.8) / poor(0.6 ~ 0.7) / fail(0.5 ~ 0.6)

glm1_p = as.factor(round(glm1_p))
confusionMatrix(glm1_p, train$Survived)
# Accuracy(정분류율): 0.83(정확하게 예측한 비율)
# Presicion(정확도): 
# Sensitivity(민감도): 0.87(참인것 중 참으로 예측한 비율)
# Specificity(특이도): 0.75(거짓인 것 중 거짓으로 예측한 비율)

"""
glm2 = glm(Survived ~ Pclass + Sex + Age + Cabin + Embarked + FamiliySize, data=train, family='binomial')
glm3 = glm(Survived ~ Pclass + Sex + Age + Embarked + FamiliySize, data=train, family='binomial')

anova(glm, glm2, glm3, test='Chisq') # glm3로 하는게 가장 좋겠다.
pR2(glm3) # 결정계수 0.36

glm3_p = predict(glm3, newdata = train, type='response')
glm3_pr = prediction(glm3_p, train$Survived)
glm3_prf = performance(glm3_pr, measure = 'tpr', x.measure = 'fpr')

plot(glm_prf)
plot(glm3_prf, add=T, colorize=T) # 별 차이 없다.

glm3_auc = performance(glm3_pr, measure = 'auc')
glm3_auc = glm3_auc@y.values[[1]]
glm3_auc

glm3_p = as.factor(round(glm3_p))
confusionMatrix(glm3_p, train$Survived) # 오히려 정분류율이 떨어짐...ㅠ 이건 쓰지말자
"""

glm_p = round(predict(glm, newdata = test, type='response'))

glm_p = as.factor(glm_p)
result$Survived = as.factor(result$Survived)

confusionMatrix(glm_p, result$Survived) # 정분류율 54%..



##### SVM(서포트 벡터 머신)
svm_tune = tune.svm(Survived~., data=train, gamma = 2^(-5:2), cost = 2^(-3:5), kernel='radial')
svm_tune$best.parameters
summary(svm_tune)

plot(svm_tune, cex.main=0.6, cex.lab=0.8, xaxs='i', yaxs='i') # 파란색이 짙은 영역이 최적의 성능을 나타내는 파라미터값을 보여준다.

svm_model = svm(Survived~., data=train, gamma=0.125, cost=8, kernel='radial')
summary(svm_model)

svm_model1_p = predict(svm_model, newdata=train, type='response')

confusionMatrix(svm_model1_p, train$Survived)


svm_predict = predict(svm_model, newdata = test, type='response')


