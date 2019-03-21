library(e1071)

rm(list=ls())

house = read.csv('file:///D:/python_local_repository/03. AI/1. Machine Learning/SVM/hosing/Housing.csv', header=T)
# str(house)

# 딱 봐도 필요없어 보이는 X열 제거
house = subset(house, select = -X)
# str(house)


# attach(house) # house 의 변수를 자유롭게 쓰기위해서 attach로 접근하는게 좋네. 불편한거 해결쓰

fianl_df = data.frame(5, 10, 20)



# 학습데이터와 검증데이터 셋 분리
train_idx = sample(1:513, 300)

train_df = house[train_idx,]
test_df = house[-train_idx,]


# 조절인자 튜닝
svm_tune = tune.svm(price ~., data=train_df, gamma = 2^(-2:2), cost = 2^(-1:4), kernel='radial')
#svm_tune$best.parameters


# 튜닝한 결과 확인 후 조절인자 조정 하고 svm모델 생성
svm_after_tune = svm(price ~ ., data=train_df, gamma=as.numeric(svm_tune$best.parameters['gamma']), cost=as.numeric(svm_tune$best.parameters['cost']), kernel='radial')
#summary(svm_after_tune)


# 서포트 벡터 확인( 몇번째 관찰값이 서포트 벡터인지 확인하는거)/ 뭔말?
#svm_after_tune$index


# 만든 svm모델로 test 데이터셋 적용 검증
svm_predict = predict(svm_after_tune, newdata = test_df)

# predict(object, newdata, decision.values = FALSE, probability = FALSE, ..., na.action = na.omit)

# object :Object of class "svm", created by svm.
# newdata :An object containing the new input data

# p <- predict(m, test, type = "response")

#type 옵션은 response, probabilities, votes 중 선택할 수 있다.
#response는 default값으로, 반응값이 그대로 반환되며
#probabilities는 모델 알고리즘에 따라 확률로 반환된다.


########################### 결과 예측 및 확인 ###############################

svm_predict = as.data.frame(svm_predict) # 데이터 프레임으로 변환

predict_5 = cbind(svm_predict, real_lwr=as.numeric(test_df$price)*0.95, real_up=as.numeric(test_df$price)*1.05)
predict_10 = cbind(svm_predict, real_lwr=as.numeric(test_df$price)*0.9, real_up=as.numeric(test_df$price)*1.1)
predict_20 = cbind(svm_predict, real_lwr=as.numeric(test_df$price)*0.8, real_up=as.numeric(test_df$price)*1.2)

# 가격의 +- 범위 값의 데이터 프레임만들기


tf= 0 # 정답률 구하기 위한거,

result_5 = cbind(predict_5, tf) 
result_10 = cbind(predict_10, tf)
result_20 = cbind(predict_20, tf)

result_5 = as.data.frame(result_5)
result_10 = as.data.frame(result_10)
result_20 = as.data.frame(result_20)
  
result_5$tf[result_5$svm_predict >= result_5$real_lwr & result_5$svm_predict <= result_5$real_up] = 1
result_10$tf[result_10$svm_predict >= result_10$real_lwr & result_10$svm_predict <= result_10$real_up] = 1
result_20$tf[result_20$svm_predict >= result_20$real_lwr & result_20$svm_predict <= result_20$real_up] = 1



sprintf('가격의 5%% 범위 정답률: %s%%', round(sum(result_5$tf)/dim(result_5)[1]*100, 2))
sprintf('가격의 10%% 범위 정답률: %s%%', round(sum(result_10$tf)/dim(result_10)[1]*100, 2))
sprintf('가격의 20%% 범위 정답률: %s%%', round(sum(result_20$tf)/dim(result_20)[1]*100, 2))
  
