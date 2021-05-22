import os, time, shap
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE

def MakeTotalData() :
    data_list = []
    file_list = os.listdir('train/')
    file_list.sort()
    print(file_list)

    for file in file_list:
        temp_data = pd.read_csv('train/' + file, sep=',', index_col=0)
        data_list.append(temp_data)

    for data in data_list :
        print(data['period'])
    total_data = pd.concat(data_list).reset_index()
    total_data = total_data.drop(['index'], axis=1)
    total_data.to_csv("total_data.csv", mode="w")

def MakeTrainData(data) :
    # 불필요한 features 제거
    data.drop(['kind', 'color', 'number', 'period','location','feature','center','department'], axis=1, inplace=True)


    # 결과값에서 사망에 중점을 두기 위해 공고중/보호중/완료(귀가)/종료(기증)/종료(방사) 제거
    data = data[data.state != "공고중"]
    data = data[data.state != "보호중"]
    data = data[data.state != "완료(귀가)"]
    data = data[data.state != "종료(기증)"]
    data = data[data.state != "종료(방사)"]


    # feature 중 weight 에 NAN 값 제거
    nan_idx = data[np.isnan(data['weight'])].index.to_numpy()
    data.drop(index=nan_idx, inplace=True)


    #feature 중 noise 값 제거
    data = data[data.sex != '특이']
    data = data[data.sex != '체적']
    data = data[data.weight < 100]


    # feature 중 birth 에서 age 값으로 변경
    data['age'] = 2022 - data['birth']
    data.drop(['birth'], axis=1, inplace=True)


    # data index 정리
    data.reset_index(inplace=True)
    data.drop(['index'], axis=1, inplace=True)


    # feature 중 state 요소 categorize
    # 사망 : 0
    # 입양 : 1
    data.loc[data[data.state == '종료(안락사)'].index.to_numpy(), 'state'] = 0
    data.loc[data[data.state == '종료(자연사)'].index.to_numpy(), 'state'] = 0
    data.loc[data[data.state == '완료(입양)'].index.to_numpy(), 'state'] = 1


    # feature 중 sex 요소 categorize
    # 암컷 : 0
    # 수컷 : 1
    data.loc[data[data.sex == '암컷'].index.to_numpy(), 'sex'] = 0
    data.loc[data[data.sex == '수컷'].index.to_numpy(), 'sex'] = 1


    # feature 중 neutralization 요소 categorize
    # OneHotEncoding 방식 사용
    data.loc[data[np.isnan(data.neutralization)].index.to_numpy(), 'neutralization'] = 2

    enc = OneHotEncoder(handle_unknown='ignore')

    enc_df = pd.DataFrame(enc.fit_transform(data[['neutralization']]).toarray())
    enc_df.columns = ['n_false', 'n_true', 'n_null']

    data = data.join(enc_df)
    data.drop(['neutralization'], axis=1, inplace=True)


    # 최근 10% 데이터를 test dataset 으로 분류
    # test data index 정리 및 저장
    test_data = data.iloc[0:int(data.index.to_numpy().size/10)]
    test_data.to_csv('test_data.csv', mode='w')


    # 나머지 90% 데이터를 train and validation dataset 으로 분류
    # train data index 정리 및 저장
    train_data = data.iloc[int(data.index.to_numpy().size/10):]
    train_data = train_data.reset_index()
    train_data = train_data.drop(['index'], axis=1)
    train_data.to_csv('train_data.csv', mode='w')

def model_evaluation(label, predict) :
    cf_matrix = confusion_matrix(label, predict)
    Accuracy = (cf_matrix[0][0] + cf_matrix[1][1]) / sum(sum(cf_matrix))
    Precision = (cf_matrix[1][1]) / (cf_matrix[1][1] + cf_matrix[0][1])
    Recall = cf_matrix[1][1] / (cf_matrix[1][1] + cf_matrix[1][0])
    F1_Score = (2 * Recall * Precision) / (Recall + Precision)

    print("Model Evaluation Result")
    print("Accuracy: ", Accuracy)
    print("Precision: ", Precision)
    print("Recall: ", Recall)
    print("F1-Score: ", F1_Score)

def TrainWithXGBoost() :
    # 초기 데이터 분석
    train_data = pd.read_csv('train/train_data.csv', sep=',', index_col=0)
    accuracy = 0

    # 5-Fold validation
    cv = KFold(n_splits=5, random_state=1, shuffle=True)

    for t, v in cv.split(train_data) :
        t = train_data.loc[t]
        v = train_data.loc[v]

        train_x = t.drop(['state'], axis=1)
        train_x['sex'] = pd.to_numeric(train_x['sex'])
        train_y = t.drop(['sex', 'weight', 'age', 'n_false', 'n_true', 'n_null'], axis=1)

        test_x = v.drop(['state'], axis=1)
        test_x['sex'] = pd.to_numeric(test_x['sex'])
        test_y = v.drop(['sex', 'weight', 'age', 'n_false', 'n_true', 'n_null'], axis=1)

        #train_x.drop(['sex'], axis=1, inplace=True)
        #test_x.drop(['sex'], axis=1, inplace=True)

        xgb_train_data = xgb.DMatrix(data=train_x, label=train_y)
        xgb_test_data = xgb.DMatrix(data=test_x)
        xgb_param = {
            'max_depth': 2, # 트리 깊이
            'learning_rate': 0.01, # Step Size
            'objective': 'binary:logistic' #목적 함수
        }

        xgb_model = xgb.train(params=xgb_param, dtrain=xgb_train_data)
        xgb_model_predict = xgb_model.predict(xgb_test_data)

        final_data = []
        for x in xgb_model_predict:
            if (x >= 0.5):
                final_data.append(1)
            else:
                final_data.append(0)


        accuracy += accuracy_score(test_y, final_data) * 100
        print(model_evaluation(test_y, final_data))

    print(accuracy / 5, '%')

    #explainer = shap.TreeExplainer(xgb_model)
    #shap_value = explainer.shap_values(test_x)

    #shap.force_plot(explainer.expected_value, shap_value[0, :], test_x.iloc[0, :],show=False,matplotlib=True)
    #shap.summary_plot(shap_value, test_x).show()
    #shap.summary_plot(shap_value, test_x, plot_type='bar').show()

def TrainWithXGBoost2() :
    train_data = pd.read_csv('train/train_data.csv', sep=',', index_col=0)
    accuracy = 0

    sm = SMOTE(random_state=15, sampling_strategy=0.6)

    # 5-Fold validation
    cv = KFold(n_splits=5, random_state=1, shuffle=True)

    for t, v in cv.split(train_data):
        t = train_data.loc[t]
        v = train_data.loc[v]

        train_x = t.drop(['state'], axis=1)
        train_x['sex'] = pd.to_numeric(train_x['sex'])
        train_y = t.drop(['sex', 'weight', 'age', 'n_false', 'n_true', 'n_null'], axis=1)

        #train_x, train_y = sm.fit_resample(train_x, train_y)

        test_x = v.drop(['state'], axis=1)
        test_x['sex'] = pd.to_numeric(test_x['sex'])
        test_y = v.drop(['sex', 'weight', 'age', 'n_false', 'n_true', 'n_null'], axis=1)

        # train_x.drop(['sex'], axis=1, inplace=True)
        # test_x.drop(['sex'], axis=1, inplace=True)

        xgb_train_data = xgb.DMatrix(data=train_x, label=train_y)
        xgb_test_data = xgb.DMatrix(data=test_x)
        xgb_param = {
            'max_depth': 2,  # 트리 깊이
            'learning_rate': 0.01,  # Step Size
            'objective': 'binary:logistic'  # 목적 함수
        }

        xgb_model = xgb.train(params=xgb_param, dtrain=xgb_train_data)
        xgb_model_predict = xgb_model.predict(xgb_test_data)

        final_data = []
        for x in xgb_model_predict:
            if (x >= 0.5):
                final_data.append(1)
            else:
                final_data.append(0)

        accuracy += accuracy_score(test_y, final_data) * 100
        print(model_evaluation(test_y, final_data))

    print(accuracy / 5, '%')

def TestWithXGBoost() :
    train_data = pd.read_csv('train/train_data.csv', sep=',', index_col=0)
    accuracy = 0

    t = train_data
    v = pd.read_csv('test_data.csv', sep=',', index_col=0)

    train_x = t.drop(['state'], axis=1)
    train_x['sex'] = pd.to_numeric(train_x['sex'])
    train_y = t.drop(['sex', 'weight', 'age', 'n_false', 'n_true', 'n_null'], axis=1)

    test_x = v.drop(['state'], axis=1)
    test_x['sex'] = pd.to_numeric(test_x['sex'])
    test_y = v.drop(['sex', 'weight', 'age', 'n_false', 'n_true', 'n_null'], axis=1)

    xgb_train_data = xgb.DMatrix(data=train_x, label=train_y)
    xgb_test_data = xgb.DMatrix(data=test_x)
    xgb_param = {
        'max_depth': 5,  # 트리 깊이
        'learning_rate': 0.01,  # Step Size
        'objective': 'binary:logistic'  # 목적 함수
    }

    xgb_model = xgb.train(params=xgb_param, dtrain=xgb_train_data)
    xgb_model_predict = xgb_model.predict(xgb_test_data)

    final_data = []
    for x in xgb_model_predict:
        if (x >= 0.5):
            final_data.append(1)
        else:
            final_data.append(0)

    accuracy += accuracy_score(test_y, final_data) * 100

    print(accuracy, '%')

def TrainDataGraph() :
    train_data = pd.read_csv('train/train_data.csv', sep=',', index_col=0)

    print(train_data['weight'].value_counts())

    fig, ax = plt.subplots(ncols=2)
    sns.relplot("weight", "age", hue='sex', data=train_data[train_data.state == 0], ax=ax[0])
    sns.relplot("weight", "age", hue='sex', data=train_data[train_data.state == 1], ax=ax[1])
    plt.show()

if (__name__ == '__main__') :
    # 초기 모델 검증 과정
    #MakeTrainData(pd.read_csv('total_data.csv', sep=',', index_col=0))
    #TrainWithXGBoost()

    #두번째 모델 검증 과정 (말티즈 종에 대한 분석)
    total_data = pd.read_csv('total_data.csv', sep=',', index_col=0)

    print(total_data['kind'].value_counts())

    '''t_data = total_data.loc[total_data[total_data['kind'] == '[개] 말티즈'].index.to_numpy()]
    MakeTrainData(t_data)
    TrainWithXGBoost2()'''