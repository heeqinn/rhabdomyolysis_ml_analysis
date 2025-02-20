# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 09:26:01 2025

@author: Dell
"""

# %% 导入包和数据
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import (train_test_split,
                                     StratifiedKFold,
                                     GridSearchCV,
                                     cross_val_score,
                                     KFold)
from sklearn.ensemble import (RandomForestClassifier as RF,
                              AdaBoostClassifier as ADA,
                              GradientBoostingClassifier as GBDT,
                              ExtraTreesClassifier as ET)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
#from sklearn.cross_validation import KFold
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import RFE,RFECV
import xgboost as xgb
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score,roc_auc_score,accuracy_score,f1_score,roc_curve
from sklearn.metrics import confusion_matrix
from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split 
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score,roc_auc_score,accuracy_score,f1_score,roc_curve,auc
from skopt import BayesSearchCV
from sklearn.preprocessing import StandardScaler
import shap
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set(font='SimHei',font_scale=0.8)

#data = pd.read_excel("G:\\1-项目\\5-数据服务\\23 肾脏病医学部 张利\\肌酸激酶\\3-1-肌酸激酶数据清洗补充.xlsx")


# %% 异常值处理
#四分位距方法识别异常数据
#lower_bound = Q1 - 1.5 * IQR
#upper_bound = Q3 + 1.5 * IQR
#对异常数据进行Winsorizing处理，将5%的最小值和5%的最大值替换为第5百分位数和第95百分位数的值。

# 对数值型数据进行异常值检测

def detect_outliers_iqr(df, column):
    """基于四分位数计算给定列的异常值"""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    return lower_bound, upper_bound

def winsorize_column(df, column, limits=(0.05, 0.05)):
    """Winsorize DataFrame中的指定列以处理异常值"""
    # 检查数据类型是否为数值型
    if not np.issubdtype(df[column].dtype, np.float64):
        print(f"Column {column} is not numeric, skipping.")
        return df
    if column in important_future:
        return df
    # 使用winsorize方法，这里我们假设上下各5%的数据为异常值
    df[column] = winsorize(df[column], limits=limits)
    
    # 检查结果是否全部为NaN
    if df[column].isna().all():
        print(f"Warning: After winsorizing, column {column} contains only NaN values.")
    
    return df

def process_dataframe(df, missing_threshold=0.5, limits=(0.05, 0.05)):
    """对DataFrame中所有数值列进行异常值检测和Winsorization，跳过缺失值比例过高的列"""
    numeric_columns = df.select_dtypes(include=[np.float64]).columns.tolist()
    
    for column in numeric_columns:
        # 计算缺失值比例
        missing_ratio = df[column].isna().mean()
        print(f"Column {column} missing ratio: {missing_ratio:.2f}")
        
        if missing_ratio > missing_threshold:
            print(f"Skipping column {column} due to high missing value ratio.")
            continue
        
        print(f"Processing column: {column}")
        
        # 检测异常值
        lower_bound, upper_bound = detect_outliers_iqr(df, column)

        # 替换异常值
        df = winsorize_column(df.copy(), column, limits)
    
    return df

# 设置缺失值比例阈值，例如50%
missing_value_threshold = 0.5

important_future = ['肌酸激酶_min','乳酸脱氢酶_min','肌酐_min',
                    '肌酸激酶_max','乳酸脱氢酶_max','肌酐_max','肌酸激酶_last','肌酐_last']
# 处理整个DataFrame
df_winsorized = process_dataframe(data.copy(), missing_threshold=missing_value_threshold)

# %% 缺失值插补
#随机森林回归预测缺失值
miss_col = df_winsorized.isnull().any()
miss_col = miss_col.reset_index()
miss_col.columns = ['col','miss']

qs = miss_col[miss_col.miss == True].col
data_miss = df_winsorized[qs]


notqs = [item for item in df_winsorized.columns if item not in set(qs)]

sortindex = np.argsort(data_miss.isnull().sum(axis=0)).values
Y1 = df_winsorized['死亡']
 
for i in sortindex:
    df = data_miss
    fillc = df.iloc[:,i]
    df = pd.concat([df.iloc[:,df.columns != i],pd.DataFrame(Y1)],axis=1)
    df_0 = SimpleImputer(missing_values=np.nan,strategy='constant',fill_value=0).fit_transform(df)
    ytrain = fillc[fillc.notnull()]
    ytest = fillc[fillc.isnull()]
    xtrain = df_0[ytrain.index,:]
    xtest = df_0[ytest.index,:]
    
    rfc = RandomForestRegressor(n_estimators=100)
    rfc = rfc.fit(xtrain,ytrain)
    y_predict = rfc.predict(xtest)
    
    #data_miss[data_miss.iloc[:,i].isnull(),i] = y_predict
    data_miss.iloc[data_miss.iloc[:,i][data_miss.iloc[:,i].isnull().values==True].index,i] = y_predict


new_data = pd.concat([df_winsorized[notqs],data_miss],axis=1) #插补后数据 拼接 无缺失数据

#new_data.to_excel("H:\\1 项目\\5-数据服务\\23 肾脏病医学部 张利\\肌酸激酶\\4-new_data2.xlsx",index=False)
#输出4-new_data3

# %% 结局：住院时长
new_data = pd.read_excel("G:\\1-项目\\5-数据服务\\23 肾脏病医学部 张利\\肌酸激酶\\4-new_data3.xlsx",sheet_name='Sheet2')
#new_data = pd.read_excel("I:\\1-项目\\5-数据服务\\23 肾脏病医学部 张利\\肌酸激酶\\4-new_data3.xlsx",sheet_name='Sheet2')

X = new_data.drop(['length of hospital stay >7 (days)','length of hospital stay (days)','PATIENT_ID','VISIT_ID','NAME','height','weight'],axis=1) 
Y = new_data['length of hospital stay >7 (days)']

#test是测试集
X1_train,X1_test,Y1_train,Y1_test = train_test_split(X,Y,test_size = 0.2,random_state=1)
#train1是训练，verify是验证
X1_train1,X_verify,Y1_train1,Y_verify = train_test_split(X1_train,Y1_train,test_size = 0.25,random_state=1)
#6:2:2

# %% 结局：死亡
X = new_data.drop(['length of hospital stay >7 (days)','death','PATIENT_ID','VISIT_ID','NAME','height','weight'],axis=1) 
Y = new_data['death']

#欠采样
from imblearn.under_sampling import ClusterCentroids
from collections import Counter
cc = ClusterCentroids(random_state=0)
X_resampled, y_resampled = cc.fit_resample(X, Y)
print(sorted(Counter(y_resampled).items()))    #[(0, 586), (1, 586)]
    
#test是测试集
X1_train,X1_test,Y1_train,Y1_test = train_test_split(X_resampled,y_resampled,test_size = 0.2,random_state=1)
#train1是训练，verify是验证
X1_train1,X_verify,Y1_train1,Y_verify = train_test_split(X1_train,Y1_train,test_size = 0.25,random_state=1)
#6:2:2

# %% 结局：肾损伤
X = new_data.drop(['length of hospital stay >7 (days)','kidney injury','PATIENT_ID','VISIT_ID','NAME','height','weight'],axis=1) #有bmi就drop身高体重了
#data.iloc[:,:-1]
Y = new_data['kidney injury']
    
  
#欠采样
from imblearn.under_sampling import ClusterCentroids
from collections import Counter
cc = ClusterCentroids(random_state=0)
X_resampled, y_resampled = cc.fit_resample(X, Y)
print(sorted(Counter(y_resampled).items()))    #[(0, 283), (1, 283)]
    
#test是测试集
X1_train,X1_test,Y1_train,Y1_test = train_test_split(X_resampled,y_resampled,test_size = 0.2,random_state=1)
#train1是训练，verify是验证
X1_train1,X_verify,Y1_train1,Y_verify = train_test_split(X1_train,Y1_train,test_size = 0.25,random_state=1)
#6:2:2

# %% 结局为血液透析
#17

X = new_data.drop(['length of hospital stay >7 (days)','HD','hemodialysis count','PATIENT_ID','VISIT_ID','NAME','height','weight'],axis=1) #有bmi就drop身高体重了
#data.iloc[:,:-1]
Y = new_data['HD']
    

#组合采样
#SMOTETomek 和SMOTEENN
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


over_values = [0.3,0.4,0.5]
under_values = [0.7,0.6,0.5]
for o in over_values:
  for u in under_values:
    # define pipeline
    model = SVC()
    over = SMOTE(sampling_strategy=o)
    under = RandomUnderSampler(sampling_strategy=u)
    steps = [('over', over), ('under', under), ('model', model)]
    pipeline = Pipeline(steps=steps)
    # evaluate pipeline
    scores = cross_val_score(pipeline, X, Y, scoring='roc_auc', cv=5, n_jobs=-1)
    score = np.mean(scores)
    print('SMOTE oversampling rate:%.1f, Random undersampling rate:%.1f , Mean ROC AUC: %.3f' % (o, u, score))




over = SMOTE(sampling_strategy=0.3,random_state=25)
under = RandomUnderSampler(sampling_strategy=0.6,random_state=25)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)
X_resampled, y_resampled = pipeline.fit_resample(X, Y)

print(sorted(Counter(y_resampled).items()))  #[(0, 10141), (1, 6085)]  [(0, 8692), (1, 6085)]

    
#test是测试集
X1_train,X1_test,Y1_train,Y1_test = train_test_split(X_resampled,y_resampled,test_size = 0.2,random_state=1)
#train1是训练，verify是验证
X1_train1,X_verify,Y1_train1,Y_verify = train_test_split(X1_train,Y1_train,test_size = 0.25,random_state=1)
#6:2:2

# %% 结局为心脏损伤
X = new_data.drop(['length of hospital stay >7 (days)','cardiac injury','PATIENT_ID','VISIT_ID','NAME','height','weight'],axis=1) #有bmi就drop身高体重了
#data.iloc[:,:-1]
Y = new_data['cardiac injury']

#test是测试集
X1_train,X1_test,Y1_train,Y1_test = train_test_split(X,Y,test_size = 0.2,random_state=1)
#train1是训练，verify是验证
X1_train1,X_verify,Y1_train1,Y_verify = train_test_split(X1_train,Y1_train,test_size = 0.25,random_state=1)
#6:2:2

# %% 结局为肺损伤
#肺3584 心脏16751

X = new_data.drop(['length of hospital stay >7 (days)','lung injury','PATIENT_ID','VISIT_ID','NAME','height','weight'],axis=1) #有bmi就drop身高体重了
#data.iloc[:,:-1]
Y = new_data['lung injury']

#组合采样
#SMOTETomek 和SMOTEENN
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import RandomUnderSampler,ClusterCentroids
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE

from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
# =============================================================================
# X_resampled, y_resampled = ADASYN().fit_resample(X, Y)
# print(sorted(Counter(y_resampled).items()))
# =============================================================================
over_values = [0.3,0.4,0.5]
under_values = [0.7,0.6,0.5]
for o in over_values:
  for u in under_values:
    # define pipeline
    model = SVC()
    over = SMOTE(sampling_strategy=o)
    under = RandomUnderSampler(sampling_strategy=u)
    steps = [('over', over), ('under', under), ('model', model)]
    pipeline = Pipeline(steps=steps)
    # evaluate pipeline
    scores = cross_val_score(pipeline, X, Y, scoring='roc_auc', cv=5, n_jobs=-1)
    score = np.mean(scores)
    print('SMOTE oversampling rate:%.1f, Random undersampling rate:%.1f , Mean ROC AUC: %.3f' % (o, u, score))

#SMOTE oversampling rate:0.5, Random undersampling rate:0.5 , Mean ROC AUC: 0.673

over = SMOTE(sampling_strategy=0.5,random_state=25)
under = RandomUnderSampler(sampling_strategy=0.5,random_state=25)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)
X_resampled, y_resampled = pipeline.fit_resample(X, Y)

    
#test是测试集
X1_train,X1_test,Y1_train,Y1_test = train_test_split(X_resampled, y_resampled,test_size = 0.2,random_state=1)
#train1是训练，verify是验证
X1_train1,X_verify,Y1_train1,Y_verify = train_test_split(X1_train,Y1_train,test_size = 0.25,random_state=1)
#6:2:2
 


# %% RFECV
#消耗资源较大，后期不再重复运行
estimator = LinearSVC()
#rf = RF(random_state=30)
selector = RFECV(estimator = estimator ,step = 1 , cv = 3 ,scoring = 'accuracy') #StratifiedKFold(3)
selector.fit(X1_train1,Y1_train1)

print("N_features %s" % selector.n_features_)
print("Support is %s" % selector.support_)
print("Ranking %s" % selector.ranking_)
#print("Grid Scores %s" % selector.grid_scores_)
print("Grid Scores %s" % selector.cv_results_)
#---------------------------------------------------------
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# 初始化分类器
clf = RandomForestClassifier(random_state=42)

# 初始化RFE
rfe = RFE(estimator=clf, n_features_to_select=1)

# 拟合数据
rfe.fit(X1_train1, Y1_train1)

# 获取每个特征的重要性排名
ranking = rfe.ranking_

# 使用交叉验证评估不同特征数量的性能
scores = []
for n in range(1, X.shape[1] + 1):
    rfe.n_features_to_select = n
    selected_features = rfe.fit_transform(X1_train1, Y1_train1)
    score = np.mean(cross_val_score(clf, selected_features, Y1_train1, cv=5))
    scores.append(score)
    print(f"Number of features: {n}, Cross-validation score: {score:.4f}")

# 找到最佳特征数量
optimal_n_features = np.argmax(scores) + 1
print(f"Optimal number of features: {optimal_n_features}")


# =============================================================================
# #绘制图表展示特征数量与交叉验证得分的关系
# plt.figure()
# plt.xlabel('number of features selected')
# plt.ylabel('cross validation score')
# 
# a = range(1,len(selector.cv_results_) +1)
# b = selector.cv_results_
# 
# plt.figure(figsize=(15,10))
# plt.figure(dpi=1000)
# plt.plot(a, b)
# for x,y in zip(a,b):
#     plt.text(x-0.3,y+0.001,str(x),ha='center',va = 'bottom' , fontsize=5)
# plt.show()
# =============================================================================

#N_features 40 住院时长
#N_features 44 死亡
#N_features 21 肾损伤
#N_features 48 血液透析
#N_features 85 心脏损伤
#N_features 87 肺损伤
# %% rfe筛选变量
#消耗资源较大，后期不再重复运行
rf = RF(random_state=30)
rfe = RFE(estimator = rf , n_features_to_select = 48 ) #筛选xx变量
rfe.fit(X1_train1,Y1_train1)
rfe_ranking = rfe.ranking_
rfe_importances = pd.DataFrame({'rank':rfe_ranking,'var':X1_train1.columns})
rfe_importances = rfe_importances.sort_values(by='rank',ascending=True)
rfe_top_59 = rfe_importances.loc[rfe_importances['rank']==1]['var'].tolist()


#住院时长变量
#rfe_top_40 = ['CK-MB-DL_min', 'Cr_last', 'CK_min', 'K_min', 'UA_min', 'PLT_max', 'Hct_max', 'AST_min', 'ALT_min', 'TP_min', 'Alb_min', 'Hb_min', 'WBC_min', 'LDH_min', 'PLT_min', 'K_max', 'P_max', 'WBC_max', 'AST_max', 'ALT_max', 'CK-MB_min', 'TP_max', 'Alb_max', 'TnI_min', 'CK_max', 'LDH_max', 'Cr_max', 'TnT_min', 'urine protein_max', 'urine volume_max', 'urine protein_min', 'Pulse_max', 'temperature_max', 'Cr_min', 'blood pressure high_max', 'CK-MB-DL_max', 'CK-MB_max', 'NT-proBNP_min', 'CK_last', 'TnI_max']
#死亡变量
#rfe_top_44 = ['age', 'Mb_max', 'LDH_max', 'CK_max', 'urine protein_min', 'Cr_min', 'MB_min', 'Cr_max', 'LDH_min', 'PaCO2_max', 'Hb_max', 'TP_max', 'ALT_max', 'AST_max', 'P_max', 'bicarbonate count', 'TT_max', 'PT_max', 'urine protein_max', 'PaO2_max', 'SaO2_min', 'PaCO2_min', 'PaO2_min', 'NT-proBNP_max', 'TnI_max', 'TnT_max', 'CK-MB-DL_max', 'NT-proBNP_min', 'TnT_min', 'CK-MB-DL_min', 'CK-MB_min', 'Cr_last', 'CK_last', 'urine volume_max', 'UA_max', 'K_max', 'SaO2_max', 'CK_min', 'gender', 'Alb_min', 'Pulse_max', 'TP_min', 'respiration_max', 'AST_min']
#肾损伤变量
#rfe_top_21 = ['SaO2_max', 'PLT_max', 'bicarbonate', 'ALT_min', 'urine volume_min', 'CK_min', 'CK_max', 'Hct_max', 'Cr_max', 'CK_last', 'CK-MB_min', 'CK-MB-DL_min', 'Mb_max', 'CK-MB_max', 'gender', 'SaO2_min', 'PaO2_min', 'TnI_max', 'Hb_max', 'blood pressure low_first', 'CK-MB-DL_max']
#心脏损伤变量
#rfe_top_85 = ['age', 'LDH_max', 'CK_max', 'urine volume_min', 'urine protein_min', 'APTT_min', 'FIB_min', 'PT_min', 'TT_min', 'Cr_min', 'Mb_max', 'MB_min', 'blood oxygen_max', 'SPO2_first', 'hypertension', 'bicarbonate count', 'BT', 'blood transfusion count', 'PaCO2_max', 'mannitol', 'mannitol count', 'LDH_min', 'PLT_max', 'Cr_max', 'PT_max', 'PaO2_max', 'SaO2_min', 'PaCO2_min', 'PaO2_min', 'NT-proBNP_max', 'TnI_max', 'TnT_max', 'CK-MB-DL_max', 'CK-MB_max', 'TT_max', 'NT-proBNP_min', 'TnT_min', 'CK-MB-DL_min', 'CK-MB_min', 'Cr_last', 'CK_last', 'urine volume_max', 'urine protein_max', 'APTT_max', 'FIB_max', 'TnI_min', 'Hct_max', 'SaO2_max', 'Hb_max', 'ALT_min', 'AST_min', 'WBC_max', 'P_min', 'UA_min', 'K_min', 'CK_min', 'history of hypertension', 'blood pressure high_max', 'blood pressure low_max', 'temperature_max', 'respiration_max', 'Pulse_max', 'blood pressure high_first', 'blood pressure low_first', 'temperature_first', 'respiration_first', 'pulse_first', 'BMI', 'length of hospital stay (days)', 'gender', 'TP_min', 'Alb_min', 'Ca_min', 'WBC_min', 'Alb_max', 'TP_max', 'ALT_max', 'AST_max', 'Ca_max', 'P_max', 'UA_max', 'Hb_min', 'K_max', 'PLT_min', 'Hct_min']
#肺损伤变量
#rfe_top_87 = ['age', 'CK_max', 'urine volume_min', 'urine protein_min', 'APTT_min', 'FIB_min', 'PT_min', 'TT_min', 'Cr_min', 'MB_min', 'LDH_max', 'LDH_min', 'SPO2_first', 'hypertension', 'bicarbonate', 'bicarbonate count', 'BT', 'blood transfusion count', 'PaCO2_max', 'mannitol', 'mannitol count', 'blood oxygen_max', 'Mb_max', 'Cr_max', 'TT_max', 'PaO2_max', 'SaO2_min', 'PaCO2_min', 'PaO2_min', 'NT-proBNP_max', 'TnI_max', 'TnT_max', 'CK-MB-DL_max', 'CK-MB_max', 'NT-proBNP_min', 'TnI_min', 'TnT_min', 'CK-MB-DL_min', 'CK-MB_min', 'Cr_last', 'CK_last', 'urine volume_max', 'urine protein_max', 'APTT_max', 'FIB_max', 'PT_max', 'PLT_max', 'Hct_max', 'SaO2_max', 'Hb_max', 'AST_min', 'Ca_min', 'WBC_max', 'UA_min', 'K_min', 'CK_min', 'history of diabetes', 'history of hypertension', 'blood pressure high_max', 'blood pressure low_max', 'temperature_max', 'respiration_max', 'Pulse_max', 'blood pressure high_first', 'blood pressure low_first', 'temperature_first', 'respiration_first', 'pulse_first', 'BMI', 'length of hospital stay (days)', 'gender', 'ALT_min', 'TP_min', 'P_min', 'Hb_min', 'WBC_min', 'Hct_min', 'PLT_min', 'Alb_max', 'Alb_min', 'TP_max', 'UA_max', 'P_max', 'K_max', 'Ca_max', 'ALT_max', 'AST_max']
#血液透析变量
#rfe_top_48 = ['age', 'P_min', 'Ca_min', 'urine volume_max', 'urine protein_max', 'APTT_max', 'FIB_max', 'TT_max', 'Hct_min', 'Cr_max', 'K_max', 'UA_min', 'UA_max', 'Mb_max', 'ALT_max', 'LDH_max', 'CK_max', 'urine protein_min', 'APTT_min', 'FIB_min', 'Cr_min', 'MB_min', 'blood oxygen_max', 'P_max', 'CK_min', 'Cr_last', 'CK-MB-DL_min', 'PaO2_max', 'length of hospital stay (days)', 'SaO2_min', 'kidney injury', 'PaO2_min', 'BMI', 'pulse_first', 'NT-proBNP_max', 'temperature_first', 'CK-MB_min', 'Pulse_max', 'TnI_max', 'temperature_max', 'TnT_min', 'NT-proBNP_min', 'respiration_max', 'CK-MB_max', 'bicarbonate count', 'CK-MB-DL_max', 'TnT_max', 'history of hypertension']

X1_train_select = X1_train1[rfe_top_21]
X_verify_select = X_verify[rfe_top_21]
X1_test1 = X1_test[rfe_top_21]

# %% xgboost贝叶斯优化调参
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')  # 设置 use_label_encoder=False 并提供 eval_metric 以避免警告

# 定义要优化的超参数空间
param_space = {
    'n_estimators': (50, 200),        # 树的数量
    'max_depth': (3, 9),              # 树的最大深度
    'learning_rate': (0.01, 0.2, 'log-uniform'),  # 学习率，'log-uniform' 表示在对数尺度上均匀分布
    'subsample': (0.8, 1.0),          # 训练每棵树时使用的样本比例
    'colsample_bytree': (0.8, 1.0)    # 构建树时使用的特征比例
}

# 初始化BayesSearchCV
bayes_search = BayesSearchCV(
    estimator=xgb,
    search_spaces=param_space,
    n_iter=32,                        # 进行32次迭代
    scoring='accuracy',               # 使用准确率作为评分标准
    cv=5,                             # 5折交叉验证
    n_jobs=-1,                        # 使用所有可用的核心
    verbose=1,
    random_state=42                   # 随机种子，确保结果可复现
)

# 拟合模型
bayes_search.fit(X1_train_select ,Y1_train1)

# 输出最佳参数和对应的准确率
print("Best parameters found: ", bayes_search.best_params_)
print("Highest accuracy found: %.2f%%" % (bayes_search.best_score_ * 100))

# 住院时长===========================================================================
# Best parameters found:  OrderedDict([('colsample_bytree', 0.8820207917706628), 
#                                      ('learning_rate', 0.08846938749167613), 
#                                      ('max_depth', 9), ('n_estimators', 97), 
#                                      ('subsample', 0.9340295896537869)])
# Highest accuracy found: 87.90%
# =============================================================================

# 死亡=============================================================================
# Best parameters found:  OrderedDict([('colsample_bytree', 0.8), 
#                                      ('learning_rate', 0.0928759783312699), 
#                                      ('max_depth', 4), ('n_estimators', 180), 
#                                      ('subsample', 0.8042951235888489)])
# =============================================================================

# 肾损伤=============================================================================
# Best parameters found:  OrderedDict([('colsample_bytree', 0.9884635880361323), 
#                                      ('learning_rate', 0.024287034614086097), 
#                                      ('max_depth', 6), ('n_estimators', 200), 
#                                      ('subsample', 0.831081304159863)])
# Highest accuracy found: 92.92%
# =============================================================================

# 心脏损伤=============================================================================
# Best parameters found:  OrderedDict([('colsample_bytree', 0.9115429791959578), 
#                                      ('learning_rate', 0.03117540397164463), 
#                                      ('max_depth', 9), ('n_estimators', 147), 
#                                      ('subsample', 0.948154004921218)])
# Highest accuracy found: 99.87%
# =============================================================================

# 肺损伤=============================================================================
# Best parameters found:  OrderedDict([('colsample_bytree', 1.0), 
#                                      ('learning_rate', 0.2), 
#                                      ('max_depth', 5), ('n_estimators', 200), 
#                                      ('subsample', 0.8)])
# Highest accuracy found: 98.25%
# =============================================================================

# 偷袭=============================================================================
# Best parameters found:  OrderedDict([('colsample_bytree', 0.8224292759924623), 
#                                      ('learning_rate', 0.048144174253709246), 
#                                      ('max_depth', 9), ('n_estimators', 50), 
#                                      ('subsample', 0.8355121389859869)])
# Highest accuracy found: 99.89%
# =============================================================================


# %% 计算结果函数
def evaluate(clf,x_verify,y_verify,x_test,y_test):  #算法预测结果
    # 使用最佳参数对验证集进行预测并评估性能
    #best_model = bayes_search.best_estimator_
    predictions = clf.predict(x_verify)
    y_pred_proba = clf.predict_proba(x_verify)[:, 1]  # 获取正类的概率
    
    print('验证集')
    auc_roc = roc_auc_score(y_verify, y_pred_proba)
    print(f"ROC AUC Score: {auc_roc:.4f}")
    precision = precision_score(y_verify, predictions, average='weighted')  # 对于多类别问题使用 'weighted'
    print(f"Precision: {precision:.4f}")
    accuracy = accuracy_score(y_verify, predictions)
    print(f"Accuracy: {accuracy:.4f}")
    #print("Accuracy on the test set with best params: %.2f%%" % (accuracy * 100))
    f1 = f1_score(y_verify, predictions)
    print(f"F1 Score: {f1:.4f}")

    # 使用最佳参数对测试集进行预测并评估性能
    print('测试集')
    predictions = clf.predict(x_test)
    y_pred_proba = clf.predict_proba(x_test)[:, 1]  # 获取正类的概率

    auc_roc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC Score: {auc_roc:.4f}")
    precision = precision_score(y_test, predictions, average='weighted')  # 对于多类别问题使用 'weighted'
    print(f"Precision: {precision:.4f}")
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.4f}")
    #print("Accuracy on the test set with best params: %.2f%%" % (accuracy * 100))
    f1 = f1_score(y_test, predictions)
    print(f"F1 Score: {f1:.4f}")
    
#%% shap图函数
#保留特征筛选数据&模型输出后，增加了shap需求。此处直接导入特征筛选后数据。

def plot_shap(clf,x_train,y_train,x_test):

    explainer = shap.TreeExplainer(clf.fit(x_train,y_train))
    #输出numpy.array数组
    shap_values = explainer.shap_values(x_test)
    

    #整体柱状图
    plt.clf()
    shap.summary_plot(shap_values, x_test, plot_type= 'bar',show=False)
    plt.savefig('C:\\Users\\Dell\\Desktop\\bar.png',dpi=300)
    #整体
    plt.clf()
    shap.summary_plot(shap_values,x_test,show=False)
    plt.savefig('C:\\Users\\Dell\\Desktop\\summary.png',dpi=300)
    #单个样本
    plt.clf()
    plt.figure(figsize=(90, 30))
    shap.force_plot(explainer.expected_value, shap_values[1,:],x_test.iloc[1,:],matplotlib=True,show=False)
    plt.savefig('C:\\Users\\Dell\\Desktop\\force.png',dpi=300)
    
    #交互图
    plt.clf()
    shap_interaction_values = explainer.shap_interaction_values(x_test)    
    shap.summary_plot(shap_interaction_values, x_test, max_display=5,show=False)
    plt.savefig('C:\\Users\\Dell\\Desktop\\interaction.png',dpi=300)                                                           


# %% xgboost 最佳参数结果及画图

# 住院时长=============================================================================
# xgb_best = XGBClassifier(
#     max_depth=9,               # 树的最大深度
#     learning_rate=0.08846938749167613,         # 学习率
#     n_estimators=97,          # 树的数量
#     subsample=0.9340295896537869,             # 训练每棵树时所用样本的比例
#     colsample_bytree=0.8820207917706628,      # 训练每棵树时所用特征的比例
#     random_state=42            # 随机种子
# )
# =============================================================================


# =============================================================================
# # 创建 StandardScaler 实例
# scaler = StandardScaler()
# 
# # 计算训练数据的均值和标准差，并对其进行标准化
# X1_train_scaled = scaler.fit_transform(X1_train_select)
# 
# print("Original training data:\n", X1_train_select[:5])
# print("Scaled training data:\n", X1_train_scaled[:5])
# 
# 
# # 使用相同的缩放参数对测试数据进行标准化（注意这里只使用 transform）
# X1_test1_scaled = scaler.transform(X1_test1)
# 
# print("Original test data:\n", X1_test1[:5])
# print("Scaled test data:\n", X1_test1_scaled[:5])
# =============================================================================
# xgb_best.fit(X1_train_scaled,Y1_train1)
# evaluate(xgb_best,X_verify_select,Y_verify,X1_test1_scaled,Y1_test)
# =============================================================================


# 死亡=============================================================================
# xgb_best = XGBClassifier(
#     max_depth=4,               # 树的最大深度
#     learning_rate=0.0928759783312699,         # 学习率
#     n_estimators=180,          # 树的数量
#     subsample=0.8042951235888489,             # 训练每棵树时所用样本的比例
#     colsample_bytree=0.8,      # 训练每棵树时所用特征的比例
#     random_state=42            # 随机种子
# )
# =============================================================================



# 心脏=============================================================================
# xgb_best = XGBClassifier(
#     max_depth=9,               # 树的最大深度
#     learning_rate=0.03117540397164463,         # 学习率
#     n_estimators=147,          # 树的数量
#     subsample=0.948154004921218,             # 训练每棵树时所用样本的比例
#     colsample_bytree=0.9115429791959578,      # 训练每棵树时所用特征的比例
#     random_state=42            # 随机种子
# )
# =============================================================================

# 肺=============================================================================
# xgb_best = XGBClassifier(
#     max_depth=5,               # 树的最大深度
#     learning_rate=0.2,         # 学习率
#     n_estimators=200,          # 树的数量
#     subsample=0.8,             # 训练每棵树时所用样本的比例
#     colsample_bytree=1,      # 训练每棵树时所用特征的比例
#     random_state=42            # 随机种子
# )
# =============================================================================

# 肾=============================================================================
# xgb_best = XGBClassifier(
#     max_depth=6,               # 树的最大深度
#     learning_rate=0.024287034614086097,         # 学习率
#     n_estimators=200,          # 树的数量
#     subsample=0.831081304159863,             # 训练每棵树时所用样本的比例
#     colsample_bytree=0.9884635880361323,      # 训练每棵树时所用特征的比例
#     random_state=42            # 随机种子
# )
# =============================================================================

# =============================================================================
# xgb_best = XGBClassifier(
#     max_depth=8,               # 树的最大深度
#     learning_rate=0.028594790675819944,         # 学习率
#     n_estimators=88,          # 树的数量
#     subsample=0.8716282009604424,             # 训练每棵树时所用样本的比例
#     colsample_bytree=0.8302604393826345,      # 训练每棵树时所用特征的比例
#     random_state=42            # 随机种子
# )
# =============================================================================

#透析 =============================================================================
# xgb_best = XGBClassifier(
#     max_depth=9,               # 树的最大深度
#     learning_rate=0.048144174253709246,         # 学习率
#     n_estimators=50,          # 树的数量
#     subsample=0.8355121389859869,             # 训练每棵树时所用样本的比例
#     colsample_bytree=0.8224292759924623,      # 训练每棵树时所用特征的比例
#     random_state=42            # 随机种子
# )
# 
# =============================================================================


xgb_best.fit(X1_train_select,Y1_train1)
evaluate(xgb_best,X_verify_select,Y_verify,X1_test1,Y1_test)
plot_shap(xgb_best,X1_train_select,Y1_train1,X1_test1)

# %% RandomForestClassifier贝叶斯优化调参

# 定义搜索空间
search_spaces = {
    'n_estimators': (10, 500),  # 树的数量
    'max_depth': (1, 30),       # 树的最大深度
    'min_samples_split': (2, 20),  # 内部节点再划分所需最小样本数
    'min_samples_leaf': (1, 20),   # 叶子节点所需最小样本数
    'bootstrap': [True, False]    # 是否有放回地采样
}

# 创建随机森林分类器
rf = RandomForestClassifier(random_state=42)

# 使用BayesSearchCV进行贝叶斯优化
bayes_search = BayesSearchCV(
    rf,
    search_spaces,
    n_iter=32,  # 搜索次数
    cv=5,       # 交叉验证折数
    scoring='accuracy',  # 评估指标
    random_state=42,
    verbose=1
)

bayes_search.fit(X1_train_select,Y1_train1)

# 输出最佳参数和对应的准确率
print("Best parameters found: ", bayes_search.best_params_)
print("Highest accuracy found: %.2f%%" % (bayes_search.best_score_ * 100))

# 住院时长=============================================================================
# Best parameters found:  OrderedDict([('bootstrap', False), ('max_depth', 25), 
#                                      ('min_samples_leaf', 1), ('min_samples_split', 3), 
#                                      ('n_estimators', 432)])
# Highest accuracy found: 87.53%
# =============================================================================

# 死亡=============================================================================
# Best parameters found:  OrderedDict([('bootstrap', False), ('max_depth', 19), 
#                                      ('min_samples_leaf', 1), ('min_samples_split', 5), 
#                                      ('n_estimators', 500)])
# Highest accuracy found: 95.72%
# =============================================================================

# 肾损伤=============================================================================
# Best parameters found:  OrderedDict([('bootstrap', False), ('max_depth', 27), 
#                                      ('min_samples_leaf', 1), ('min_samples_split', 2), 
#                                      ('n_estimators', 500)])
# Highest accuracy found: 91.44%
# =============================================================================

#心脏 =============================================================================
# Best parameters found:  OrderedDict([('bootstrap', False), ('max_depth', 18), 
#                                      ('min_samples_leaf', 1), ('min_samples_split', 5), 
#                                      ('n_estimators', 41)])
# Highest accuracy found: 99.78%
# =============================================================================

# 肺=============================================================================
# Best parameters found:  OrderedDict([('bootstrap', False), 
#                                      ('max_depth', 20), ('min_samples_leaf', 1), 
#                                      ('min_samples_split', 2), ('n_estimators', 500)])
# Highest accuracy found: 97.92%
# 
# =============================================================================








# %% RandomForestClassifier 最佳参数结果及画图

# =============================================================================
# 住院时长=============================================================================
# rf_best = RandomForestClassifier(
#     n_estimators=432,            # 树的数量
#     max_depth=25,             # 树的最大深度
#     min_samples_split=3,        # 内部节点再划分所需最小样本数
#     min_samples_leaf=1,         # 叶子节点所需最小样本数
#     bootstrap=False,             # 是否使用有放回抽样
#     n_jobs=-1,                  # 使用所有可用的核心进行并行处理
#     random_state=42,            # 随机种子
# )
# =============================================================================
# =============================================================================


# 死亡=============================================================================
# rf_best = RandomForestClassifier(
#     n_estimators=500,            # 树的数量
#     max_depth=19,             # 树的最大深度
#     min_samples_split=5,        # 内部节点再划分所需最小样本数
#     min_samples_leaf=1,         # 叶子节点所需最小样本数
#     bootstrap=False,             # 是否使用有放回抽样
#     n_jobs=-1,                  # 使用所有可用的核心进行并行处理
#     random_state=42,            # 随机种子
# )
# =============================================================================



# 肾损伤=============================================================================
# rf_best = RandomForestClassifier(
#     n_estimators=500,            # 树的数量
#     max_depth=27,             # 树的最大深度
#     min_samples_split=2,        # 内部节点再划分所需最小样本数
#     min_samples_leaf=1,         # 叶子节点所需最小样本数
#     bootstrap=False,             # 是否使用有放回抽样
#     n_jobs=-1,                  # 使用所有可用的核心进行并行处理
#     random_state=42,            # 随机种子
# )
# =============================================================================


# 心脏=============================================================================
# rf_best = RandomForestClassifier(
#     n_estimators=41,            # 树的数量
#     max_depth=18,             # 树的最大深度
#     min_samples_split=5,        # 内部节点再划分所需最小样本数
#     min_samples_leaf=1,         # 叶子节点所需最小样本数
#     bootstrap=False,             # 是否使用有放回抽样
#     n_jobs=-1,                  # 使用所有可用的核心进行并行处理
#     random_state=42,            # 随机种子
# )
# =============================================================================

# 肺=============================================================================
# rf_best = RandomForestClassifier(
#     n_estimators=500,            # 树的数量
#     max_depth=20,             # 树的最大深度
#     min_samples_split=2,        # 内部节点再划分所需最小样本数
#     min_samples_leaf=1,         # 叶子节点所需最小样本数
#     bootstrap=False,             # 是否使用有放回抽样
#     n_jobs=-1,                  # 使用所有可用的核心进行并行处理
#     random_state=42,            # 随机种子
# )
# =============================================================================

# 偷袭=============================================================================
# Best parameters found:  OrderedDict([('bootstrap', False), 
#                                      ('max_depth', 25), ('min_samples_leaf', 1), 
#                                      ('min_samples_split', 2), ('n_estimators', 500)])
# Highest accuracy found: 100.00%
# =============================================================================

# 透析=============================================================================
# rf_best = RandomForestClassifier(
#     n_estimators=500,            # 树的数量
#     max_depth=25,             # 树的最大深度
#     min_samples_split=2,        # 内部节点再划分所需最小样本数
#     min_samples_leaf=1,         # 叶子节点所需最小样本数
#     bootstrap=False,             # 是否使用有放回抽样
#     n_jobs=-1,                  # 使用所有可用的核心进行并行处理
#     random_state=42,            # 随机种子
# )
# =============================================================================


rf_best.fit(X1_train_select,Y1_train1)
evaluate(rf_best,X_verify_select,Y_verify,X1_test1,Y1_test)
plot_shap(rf_best,X1_train_select,Y1_train1,X1_test1)
# %%% DecisionTreeClassifier贝叶斯优化调参

# 定义决策树分类器
dt = tree.DecisionTreeClassifier(random_state=42)


# 定义搜索空间
search_spaces = {
    'criterion': ['gini', 'entropy'],  # 分割质量的标准
    'splitter': ['best', 'random'],    # 分割策略
    'max_depth': (1, 20),              # 树的最大深度
    'min_samples_split': (2, 20),      # 内部节点再划分所需最小样本数
    'min_samples_leaf': (1, 20),       # 叶子节点所需最小样本数
    'max_features': ['sqrt', 'log2', None],  # 寻找最佳分割时考虑的特征数量 'auto', 
}



# 使用BayesSearchCV进行贝叶斯优化
bayes_search = BayesSearchCV(
    dt,
    search_spaces,
    n_iter=32,  # 搜索次数
    cv=3,       # 交叉验证折数
    scoring='accuracy',  # 评估指标
    random_state=42,
    verbose=1
)

bayes_search.fit(X1_train_select,Y1_train1)
# 输出最佳参数和对应的准确率
print("Best parameters found: ", bayes_search.best_params_)
print("Highest accuracy found: %.2f%%" % (bayes_search.best_score_ * 100))

# 住院时长=============================================================================
# Best parameters found:  OrderedDict([('criterion', 'entropy'), ('max_depth', 5), 
#                                      ('max_features', 'sqrt'), ('min_samples_leaf', 20), 
#                                      ('min_samples_split', 7), ('splitter', 'random')])
# Highest accuracy found: 85.81%
# =============================================================================

# 死亡=============================================================================
# Best parameters found:  OrderedDict([('criterion', 'gini'), ('max_depth', 17), 
#                                      ('max_features', 'auto'), ('min_samples_leaf', 1), 
#                                      ('min_samples_split', 5), ('splitter', 'random')])
# Highest accuracy found: 90.46%
# =============================================================================

# 肾损伤=============================================================================
# Best parameters found:  OrderedDict([('criterion', 'entropy'), ('max_depth', 15), 
#                                      ('max_features', 'log2'), ('min_samples_leaf', 3), 
#                                      ('min_samples_split', 20), ('splitter', 'best')])
# Highest accuracy found: 86.73%
# =============================================================================

# 心脏损伤=============================================================================
# Best parameters found:  OrderedDict([('criterion', 'entropy'), 
#                                      ('max_depth', 16), ('max_features', None), 
#                                      ('min_samples_leaf', 1), ('min_samples_split', 2), 
#                                      ('splitter', 'best')])
# Highest accuracy found: 99.97%
# =============================================================================

# 肺=============================================================================
# Best parameters found:  OrderedDict([('criterion', 'gini'), 
#                                      ('max_depth', 9), ('max_features', None),
#                                      ('min_samples_leaf', 2), ('min_samples_split', 20), 
#                                      ('splitter', 'best')])
# Highest accuracy found: 97.06%
# =============================================================================


# 透析=============================================================================
# Best parameters found:  OrderedDict([('criterion', 'gini'), 
#                                      ('max_depth', 20), ('max_features', 'sqrt'), 
#                                      ('min_samples_leaf', 1), ('min_samples_split', 2), 
#                                      ('splitter', 'best')])
# Highest accuracy found: 99.77%
# =============================================================================



# %% DecisionTree 最佳参数结果及画图
# =============================================================================
# 住院时长=============================================================================
# dt_best = tree.DecisionTreeClassifier(
#     criterion='entropy',               # 分割标准
#     max_depth=5,                 # 树的最大深度
#     min_samples_split=7,            # 内部节点再划分所需最小样本数
#     min_samples_leaf=20,             # 叶子节点所需最小样本数
#     max_features='sqrt',              # 寻找最佳分割时考虑的特征数量
#     splitter= 'random' ,             
#     random_state=42)
# =============================================================================
# =============================================================================


# 死亡=============================================================================
# dt_best = tree.DecisionTreeClassifier(
#     criterion='gini',               # 分割标准
#     max_depth=17,                 # 树的最大深度
#     min_samples_split=5,            # 内部节点再划分所需最小样本数
#     min_samples_leaf=1,             # 叶子节点所需最小样本数
#     #max_features='auto',              # 寻找最佳分割时考虑的特征数量
#     splitter= 'random' ,             
# random_state=42)
# =============================================================================



# 肾损伤=============================================================================
# dt_best = tree.DecisionTreeClassifier(
#     criterion='entropy',               # 分割标准
#     max_depth=15,                 # 树的最大深度
#     min_samples_split=20,            # 内部节点再划分所需最小样本数
#     min_samples_leaf=3,             # 叶子节点所需最小样本数
#     max_features='log2',              # 寻找最佳分割时考虑的特征数量
#     splitter= 'best' ,             
# random_state=42)
# =============================================================================


# 心脏=============================================================================
# dt_best = tree.DecisionTreeClassifier(
#     criterion='entropy',               # 分割标准
#     max_depth=16,                 # 树的最大深度
#     min_samples_split=2,            # 内部节点再划分所需最小样本数
#     min_samples_leaf=1,             # 叶子节点所需最小样本数
#     splitter= 'best' ,             
# random_state=42)
# =============================================================================


# 肺=============================================================================
# dt_best = tree.DecisionTreeClassifier(
#     criterion='gini',               # 分割标准
#     max_depth=9,                 # 树的最大深度
#     min_samples_split=20,            # 内部节点再划分所需最小样本数
#     min_samples_leaf=2,             # 叶子节点所需最小样本数
#     splitter= 'best' ,             
# random_state=42)
# =============================================================================

dt_best = tree.DecisionTreeClassifier(
    criterion='gini',               # 分割标准
    max_depth=20,                 # 树的最大深度
    min_samples_split=2,            # 内部节点再划分所需最小样本数
    min_samples_leaf=1,             # 叶子节点所需最小样本数
    splitter= 'best' ,             
random_state=42)

dt_best.fit(X1_train_select,Y1_train1)
evaluate(dt_best,X_verify_select,Y_verify,X1_test1,Y1_test)

# %%% AdaBoostClassifier贝叶斯优化调参
# 定义AdaBoost分类器
ada = AdaBoostClassifier(random_state=42)

# 定义搜索空间
search_spaces = {
    'n_estimators': (50, 500),         # 弱学习器的数量
    'learning_rate': (0.01, 1.0, 'log-uniform'),  # 学习率，采用对数均匀分布
    'algorithm': ['SAMME', 'SAMME.R'], # 算法类型
}

# 使用BayesSearchCV进行贝叶斯优化
bayes_search = BayesSearchCV(
    ada,
    search_spaces,
    n_iter=32,  # 搜索次数
    cv=3,       # 交叉验证折数
    scoring='accuracy',  # 评估指标
    random_state=42,
    verbose=1,
    n_jobs=-1   # 使用所有可用的核心进行并行处理
)

bayes_search.fit(X1_train_select,Y1_train1)
# 输出最佳参数和对应的准确率
print("Best parameters found: ", bayes_search.best_params_)
print("Highest accuracy found: %.2f%%" % (bayes_search.best_score_ * 100))

# 住院时长=============================================================================
# Best parameters found:  OrderedDict([('algorithm', 'SAMME.R'), 
#                                      ('learning_rate', 0.13658427851533933), 
#                                      ('n_estimators', 500)])
# Highest accuracy found: 87.33%
# =============================================================================

# 死亡=============================================================================
# Best parameters found:  OrderedDict([('algorithm', 'SAMME'), 
#                                      ('learning_rate', 0.9873624775671371), 
#                                      ('n_estimators', 288)])
# Highest accuracy found: 96.44%
# =============================================================================

# 肾损伤=============================================================================
# Best parameters found:  OrderedDict([('algorithm', 'SAMME'), 
#                                      ('learning_rate', 0.5942937355266751), 
#                                      ('n_estimators', 57)])
# Highest accuracy found: 92.63%
# =============================================================================

# 心脏损伤=============================================================================
# Best parameters found:  OrderedDict([('algorithm', 'SAMME.R'), 
#                                      ('learning_rate', 0.5842928269761146), 
#                                      ('n_estimators', 187)])
# Highest accuracy found: 99.97%
# =============================================================================

# 肺=============================================================================
# Best parameters found:  OrderedDict([('algorithm', 'SAMME'), 
#                                      ('learning_rate', 1.0), 
#                                      ('n_estimators', 500)])
# Highest accuracy found: 97.50%
# =============================================================================

# =============================================================================
# Best parameters found:  OrderedDict([('algorithm', 'SAMME.R'), 
#                                      ('learning_rate', 0.6586033008288555), 
#                                      ('n_estimators', 497)])
# Highest accuracy found: 99.97%
# =============================================================================





# %% AdaBoostClassifier 最佳参数结果及画图


# 住院=============================================================================
# ada_best = AdaBoostClassifier(
#     algorithm = 'SAMME.R',
#     learning_rate = 0.13658427851533933,
#     n_estimators = 500,
#     random_state=42)
# =============================================================================



# 死 =============================================================================
# ada_best = AdaBoostClassifier(
#     algorithm = 'SAMME',
#     learning_rate = 0.9873624775671371,
#     n_estimators = 288,
# random_state=42)
# =============================================================================


# 肾=============================================================================
# ada_best = AdaBoostClassifier(
#     algorithm = 'SAMME',
#     learning_rate = 0.5942937355266751,
#     n_estimators = 57,
# random_state=42)
# =============================================================================


# 心脏=============================================================================
# ada_best = AdaBoostClassifier(
#     algorithm = 'SAMME.R',
#     learning_rate = 0.5842928269761146,
#     n_estimators = 187,
# random_state=42)
# =============================================================================

# 肺=============================================================================
# ada_best = AdaBoostClassifier(
#     algorithm = 'SAMME',
#     learning_rate = 1.0,
#     n_estimators = 500,
# random_state=42)
# =============================================================================

# 透析=============================================================================
# ada_best = AdaBoostClassifier(
#     algorithm = 'SAMME.R',
#     learning_rate = 0.6586033008288555,
#     n_estimators = 497,
# random_state=42)
# =============================================================================


ada_best.fit(X1_train_select,Y1_train1)
evaluate(ada_best,X_verify_select,Y_verify,X1_test1,Y1_test)

# %%% LogisticRegression贝叶斯优化调参
# 标准化特征（对于LogisticRegression通常是有益的）
#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)

# 定义逻辑回归分类器
log_reg = LogisticRegression(random_state=42, max_iter=10000)

# 定义搜索空间
search_spaces = {
    'C': (1e-6, 1e+6, 'log-uniform'),  # 正则化强度的倒数
    'penalty': ['l1', 'l2'],           # 正则化类型
    'solver': ['liblinear', 'saga'],   # 解决器算法，某些求解器支持特定的正则化类型
}

# 使用BayesSearchCV进行贝叶斯优化
bayes_search = BayesSearchCV(
    log_reg,
    search_spaces,
    n_iter=32,  # 搜索次数
    cv=3,       # 交叉验证折数
    scoring='accuracy',  # 评估指标
    random_state=42,
    verbose=1,
    n_jobs=-1   # 使用所有可用的核心进行并行处理
)

bayes_search.fit(X1_train_select,Y1_train1)
# 输出最佳参数和对应的准确率
print("Best parameters found: ", bayes_search.best_params_)
print("Highest accuracy found: %.2f%%" % (bayes_search.best_score_ * 100))

# 住=============================================================================
# Best parameters found:  OrderedDict([('C', 3.317697704417197), ('penalty', 'l2'), ('solver', 'liblinear')])
# Highest accuracy found: 86.82%
# =============================================================================

#死 =============================================================================
# Best parameters found:  OrderedDict([('C', 1645.7126079253858), ('penalty', 'l1'), ('solver', 'liblinear')])
# Highest accuracy found: 87.18%
# =============================================================================

# 肾=============================================================================
# Best parameters found:  OrderedDict([('C', 3.317697704417197), ('penalty', 'l2'), ('solver', 'liblinear')])
# Highest accuracy found: 85.84%
# =============================================================================

# 心=============================================================================
# Best parameters found:  OrderedDict([('C', 1000000.0), ('penalty', 'l2'), ('solver', 'liblinear')])
# Highest accuracy found: 96.34%
# =============================================================================

# 肺=============================================================================
# Best parameters found:  OrderedDict([('C', 3.317697704417197), 
#                                      ('penalty', 'l2'), 
#                                      ('solver', 'liblinear')])
# Highest accuracy found: 94.00%
# =============================================================================


# 透析=============================================================================
# Best parameters found:  OrderedDict([('C', 1000000.0), ('penalty', 'l1'), ('solver', 'liblinear')])
# Highest accuracy found: 99.53%
# =============================================================================



# %% LogisticRegression 最佳参数结果及画图

# 住院时长=============================================================================
# log_best = LogisticRegression(random_state = 42, 
#                               C = 3.317697704417197,
#                               penalty = 'l2', 
#                               solver = 'liblinear',
#                               max_iter = 10000)
# =============================================================================


# 死亡=============================================================================
# log_best = LogisticRegression(random_state = 42, 
#                               C = 1645.7126079253858,
#                               penalty = 'l1', 
#                               solver = 'liblinear',
#                               max_iter = 10000)
# =============================================================================


# 肾=============================================================================
# log_best = LogisticRegression(random_state = 42, 
#                               C = 3.317697704417197,
#                               penalty = 'l2', 
#                               solver = 'liblinear',
#                               max_iter = 10000)
# =============================================================================



# 心脏=============================================================================
# log_best = LogisticRegression(random_state = 42, 
#                               C = 1000000.0,
#                               penalty = 'l2', 
#                               solver = 'liblinear',
#                               max_iter = 10000)
# =============================================================================

# 肺=============================================================================
# log_best = LogisticRegression(random_state = 42, 
#                               C = 3.317697704417197,
#                               penalty = 'l2', 
#                               solver = 'liblinear',
#                               max_iter = 10000)
# =============================================================================

log_best = LogisticRegression(random_state = 42, 
                              C = 1000000.0,
                              penalty = 'l1', 
                              solver = 'liblinear',
                              max_iter = 10000)


log_best.fit(X1_train_select,Y1_train1)
evaluate(log_best,X_verify_select,Y_verify,X1_test1,Y1_test)


# %%% 堆叠 (Stacking)
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# 定义基础模型
base_models = [
    ('ada', ada_best),
    ('xgb', xgb_best)
]

base_models = [
    ('rf', rf_best),
    ('xgb', xgb_best)
]
# 定义元模型
meta_model = LogisticRegression()

# 创建堆叠分类器
stacked_clf = StackingClassifier(
    estimators = base_models,
    final_estimator = meta_model,
    cv=5
)

# 训练堆叠分类器
stacked_clf.fit(X1_train_select,Y1_train1)

evaluate(stacked_clf,X_verify_select,Y_verify,X1_test1,Y1_test)
   
    #交互图
    #plt.clf()
    #shap_interaction_values = explainer.shap_interaction_values(X1_test1)
    #shap.summary_plot(shap_interaction_values, X1_test1, max_display=7,show=False)
    #plt.savefig('G:/1 项目/5-数据服务/23 肾脏病医学部 张利/肌酸激酶/数据导出/4.png',dpi=300)
    
    #局部条形图
    #plt.clf()
    #shap.plots.bar(explainer2[1], show_data=True) #,show=False
    #plt.savefig('/Users/dongdongdong/Desktop/5.png',dpi=600)
    
    #部分依赖图
    #plt.clf()
    #shap.dependence_plot('血氧_first', shap_values,X1_test1,show=False)
    #plt.savefig('G:/1 项目/5-数据服务/23 肾脏病医学部 张利/肌酸激酶/数据导出/5.png',dpi=300)

# =============================================================================
# 验证集
# ROC AUC Score: 0.9003
# Precision: 0.8682
# Accuracy: 0.8820
# F1 Score: 0.9331
# 测试集
# ROC AUC Score: 0.9038
# Precision: 0.8684
# Accuracy: 0.8825
# F1 Score: 0.9336
# =============================================================================

# =============================================================================
# 验证集
# ROC AUC Score: 0.9892
# Precision: 0.9532
# Accuracy: 0.9532
# F1 Score: 0.9547
# 测试集
# ROC AUC Score: 0.9958
# Precision: 0.9618
# Accuracy: 0.9617
# F1 Score: 0.9585
# =============================================================================

# %%% 数据导出留存
import xlsxwriter
excel_file_path = 'C:\\Users\\Dell\\Desktop\\output-肺损伤.xlsx'
with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
    # 将每个 DataFrame 写入不同的工作表
    X1_train1.to_excel(writer, sheet_name='X1_train1', index=False)
    Y1_train1.to_excel(writer, sheet_name='Y1_train1', index=False)
    
    X_verify.to_excel(writer, sheet_name='X_verify', index=False)
    Y_verify.to_excel(writer, sheet_name='Y_verify', index=False)
    
    X1_test.to_excel(writer, sheet_name='X1_test', index=False)
    Y1_test.to_excel(writer, sheet_name='Y1_test', index=False)
    
    X1_train_select.to_excel(writer, sheet_name='X1_train_select', index=False)
    X_verify_select.to_excel(writer, sheet_name='X_verify_select', index=False)
    X1_test1.to_excel(writer, sheet_name='X1_test1', index=False)



# %%% 画图
# 创建多个模型
models = {
    'XGBClassifier' : xgb_best,
    'RandomForestClassifier' : rf_best,
    'DecisionTreeClassifier' : dt_best,
    'AdaBoostClassifier' : ada_best,
    'LogisticRegression' : log_best
}

# 存储各个模型的 FPR, TPR 和 AUC 值
fpr_dict = {}
tpr_dict = {}
auc_dict = {}

# 设置绘图风格
sns.set(style="whitegrid")

# 创建图形对象
plt.figure(figsize=(10, 8))

# 定义颜色和线条样式
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
linestyles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1))]

# 训练模型并计算 FPR, TPR 和 AUC
for i, (name, model) in enumerate(models.items()):
    model.fit(X1_train_select, Y1_train1)
    y_pred_proba = model.predict_proba(X1_test1)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X1_test1)
    fpr, tpr, _ = roc_curve(Y1_test, y_pred_proba)
    fpr_dict[name] = fpr
    tpr_dict[name] = tpr
    auc_dict[name] = auc(fpr, tpr)

# 绘制每个模型的 ROC 曲线
for i, (name, _) in enumerate(models.items()):
    plt.plot(fpr_dict[name], tpr_dict[name], color=colors[i], lw=2, linestyle=linestyles[i],
             label=f'{name} (AUC = {auc_dict[name]:.4f})')

# 绘制对角线（随机猜测）
plt.plot([0, 1], [0, 1], 'k--', lw=2)

# 添加标题和标签
plt.title('ROC Curves of Multiple Classifiers', fontsize=16)
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)

# 添加图例
plt.legend(loc='lower right', fontsize=12)

# 美化图表
plt.grid(True)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

# 调整字体大小和样式
plt.rcParams.update({'font.size': 14})

# 如果有特别重要的点，可以添加注释
# plt.annotate('Important Point', xy=(0.3, 0.7), xytext=(0.4, 0.8),
#              arrowprops=dict(facecolor='black', shrink=0.05))

# 显示图表
plt.show()

# 保存高质量图片
plt.savefig('C:\\Users\\Dell\\Desktop\\roc_curves.png', dpi=300, bbox_inches='tight')

# %%画图函数

def multi_models_roc(names, sampling_methods, colors, X_test, y_test, save=True, dpin=300):
    plt.figure(figsize=(20, 20), dpi=dpin)  
# =============================================================================
#         将多个机器模型的roc图输出到一张图上
#         
#         Args:
#             names: list, 多个模型的名称
#             sampling_methods: list, 多个模型的实例化对象
#             save: 选择是否将结果保存（默认为png格式）
#             
#         Returns:
#             返回图片对象plt
# =============================================================================
    
    for (name, method, colorname) in zip(names, sampling_methods, colors):
            
        pred = method.predict(X_test)
        prob = method.predict_proba(X_test)[:,1]
        fpr, tpr, thresholds = roc_curve(y_test, prob, pos_label=1)
        
        plt.plot(fpr, tpr, lw=5, label='{} (AUC={:.3f})'.format(name, auc(fpr, tpr)),color = colorname)
        plt.plot([0, 1], [0, 1], '--', lw=5, color = 'grey')
        plt.axis('square')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate',fontsize=20)
        plt.ylabel('True Positive Rate',fontsize=20)
        plt.title('ROC Curve',fontsize=25)
        plt.legend(loc='lower right',fontsize=20)

# =============================================================================
#     if save:
#         plt.savefig('G:\\0 工作备份\\000work\\1 项目\\5-数据服务\\23 肾脏病医学部 张利\\肌酸激酶\\数据导出\\4血液透析-roc.png')
#             
# =============================================================================
    return plt

colors =['crimson','orange','gold','mediumseagreen','steelblue', 'mediumpurple'  ]
names  =['XGBoost','RandomForest','DecisionTree','AdaBoost','LogisticRegression']
sampling_methods = [xgb_best,rf_best,dt_best,ada_best,log_best]



train_roc_graph = multi_models_roc(names, sampling_methods, colors, X1_test1, Y1_test, save = True)
#train_roc_graph.savefig('G:\\0 工作备份\\000work\\1 项目\\5-数据服务\\23 肾脏病医学部 张利\\肌酸激酶\\数据导出\\1住院时长-roc.png')

f = plt.gcf()
f.savefig('C:\\Users\\Dell\\Desktop\\肾roc.png')
plt.show()

