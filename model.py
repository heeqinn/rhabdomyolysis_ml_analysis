# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 19:57:46 2023

@author: Dell
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 15:47:52 2020

@author: Administrator
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

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set(font='SimHei',font_scale=0.8)

#data = pd.read_excel("G:\\0 工作备份\\000work\\1 项目\\5-数据服务\\23 肾脏病医学部\\肌酸激酶\\3-肌酸激酶-数据清洗1.xlsx")
data = pd.read_excel("H:\\1 项目\\5-数据服务\\23 肾脏病医学部\\肌酸激酶\\3-1-肌酸激酶数据清洗补充.xlsx")
# %% 缺失值插补
#随机森林回归预测缺失值
miss_col = data.isnull().any()
miss_col = miss_col.reset_index()
miss_col.columns = ['col','miss']

qs = miss_col[miss_col.miss == True].col
data_miss = data[qs]


notqs = [item for item in data.columns if item not in set(qs)]

sortindex = np.argsort(data_miss.isnull().sum(axis=0)).values
Y1 = data['死亡']
 
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


new_data = pd.concat([data[notqs],data_miss],axis=1) #插补后数据 拼接 无缺失数据

new_data.to_excel("H:\\1 项目\\5-数据服务\\23 肾脏病医学部\\肌酸激酶\\4-new_data2.xlsx",index=False)

# %% 数据分割

new_data = pd.read_excel("G:\\0 工作备份\\000work\\1 项目\\5-数据服务\\23 肾脏病医学部\\肌酸激酶\\4-new_data.xlsx")

X = new_data.drop(['住院时长>7（天）','住院时长（天）','序号','PATIENT_ID','VISIT_ID','NAME','身高','体重'],axis=1) 
#data.iloc[:,:-1]
Y = new_data['住院时长>7（天）']

#aaa = X.corr()

#test是测试集
X1_train,X1_test,Y1_train,Y1_test = train_test_split(X,Y,test_size = 0.2,random_state=1)
#train1是训练，verify是验证
X1_train1,X_verify,Y1_train1,Y_verify = train_test_split(X1_train,Y1_train,test_size = 0.25,random_state=1)
#6:2:2

# %% RFECV
estimator = LinearSVC()
#rf = RF(random_state=30)
selector = RFECV(estimator = estimator ,step = 1 , cv = 3 ,scoring = 'accuracy') #StratifiedKFold(3)
selector.fit(X1_train1,Y1_train1)

print("N_features %s" % selector.n_features_)
print("Support is %s" % selector.support_)
print("Ranking %s" % selector.ranking_)
print("Grid Scores %s" % selector.grid_scores_)

# =============================================================================
# plt.figure()
# plt.xlabel('number of features selected')
# plt.ylabel('cross validation score')
# 
# a = range(1,len(selector.grid_scores_) +1)
# b = selector.grid_scores_
# 
# #plt.figure(figsize=(15,10))
# plt.figure(dpi=1000)
# plt.plot(a, b)
# for x,y in zip(a,b):
#     plt.text(x-0.3,y+0.001,str(x),ha='center',va = 'bottom' , fontsize=5)
# plt.show()
# =============================================================================


#N_features 59
# %% rfe筛选变量

rf = RF(random_state=30)
rfe = RFE(estimator = rf , n_features_to_select = 59 ) #筛选45变量
rfe.fit(X1_train1,Y1_train1)
rfe_ranking = rfe.ranking_
rfe_importances = pd.DataFrame({'rank':rfe_ranking,'var':X1_train1.columns})
rfe_importances = rfe_importances.sort_values(by='rank',ascending=True)
rfe_top_59 = rfe_importances.loc[rfe_importances['rank']==1]['var'].tolist()


X1_train_select = X1_train1[rfe_top_59]
X_verify_select = X_verify[rfe_top_59]

# %% 随机森林
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=30)
rf.fit(X1_train_select,Y1_train1)

y_train_pred = rf.predict(X1_train_select) 

y_train_prob = rf.predict_proba(X1_train_select)[:,1] 
#thres = 0.4
#y_train_pred = (y_train_prob > thres).astype(int) 
#thres = 0.6
#y_train_pred1 = (y_train_prob > thres) + 0
print('rf Train ROC IS: %s' % roc_auc_score(Y1_train1,y_train_prob)) 
print('rf verify Precision IS: %s' % precision_score(Y1_train1,y_train_pred))
print('rf verify Accuracy IS: %s' % accuracy_score(Y1_train1,y_train_pred)) #准确率
print('rf verify F1 IS: %s' % f1_score(Y1_train1,y_train_pred))

confusion = confusion_matrix(Y1_train1,y_train_pred)
tn,fp,fn,tp = confusion.ravel()
sen = tp/(tp+fn)
spe = tn/(tn+fp)
print('rf sen IS: %s' % sen)
print('rf spe IS: %s' % spe)

#验证
y_verify_pred = rf.predict(X_verify_select)
y_verify_prob = rf.predict_proba(X_verify_select)[:,1] 
#y_verify_pred1 = (y_verify_prob > thres) + 0

print('rf verify ROC IS: %s' % roc_auc_score(Y_verify,y_verify_prob))  #auc
print('rf verify Precision IS: %s' % precision_score(Y_verify,y_verify_pred))
print('rf verify Accuracy IS: %s' % accuracy_score(Y_verify,y_verify_pred)) #准确率
print('rf verify F1 IS: %s' % f1_score(Y_verify,y_verify_pred))


confusion = confusion_matrix(Y_verify,y_verify_pred)
tn,fp,fn,tp = confusion.ravel()
sen = tp/(tp+fn)
spe = tn/(tn+fp)
print('rf sen IS: %s' % sen)
print('rf spe IS: %s' % spe)

#测试
X1_test1 = X1_test[rfe_top_59]
y_test_pred = rf.predict(X1_test1)
y_test_prob = rf.predict_proba(X1_test1)[:,1] 
#y_verify_pred1 = (y_verify_prob > thres) + 0

print('rf test ROC IS: %s' % roc_auc_score(Y1_test,y_test_prob))  #auc
print('rf test Precision IS: %s' % precision_score(Y1_test,y_test_pred))
print('rf test Accuracy IS: %s' % accuracy_score(Y1_test,y_test_pred)) #准确率
print('rf test F1 IS: %s' % f1_score(Y1_test,y_test_pred))

confusion = confusion_matrix(Y1_test,y_test_pred)
tn,fp,fn,tp = confusion.ravel()
sen = tp/(tp+fn)
spe = tn/(tn+fp)
print('rf sen IS: %s' % sen)
print('rf spe IS: %s' % spe)
# %% 多模型比较
import pandas as pd
from sklearn.model_selection import train_test_split 
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score,roc_auc_score,accuracy_score,f1_score,roc_curve
xgb = XGBClassifier(seed=30)
rf = RandomForestClassifier(random_state=30)
dt = tree.DecisionTreeClassifier(max_depth = 7 , random_state=30)
ada = AdaBoostClassifier(random_state=30)
reg = LogisticRegression(solver='newton-cg',n_jobs=-1,random_state=30)


alg_list = [xgb,rf,dt,ada,reg]

def algorithm(X1_train1,Y1_train,X_verify, Y_verify,X1_test1,Y1_test,
              somelist,train_list,test_list,verify_list,im_list,prob_list1,prob_list2):
    for i in somelist:
        i.fit(X1_train1,Y1_train)
   
        #训练
        train_pred = i.predict(X1_train1)
        train_prob = i.predict_proba(X1_train1)[:,1] 
        
        confusion = confusion_matrix(Y1_train,train_pred)
        tn,fp,fn,tp = confusion.ravel()
             
        Train_AUC = roc_auc_score(Y1_train,train_prob)
        Train_Precision = precision_score(Y1_train,train_pred)
        Train_Accuracy = accuracy_score(Y1_train,train_pred)
        Train_F1 = f1_score(Y1_train,train_pred)
        Train_sen = tp/(tp+fn)
        Train_spe = tn/(tn+fp)        
        
        #验证
        y_verify_pred = i.predict(X_verify) 
        y_verify_prob = i.predict_proba(X_verify)[:,1]  
        confusion1 = confusion_matrix(Y_verify,y_verify_pred)
        tn1,fp1,fn1,tp1 = confusion1.ravel()
        
        Verify_AUC = roc_auc_score(Y_verify,y_verify_prob)
        Verify_Precision = precision_score(Y_verify,y_verify_pred)
        Verify_Accuracy = accuracy_score(Y_verify,y_verify_pred)
        Verify_F1 = f1_score(Y_verify,y_verify_pred)
        sen_Verify = tp1/(tp1+fn1)
        spe_Verify = tn1/(tn1+fp1)
        
        #测试
        y_pred = i.predict(X1_test1)
        y_prob = i.predict_proba(X1_test1)[:,1] 
        
        confusion = confusion_matrix(Y1_test,y_pred)
        tn,fp,fn,tp = confusion.ravel()
             
        Test_AUC = roc_auc_score(Y1_test,y_prob)
        Test_Precision = precision_score(Y1_test,y_pred)
        Test_Accuracy = accuracy_score(Y1_test,y_pred)
        Test_F1 = f1_score(Y1_test,y_pred)
        sen = tp/(tp+fn)
        spe = tn/(tn+fp)
        

        
        
        
        train_list.append([i,Train_AUC,Train_Precision,Train_Accuracy,Train_F1,Train_sen,Train_spe])
        verify_list.append([i,Verify_AUC,Verify_Precision,Verify_Accuracy,Verify_F1,sen_Verify,spe_Verify])
        test_list.append([i,Test_AUC,Test_Precision,Test_Accuracy,Test_F1,sen,spe])
        
        prob_list1.append(y_prob)
        prob_list2.append(y_verify_prob)
        
        
        if i in [xgb,rf,dt,ada]:
            im = pd.DataFrame({'importance':i.feature_importances_,'var':X1_train1.columns})
            im = im.sort_values(by='importance',ascending=False)
        else:
            im = pd.DataFrame({'importance':abs(i.coef_[0]),'var':X1_train1.columns})
            im = im.sort_values(by='importance',ascending=False)
        im_list.append(np.array(im).tolist())
    return test_list,verify_list,im_list,prob_list1,prob_list2,train_list


##########################################################################








def DF2Excel(data_path,data_list,sheet_name_list):
    
    '''将多个dataframe 保存到同一个excel 的不同sheet 上
    参数：
    data_path：str
        需要保存的文件地址及文件名 
    data_list：list
        需要保存到excel的dataframe 
    sheet_name_list：list
        sheet name 每个sheet 的名称
    '''

    write = pd.ExcelWriter(data_path ) 
    for da,sh_name  in zip(data_list,sheet_name_list):
        da.to_excel(write,sheet_name = sh_name,index=False)
    
    #必须运行write.save()，不然不能输出到本地
    write._save()




def main():
    train_list = []
    result_list = []
    verify_list = []
    prob_list1 = []
    prob_list2 = []
    im_list = []
    re = algorithm(X1_train_select , Y1_train1,X_verify_select, Y_verify, X1_test1, Y1_test, 
                   alg_list,train_list,result_list,verify_list,im_list,prob_list1,prob_list2)
    
    # 需要保存的文件地址及文件名
    data_path = 'C:\\Users\\Dell\\Desktop\\result_BeInHospital.xlsx'
    # 要保存的每个sheet 名称
    sheet_name_list = ['test_result', 'verify_result','im','train_result']
    # dataframe list 
    df = pd.DataFrame(re[0],columns=['算法','AUC','Precision','Accuracy','F1','sen','spe'])
    df1 = pd.DataFrame(re[1],columns=['算法','AUC','Precision','Accuracy','F1','sen','spe'])
    df2 = pd.DataFrame(re[2])
    df5 = pd.DataFrame(re[5],columns=['算法','AUC','Precision','Accuracy','F1','sen','spe'])
    
    data_list = [df,df1,df2,df5]
    # 调用函数 
    DF2Excel(data_path,data_list,sheet_name_list)
    

# =============================================================================
#     df = pd.DataFrame(re[0],columns=['算法','AUC','Precision','Accuracy','F1','sen','spe'])
#     df.to_excel('C:\\Users\\Dell\\Desktop\\result_BeInHospital.xlsx',index=False,encoding='utf-8',sheet_name='test_result')
#     
#     df1 = pd.DataFrame(re[1],columns=['算法','AUC','Precision','Accuracy','F1','sen','spe'])
#     df1.to_excel('C:\\Users\\Dell\\Desktop\\result_BeInHospital.xlsx',index=False,encoding='utf-8',sheet_name='verify_result')
# 
#     df2 = pd.DataFrame(re[2])
#     df2.to_excel('C:\\Users\\Dell\\Desktop\\result_BeInHospital.xlsx',index=False,encoding='utf-8',sheet_name='im')
# 
#     df3 = pd.DataFrame(re[3])
#     df4 = pd.DataFrame(re[4])
#     
# 
#     df3.to_excel('C:\\Users\\Dell\\Desktop\\result_BeInHospital.xlsx',index=False,encoding='utf-8',sheet_name='test')
#     df4.to_excel('C:\\Users\\Dell\\Desktop\\result_BeInHospital.xlsx',index=False,encoding='utf-8',sheet_name='veriify')
# 
#     df5 = pd.DataFrame(re[5],columns=['算法','AUC','Precision','Accuracy','F1','sen','spe'])
#     df5.to_excel('C:\\Users\\Dell\\Desktop\\result_BeInHospital.xlsx',index=False,encoding='utf-8',sheet_name='train_result')
# =============================================================================

main()

    
    
# %% 结局为死亡
#586

X = new_data.drop(['住院时长>7（天）','死亡','序号','PATIENT_ID','VISIT_ID','NAME','身高','体重'],axis=1) #有bmi就drop身高体重了
#data.iloc[:,:-1]
Y = new_data['死亡']
    
  
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

# %% RFECV
estimator = LinearSVC()
#rf = RF(random_state=30)
selector = RFECV(estimator = estimator ,step = 1 , cv = 3 ,scoring = 'accuracy') #StratifiedKFold(3)
selector.fit(X1_train1,Y1_train1)

print("N_features %s" % selector.n_features_) #16


# %% rfe筛选变量

rf = RF(random_state=25)
rfe = RFE(estimator = rf , n_features_to_select = 16 ) #筛选变量
rfe.fit(X1_train1,Y1_train1)
rfe_ranking = rfe.ranking_
rfe_importances = pd.DataFrame({'rank':rfe_ranking,'var':X1_train1.columns})
rfe_importances = rfe_importances.sort_values(by='rank',ascending=True)
rfe_top_16 = rfe_importances.loc[rfe_importances['rank']==1]['var'].tolist()


X1_train_select = X1_train1[rfe_top_16]
X_verify_select = X_verify[rfe_top_16]

# %% rfe筛选变量
def main():
    train_list = []
    result_list = []
    verify_list = []
    prob_list1 = []
    prob_list2 = []
    im_list = []
    re = algorithm(X1_train_select , Y1_train1,X_verify_select, Y_verify, X1_test1, Y1_test, 
                   alg_list,train_list,result_list,verify_list,im_list,prob_list1,prob_list2)
    
    # 需要保存的文件地址及文件名
    data_path = 'C:\\Users\\Dell\\Desktop\\result_Death.xlsx'
    # 要保存的每个sheet 名称
    sheet_name_list = ['test_result', 'verify_result','im','train_result']
    # dataframe list 
    df = pd.DataFrame(re[0],columns=['算法','AUC','Precision','Accuracy','F1','sen','spe'])
    df1 = pd.DataFrame(re[1],columns=['算法','AUC','Precision','Accuracy','F1','sen','spe'])
    df2 = pd.DataFrame(re[2])
    df5 = pd.DataFrame(re[5],columns=['算法','AUC','Precision','Accuracy','F1','sen','spe'])
    
    data_list = [df,df1,df2,df5]
    # 调用函数 
    DF2Excel(data_path,data_list,sheet_name_list)
    

main()    
    
# %% 结局为肾损伤
#283

X = new_data.drop(['住院时长>7（天）','肾损伤','序号','PATIENT_ID','VISIT_ID','NAME','身高','体重'],axis=1) #有bmi就drop身高体重了
#data.iloc[:,:-1]
Y = new_data['肾损伤']
    
  
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
 
# %% RFECV
estimator = LinearSVC()
#rf = RF(random_state=30)
selector = RFECV(estimator = estimator ,step = 1 , cv = 3 ,scoring = 'accuracy') #StratifiedKFold(3)
selector.fit(X1_train1,Y1_train1)

print("N_features %s" % selector.n_features_) #17


# %% rfe筛选变量

rf = RF(random_state=25)
rfe = RFE(estimator = rf , n_features_to_select = 17 ) #筛选变量
rfe.fit(X1_train1,Y1_train1)
rfe_ranking = rfe.ranking_
rfe_importances = pd.DataFrame({'rank':rfe_ranking,'var':X1_train1.columns})
rfe_importances = rfe_importances.sort_values(by='rank',ascending=True)
rfe_top_17 = rfe_importances.loc[rfe_importances['rank']==1]['var'].tolist()


X1_train_select = X1_train1[rfe_top_17]
X_verify_select = X_verify[rfe_top_17]
X1_test1 = X1_test[rfe_top_17]
# %% rfe筛选变量
def main():
    train_list = []
    result_list = []
    verify_list = []
    prob_list1 = []
    prob_list2 = []
    im_list = []
    re = algorithm(X1_train_select , Y1_train1,X_verify_select, Y_verify, X1_test1, Y1_test, 
                   alg_list,train_list,result_list,verify_list,im_list,prob_list1,prob_list2)
    
    # 需要保存的文件地址及文件名
    data_path = 'C:\\Users\\Dell\\Desktop\\result_Shensunshang.xlsx'
    # 要保存的每个sheet 名称
    sheet_name_list = ['test_result', 'verify_result','im','train_result']
    # dataframe list 
    df = pd.DataFrame(re[0],columns=['算法','AUC','Precision','Accuracy','F1','sen','spe'])
    df1 = pd.DataFrame(re[1],columns=['算法','AUC','Precision','Accuracy','F1','sen','spe'])
    df2 = pd.DataFrame(re[2])
    df5 = pd.DataFrame(re[5],columns=['算法','AUC','Precision','Accuracy','F1','sen','spe'])
    
    data_list = [df,df1,df2,df5]
    # 调用函数 
    DF2Excel(data_path,data_list,sheet_name_list)
    

main()    




# %% 结局为血液透析
#17

X = new_data.drop(['住院时长>7（天）','血液透析','血液透析count','序号','PATIENT_ID','VISIT_ID','NAME','身高','体重'],axis=1) #有bmi就drop身高体重了
#data.iloc[:,:-1]
Y = new_data['血液透析']
    
  
#组合采样
#SMOTETomek 和SMOTEENN
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import RandomUnderSampler,ClusterCentroids,NearMiss
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


over = SMOTE(sampling_strategy=0.3,random_state=25)
under = RandomUnderSampler(sampling_strategy=0.7,random_state=25)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)
X_resampled, y_resampled = pipeline.fit_resample(X, Y)

print(sorted(Counter(y_resampled).items()))  #[(0, 8692), (1, 6085)]


# =============================================================================
# from imblearn.combine import SMOTEENN
# smote_enn = SMOTEENN(random_state=0)
# X_resampled, y_resampled = smote_enn.fit_resample(X, Y)
# print(sorted(Counter(y_resampled).items())) #[(0, 19989), (1, 20285)]
#  
# from imblearn.combine import SMOTETomek
# smote_tomek = SMOTETomek(random_state=0)
# X_resampled, y_resampled = smote_tomek.fit_resample(X, Y)
# print(sorted(Counter(y_resampled).items())) #[(0, 20285), (1, 20285)]
# =============================================================================
    
#test是测试集
X1_train,X1_test,Y1_train,Y1_test = train_test_split(X_resampled, y_resampled,test_size = 0.2,random_state=1)
#train1是训练，verify是验证
X1_train1,X_verify,Y1_train1,Y_verify = train_test_split(X1_train,Y1_train,test_size = 0.25,random_state=1)
#6:2:2
 
# %% RFECV
estimator = LinearSVC()
#rf = RF(random_state=30)
selector = RFECV(estimator = estimator ,step = 1 , cv = 5 ,scoring = 'accuracy',n_jobs=-1) #StratifiedKFold(3)
#,min_features_to_select=5
selector.fit(X1_train1,Y1_train1)

print("N_features %s" % selector.n_features_) #50

# %% rfe筛选变量

rf = RF(random_state=25)
rfe = RFE(estimator = rf , n_features_to_select = 50 ) #筛选变量
rfe.fit(X1_train1,Y1_train1)
rfe_ranking = rfe.ranking_
rfe_importances = pd.DataFrame({'rank':rfe_ranking,'var':X1_train1.columns})
rfe_importances = rfe_importances.sort_values(by='rank',ascending=True)
rfe_top_50 = rfe_importances.loc[rfe_importances['rank']==1]['var'].tolist()


X1_train_select = X1_train1[rfe_top_50]
X_verify_select = X_verify[rfe_top_50]
X1_test1 = X1_test[rfe_top_50]
# %% main
def main():
    train_list = []
    result_list = []
    verify_list = []
    prob_list1 = []
    prob_list2 = []
    im_list = []
    re = algorithm(X1_train_select , Y1_train1,X_verify_select, Y_verify, X1_test1, Y1_test, 
                   alg_list,train_list,result_list,verify_list,im_list,prob_list1,prob_list2)
    
    # 需要保存的文件地址及文件名
    data_path = 'C:\\Users\\Dell\\Desktop\\result_touxi.xlsx'
    # 要保存的每个sheet 名称
    sheet_name_list = ['test_result', 'verify_result','im','train_result']
    # dataframe list 
    df = pd.DataFrame(re[0],columns=['算法','AUC','Precision','Accuracy','F1','sen','spe'])
    df1 = pd.DataFrame(re[1],columns=['算法','AUC','Precision','Accuracy','F1','sen','spe'])
    df2 = pd.DataFrame(re[2])
    df5 = pd.DataFrame(re[5],columns=['算法','AUC','Precision','Accuracy','F1','sen','spe'])
    
    data_list = [df,df1,df2,df5]
    # 调用函数 
    DF2Excel(data_path,data_list,sheet_name_list)
    

main()    


# %% 结局为肺损伤
#肺3584 心脏16751
new_data = pd.read_excel("G:\\1 项目\\5-数据服务\\23 肾脏病医学部\\肌酸激酶\\4-new_data1.xlsx")

X = new_data.drop(['住院时长>7（天）','肺损伤','序号','PATIENT_ID','VISIT_ID','NAME','身高','体重'],axis=1) #有bmi就drop身高体重了
#data.iloc[:,:-1]
Y = new_data['肺损伤']

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
estimator = LinearSVC()
#rf = RF(random_state=30)
selector = RFECV(estimator = estimator ,step = 1 , cv = 3 ,scoring = 'accuracy') #StratifiedKFold(3)
selector.fit(X1_train1,Y1_train1)

print("N_features %s" % selector.n_features_) #57
# %% rfe筛选变量
rf = RF(random_state=30)
rfe = RFE(estimator = rf , n_features_to_select = 57 ) #筛选57变量
rfe.fit(X1_train1,Y1_train1)
rfe_ranking = rfe.ranking_
rfe_importances = pd.DataFrame({'rank':rfe_ranking,'var':X1_train1.columns})
rfe_importances = rfe_importances.sort_values(by='rank',ascending=True)
rfe_top_57 = rfe_importances.loc[rfe_importances['rank']==1]['var'].tolist()

X1_train_select = X1_train1[rfe_top_57]
X_verify_select = X_verify[rfe_top_57]
X1_test1 = X1_test[rfe_top_57]


# %% main
def main():
    train_list = []
    result_list = []
    verify_list = []
    prob_list1 = []
    prob_list2 = []
    im_list = []
    re = algorithm(X1_train_select , Y1_train1,X_verify_select, Y_verify, X1_test1, Y1_test, 
                   alg_list,train_list,result_list,verify_list,im_list,prob_list1,prob_list2)
    
    # 需要保存的文件地址及文件名
    data_path = 'C:\\Users\\Dell\\Desktop\\result_lungdamage2.xlsx'
    # 要保存的每个sheet 名称
    sheet_name_list = ['test_result', 'verify_result','im','train_result']
    # dataframe list 
    df = pd.DataFrame(re[0],columns=['算法','AUC','Precision','Accuracy','F1','sen','spe'])
    df1 = pd.DataFrame(re[1],columns=['算法','AUC','Precision','Accuracy','F1','sen','spe'])
    df2 = pd.DataFrame(re[2])
    df5 = pd.DataFrame(re[5],columns=['算法','AUC','Precision','Accuracy','F1','sen','spe'])
    
    data_list = [df,df1,df2,df5]
    # 调用函数 
    DF2Excel(data_path,data_list,sheet_name_list)
    

main()    
# %% 结局为心脏损伤
#肺3584 心脏16751
new_data = pd.read_excel("H:\\1 项目\\5-数据服务\\23 肾脏病医学部\\肌酸激酶\\4-new_data1.xlsx")

X = new_data.drop(['住院时长>7（天）','心脏损伤','序号','PATIENT_ID','VISIT_ID','NAME','身高','体重'],axis=1) #有bmi就drop身高体重了
#data.iloc[:,:-1]
Y = new_data['心脏损伤']

    
#test是测试集
X1_train,X1_test,Y1_train,Y1_test = train_test_split(X,Y,test_size = 0.2,random_state=1)
#train1是训练，verify是验证
X1_train1,X_verify,Y1_train1,Y_verify = train_test_split(X1_train,Y1_train,test_size = 0.25,random_state=1)
#6:2:2

# %% RFECV
estimator = LinearSVC()
#rf = RF(random_state=30)
selector = RFECV(estimator = estimator ,step = 1 , cv = 3 ,scoring = 'accuracy') #StratifiedKFold(3)
selector.fit(X1_train1,Y1_train1)

print("N_features %s" % selector.n_features_) #54
# %% rfe筛选变量
rf = RF(random_state=30)
rfe = RFE(estimator = rf , n_features_to_select = 54 ) #筛选54变量
rfe.fit(X1_train1,Y1_train1)
rfe_ranking = rfe.ranking_
rfe_importances = pd.DataFrame({'rank':rfe_ranking,'var':X1_train1.columns})
rfe_importances = rfe_importances.sort_values(by='rank',ascending=True)
rfe_top_54 = rfe_importances.loc[rfe_importances['rank']==1]['var'].tolist()

X1_train_select = X1_train1[rfe_top_54]
X_verify_select = X_verify[rfe_top_54]
X1_test1 = X1_test[rfe_top_54]