# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 15:16:54 2023

@author: Dell
"""
# %% 导入包
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
import pandas as pd
from sklearn.model_selection import train_test_split 
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score,roc_auc_score,accuracy_score,f1_score,roc_curve,auc


sns.set(font='SimHei',font_scale=0.8)

new_data = pd.read_excel("G:\\1 项目\\5-数据服务\\23 肾脏病医学部\\肌酸激酶\\4-new_data.xlsx")

# %% 结局：住院时长
X = new_data.drop(['住院时长>7（天）','住院时长（天）','序号','PATIENT_ID','VISIT_ID','NAME','身高','体重'],axis=1) 
Y = new_data['住院时长>7（天）']

#test是测试集
X1_train,X1_test,Y1_train,Y1_test = train_test_split(X,Y,test_size = 0.2,random_state=1)
#train1是训练，verify是验证
X1_train1,X_verify,Y1_train1,Y_verify = train_test_split(X1_train,Y1_train,test_size = 0.25,random_state=1)
#6:2:2

rfe_top_59 = ['AGE','尿蛋白定量测定_min','血浆活化部分凝血活酶时间测定_min',
              '血浆纤维蛋白原测定_min','血浆凝血酶原时间测定_min','凝血酶时间测定_min',
              '血小板计数_min','红细胞比积测定_min','白细胞计数_min','尿量_min',
              '血红蛋白测定_min','总蛋白_min','丙氨酸氨基转移酶_min','天冬氨酸氨基转移酶_min',
              '钙_min','无机磷_min','血清尿酸_min','钾_min','肌酐_min','血清白蛋白_min',
              '肌酸激酶_max','乳酸脱氢酶_max','肌红蛋白定量_max','尿量_max','尿蛋白定量测定_max',
              '血浆活化部分凝血活酶时间测定_max','血浆纤维蛋白原测定_max','血浆凝血酶原时间测定_max',
              '凝血酶时间测定_max','血小板计数_max','红细胞比积测定_max','白细胞计数_max',
              '血红蛋白测定_max','血清白蛋白_max','总蛋白_max','丙氨酸氨基转移酶_max',
              '天冬氨酸氨基转移酶_max','钙_max','无机磷_max','血清尿酸_max','钾_max',
              '肌酐_max','肌红蛋白定量_min','乳酸脱氢酶_min','血氧_max','血压high_max',
              '肌酸激酶_min','肌酸激酶_last','BMI-1','脉搏_first','体温_first',
              '血压Low_first','肌酐_last','血氧_first','体温_max','血压high_first',
              '血压Low_max','脉搏_max','呼吸_max']

X1_train_select = X1_train1[rfe_top_59]
X_verify_select = X_verify[rfe_top_59]
X1_test1 = X1_test[rfe_top_59]

xgb = XGBClassifier(seed=30).fit(X1_train_select,Y1_train1)
rf = RandomForestClassifier(random_state=30).fit(X1_train_select,Y1_train1)
dt = tree.DecisionTreeClassifier(max_depth = 7 , random_state=30).fit(X1_train_select,Y1_train1)
ada = AdaBoostClassifier(random_state=30).fit(X1_train_select,Y1_train1)
reg = LogisticRegression(solver='newton-cg',n_jobs=-1,random_state=30).fit(X1_train_select,Y1_train1)

# %% 结局：死亡
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

rfe_top_16 = ['血氧_max','呼吸_max','肌酸激酶_last','肌红蛋白定量_min','天冬氨酸氨基转移酶_min','尿蛋白定量测定_min','血氧_first','肌酸激酶_max','乳酸脱氢酶_max','肌红蛋白定量_max','呼吸_first','天冬氨酸氨基转移酶_max','丙氨酸氨基转移酶_max','凝血酶时间测定_max','SEX','肌酐_max']
X1_train_select = X1_train1[rfe_top_16]
X_verify_select = X_verify[rfe_top_16]
X1_test1 = X1_test[rfe_top_16]

xgb = XGBClassifier(seed=30).fit(X1_train_select,Y1_train1)
rf = RandomForestClassifier(random_state=30).fit(X1_train_select,Y1_train1)
dt = tree.DecisionTreeClassifier(max_depth = 7 , random_state=30).fit(X1_train_select,Y1_train1)
ada = AdaBoostClassifier(random_state=30).fit(X1_train_select,Y1_train1)
reg = LogisticRegression(solver='newton-cg',n_jobs=-1,random_state=30).fit(X1_train_select,Y1_train1)
# %% 结局：肾损伤
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

rfe_top_17 = ['血氧_max','血氧_first','肌酸激酶_min','肌酸激酶_last','肌酐_min','丙氨酸氨基转移酶_min','尿量_min','血压Low_first','肌酸激酶_max','肌红蛋白定量_max','肌酐_max','天冬氨酸氨基转移酶_max','丙氨酸氨基转移酶_max','尿量_max','SEX','乳酸脱氢酶_max','碳酸氢钠']
X1_train_select = X1_train1[rfe_top_17]
X_verify_select = X_verify[rfe_top_17]
X1_test1 = X1_test[rfe_top_17]

xgb = XGBClassifier(seed=30).fit(X1_train_select,Y1_train1)
rf = RandomForestClassifier(random_state=30).fit(X1_train_select,Y1_train1)
dt = tree.DecisionTreeClassifier(max_depth = 7 , random_state=30).fit(X1_train_select,Y1_train1)
ada = AdaBoostClassifier(random_state=30).fit(X1_train_select,Y1_train1)
reg = LogisticRegression(solver='newton-cg',n_jobs=-1,random_state=30).fit(X1_train_select,Y1_train1)
# %% 结局为血液透析
#17

X = new_data.drop(['住院时长>7（天）','血液透析','血液透析count','序号','PATIENT_ID','VISIT_ID','NAME','身高','体重'],axis=1) #有bmi就drop身高体重了
#data.iloc[:,:-1]
Y = new_data['血液透析']
    
  
#组合采样
#SMOTETomek 和SMOTEENN
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# =============================================================================
# over_values = [0.3,0.4,0.5]
# under_values = [0.7,0.6,0.5]
# for o in over_values:
#   for u in under_values:
#     # define pipeline
#     model = SVC()
#     over = SMOTE(sampling_strategy=o)
#     under = RandomUnderSampler(sampling_strategy=u)
#     steps = [('over', over), ('under', under), ('model', model)]
#     pipeline = Pipeline(steps=steps)
#     # evaluate pipeline
#     scores = cross_val_score(pipeline, X, Y, scoring='roc_auc', cv=5, n_jobs=-1)
#     score = np.mean(scores)
#     print('SMOTE oversampling rate:%.1f, Random undersampling rate:%.1f , Mean ROC AUC: %.3f' % (o, u, score))
# 
# 
# =============================================================================
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

rfe_top_50 = ['AGE','血浆活化部分凝血活酶时间测定_min','凝血酶时间测定_min','红细胞比积测定_min','血红蛋白测定_min','血清白蛋白_min','总蛋白_min','丙氨酸氨基转移酶_min','钙_min','无机磷_min','血清尿酸_min','肌酐_min','肌红蛋白定量_min','肌酸激酶_last','血压Low_max','体温_max','尿蛋白定量测定_min','脉搏_max','肌酸激酶_max','肌红蛋白定量_max','尿量_max','尿蛋白定量测定_max','血浆活化部分凝血活酶时间测定_max','血浆纤维蛋白原测定_max','血浆凝血酶原时间测定_max','凝血酶时间测定_max','红细胞比积测定_max','白细胞计数_max','丙氨酸氨基转移酶_max','天冬氨酸氨基转移酶_max','钙_max','无机磷_max','血清尿酸_max','钾_max','肌酐_max','乳酸脱氢酶_max','血氧_first','肌酐_last','碳酸氢钠count','住院时长（天）','体温_first','呼吸_first','肾损伤','肌酸激酶_min','脉搏_first','BMI-1','甘露醇count','甘露醇','输血count','碳酸氢钠']
X1_train_select = X1_train1[rfe_top_50]
X_verify_select = X_verify[rfe_top_50]
X1_test1 = X1_test[rfe_top_50]


xgb = XGBClassifier(seed=30).fit(X1_train_select,Y1_train1)
rf = RandomForestClassifier(random_state=30).fit(X1_train_select,Y1_train1)
dt = tree.DecisionTreeClassifier(max_depth = 7 , random_state=30).fit(X1_train_select,Y1_train1)
ada = AdaBoostClassifier(random_state=30).fit(X1_train_select,Y1_train1)
reg = LogisticRegression(solver='newton-cg',n_jobs=-1,random_state=30).fit(X1_train_select,Y1_train1)

# %% 结局为心脏损伤
#3584

#组合采样
#SMOTETomek 和SMOTEENN
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


over = SMOTE(sampling_strategy=0.3,random_state=25)
under = RandomUnderSampler(sampling_strategy=0.7,random_state=25)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)
X_resampled, y_resampled = pipeline.fit_resample(X, Y)

print(sorted(Counter(y_resampled).items()))  #[(0, 8692), (1, 6085)]


xgb = XGBClassifier(seed=30).fit(X1_train_select,Y1_train1)
rf = RandomForestClassifier(random_state=30).fit(X1_train_select,Y1_train1)
dt = tree.DecisionTreeClassifier(max_depth = 7 , random_state=30).fit(X1_train_select,Y1_train1)
ada = AdaBoostClassifier(random_state=30).fit(X1_train_select,Y1_train1)
reg = LogisticRegression(solver='newton-cg',n_jobs=-1,random_state=30).fit(X1_train_select,Y1_train1)

# %% 结局为肺损伤
#3584
xgb = XGBClassifier(seed=30).fit(X1_train_select,Y1_train1)
rf = RandomForestClassifier(random_state=30).fit(X1_train_select,Y1_train1)
dt = tree.DecisionTreeClassifier(max_depth = 7 , random_state=30).fit(X1_train_select,Y1_train1)
ada = AdaBoostClassifier(random_state=30).fit(X1_train_select,Y1_train1)
reg = LogisticRegression(solver='newton-cg',n_jobs=-1,random_state=30).fit(X1_train_select,Y1_train1)
# %% 计算函数
#计算train verify test结果函数
def algorithm(X_frame,Y_frame,state,sampling_methods,block_list,im_list):
    for a, b, c in zip(X_frame,Y_frame,state):
        for i in sampling_methods:
            pred = i.predict(a)
            prob = i.predict_proba(a)[:,1]

            confusion = confusion_matrix(b,pred)
            tn,fp,fn,tp = confusion.ravel()
      
            AUC = roc_auc_score(b,prob)
            Precision = precision_score(b,pred)
            Accuracy = accuracy_score(b,pred)
            F1 = f1_score(b,pred)
            sen = tp/(tp+fn)
            spe = tn/(tn+fp)
            block_list.append([c,i,AUC,Precision,Accuracy,F1,sen,spe])
            if c =='test':
                if i in [xgb,rf,dt,ada]:
                    im = pd.DataFrame({'importance':i.feature_importances_,'var':a.columns})
                    im = im.sort_values(by='importance',ascending=False)
                else:
                    im = pd.DataFrame({'importance':abs(i.coef_[0]),'var':a.columns})
                    im = im.sort_values(by='importance',ascending=False)
                im_list.append([i,np.array(im).tolist()])          
    result = [block_list[i:i+5] for i in range(0,len(block_list),5)]   
    #importance = [im_list[i:i+5] for i in range(0,len(im_list),5)]       
    return result , im_list

#保存excel函数
def DF2Excel(data_path,data_list,sheet_name_list):
    write = pd.ExcelWriter(data_path ) 
    for da,sh_name  in zip(data_list,sheet_name_list):
        da.to_excel(write,sheet_name = sh_name,index=False)
    write._save()
    
# %%主函数
def main():
    #names = ['XGBoost','RandomForest','DecisionTree','AdaBoost','LogisticRegression']
    sampling_methods = [xgb,rf,dt,ada,reg]
    X_frame = [X1_train_select, X_verify_select, X1_test1]
    Y_frame = [Y1_train1,Y_verify,Y1_test]
    state = ['train','verify','test']
    block_list = []
    im_list = []
    re = algorithm(X_frame,Y_frame,state,sampling_methods,block_list,im_list)
    # 需要保存的文件地址及文件名
    data_path = 'C:\\Users\\Dell\\Desktop\\result_lungdamage3.xlsx'
    # 要保存的每个sheet 名称
    col = ['状态','算法','AUC','Precision','Accuracy','F1','sen','spe']
    # dataframe list 
    df0 = pd.DataFrame(re[0][0],columns=col)
    df1 = pd.DataFrame(re[0][1],columns=col)
    df2 = pd.DataFrame(re[0][2],columns=col)
    df3 = pd.DataFrame(re[1])
    data_list = [df0,df1,df2,df3]
    sheet_name = ['train','verify','test','importance-test']
    # 调用函数 
    DF2Excel(data_path,data_list,sheet_name)
    

main()  


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
#         plt.savefig('G:\\0 工作备份\\000work\\1 项目\\5-数据服务\\23 肾脏病医学部\\肌酸激酶\\数据导出\\4血液透析-roc.png')
#             
# =============================================================================
    return plt

colors =['crimson','orange','gold','mediumseagreen','steelblue', 'mediumpurple'  ]
names  =['XGBoost','RandomForest','DecisionTree','AdaBoost','LogisticRegression']
sampling_methods = [xgb,rf,dt,ada,reg]



train_roc_graph = multi_models_roc(names, sampling_methods, colors, X1_test1, Y1_test, save = True)
#train_roc_graph.savefig('G:\\0 工作备份\\000work\\1 项目\\5-数据服务\\23 肾脏病医学部\\肌酸激酶\\数据导出\\1住院时长-roc.png')

f = plt.gcf()
f.savefig('G:\\1 项目\\5-数据服务\\23 肾脏病医学部\\肌酸激酶\\数据导出\\6心脏损伤-roc.png')
plt.show()

#%%shap图
#保留特征筛选数据&模型输出后，增加了shap需求。此处直接导入特征筛选后数据。
import shap
import matplotlib.pyplot as plt
#plt.rcParams['font.sans-serif'] = ['SimHei']
#plt.rcParams['axes.unicode_minus'] = False


data = pd.read_excel("G:/1 项目/5-数据服务/23 肾脏病医学部/肌酸激酶/数据导出/6心脏损伤.xlsx",sheet_name='train')
X1_train_select = data.iloc[:,:-1]
Y1_train1 = data.iloc[:, -1]

data1 = pd.read_excel("G:/1 项目/5-数据服务/23 肾脏病医学部/肌酸激酶/数据导出/6心脏损伤.xlsx",sheet_name='test')
X1_test1 = data1.iloc[:,:-1]

#im = pd.DataFrame({'importance':xgb.feature_importances_,'var':X1_train_select.columns})
#im = im.sort_values(by='importance',ascending=False)

xgb = XGBClassifier(seed=30).fit(X1_train_select,Y1_train1)
#rf = RandomForestClassifier(random_state=30).fit(X1_train_select,Y1_train1)

explainer = shap.TreeExplainer(xgb)
#输出numpy.array数组
shap_values = explainer.shap_values(X1_test1)
#输出shap.Explanation对象
explainer2 = explainer(X1_test1) 


#整体柱状图
plt.clf()
shap.summary_plot(shap_values, X1_test1 , plot_type= 'bar',show=False)
plt.savefig('G:/1 项目/5-数据服务/23 肾脏病医学部/肌酸激酶/数据导出/1.png',dpi=300)
#整体
plt.clf()
shap.summary_plot(shap_values,X1_test1,show=False)
plt.savefig('G:/1 项目/5-数据服务/23 肾脏病医学部/肌酸激酶/数据导出/2.png',dpi=300)
#单个样本
plt.clf()
plt.figure(figsize=(90, 30))
shap.force_plot(explainer.expected_value, shap_values[1,:],X1_test1.iloc[1,:],matplotlib=True,show=False)
plt.savefig('G:/1 项目/5-数据服务/23 肾脏病医学部/肌酸激酶/数据导出/3.png',dpi=300)


#交互图
plt.clf()
shap_interaction_values = explainer.shap_interaction_values(X1_test1)
shap.summary_plot(shap_interaction_values, X1_test1, max_display=7,show=False)
plt.savefig('G:/1 项目/5-数据服务/23 肾脏病医学部/肌酸激酶/数据导出/4.png',dpi=300)

#局部条形图
#plt.clf()
#shap.plots.bar(explainer2[1], show_data=True) #,show=False
#plt.savefig('/Users/dongdongdong/Desktop/5.png',dpi=600)

#部分依赖图
plt.clf()
shap.dependence_plot('血氧_first', shap_values,X1_test1,show=False)
plt.savefig('G:/1 项目/5-数据服务/23 肾脏病医学部/肌酸激酶/数据导出/5.png',dpi=300)



#前10变量

ll = ['肌红蛋白定量_max',
'乳酸脱氢酶_max',
'尿量_max',
'丙氨酸氨基转移酶_max',
'AGE',
'乳酸脱氢酶_min',
'尿量_min',
'凝血酶时间测定_max',
'红细胞比积测定_min',
'总蛋白_min']


for i in ll:
    plt.clf()
    shap.dependence_plot(i, shap_values,X1_test1,show=False)
    path1 = 'G:/1 项目/5-数据服务/23 肾脏病医学部/肌酸激酶/数据导出/shap图/6-心脏损伤shap/'
    path2 = '.png'
    path = path1 + i + path2
    plt.savefig( path ,dpi=300)












