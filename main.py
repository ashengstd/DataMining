# %% [markdown]
# # 引入必要库

# %%
#import needed libraries、
import pandas as pd
import numpy as np
import torch
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.over_sampling import RandomOverSampler, SMOTE
import matplotlib.pyplot as plt
from VAE_pipeline import train_vae

# 选择设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# # 数据准备

# %%
df_application_record = pd.read_csv("application_record.csv")
df_credit_record = pd.read_csv("credit_record.csv")

print(df_application_record.info())
print(df_application_record.nunique())

# %%
#For each set of duplicate ID's drop both of them
df_application_record = df_application_record.drop_duplicates(subset = 'ID', keep = False)
print(df_application_record.info())

# %%
print(df_credit_record.info())

# %%
#show how many unique IDs we will be able to work with in the dataframes
print("# of unique IDs that are consistent between both datasets", df_application_record[df_application_record['ID'].isin(df_credit_record['ID'])]['ID'].nunique())

#adjust the dataframes so that we only work with the consistent IDs
df_application_record = df_application_record[df_application_record['ID'].isin(df_credit_record['ID'])]
df_credit_record = df_credit_record[df_credit_record['ID'].isin(df_application_record['ID'])]
print("New # of IDs in application_record", df_application_record['ID'].nunique())
print("New # of IDs in credit_record", df_credit_record['ID'].nunique())

# %% [markdown]
# # 数据清洗

# %%
df_credit_record['APPROVED'] = df_credit_record['STATUS'].map({'1':0,'2':0,'3':0,'4':0,'5':0,'X':-1,'C':1,'0':1})
df_credit_record = df_credit_record[df_credit_record['APPROVED']!=-1]
print(df_credit_record['STATUS'].value_counts())

# %%
df_application_record = df_application_record.merge(df_credit_record, on='ID')
print(df_application_record.head())

# %%
df_application_record = df_application_record[df_application_record['MONTHS_BALANCE']==-4]
df_application_record.loc[df_application_record["NAME_INCOME_TYPE"]=="Pensioner","OCCUPATION_TYPE"] = "Pension"
df_application_record.loc[df_application_record["NAME_INCOME_TYPE"]=="Commercial associate","OCCUPATION_TYPE"] = "Commercial associate"
df_application_record.loc[df_application_record["NAME_INCOME_TYPE"]=="State servant","OCCUPATION_TYPE"] = "State servant"
df_application_record.loc[df_application_record["NAME_INCOME_TYPE"]=="Student","OCCUPATION_TYPE"] = "Student"
df_application_record = df_application_record.dropna()
print(df_application_record.isna().sum())

# %%
df_application_record['Work_Time'] = -(df_application_record['DAYS_EMPLOYED'])//365

df_application_record = df_application_record.drop(df_application_record[df_application_record['Work_Time']>50].index)
df_application_record = df_application_record.drop(df_application_record[df_application_record['Work_Time']<0].index)
# df_application_record['Work_Time'].plot(kind='hist',bins=20,density=True)
df_application_record = df_application_record.drop(columns=['STATUS'])
df_application_record.drop(['DAYS_EMPLOYED'],axis=1,inplace=True)
print(df_application_record.head())


# %%
baseline_date = pd.to_datetime('2023-01-01')
df_application_record['BIRTH_DATE'] = baseline_date + pd.to_timedelta(df_application_record['DAYS_BIRTH'], unit='D')
df_application_record['AGE'] = (baseline_date - df_application_record['BIRTH_DATE']).dt.days // 365
df_application_record = df_application_record.drop(columns=['DAYS_BIRTH','BIRTH_DATE'])
print(df_application_record.isna().sum())

# %%
categorical_columns = ['CODE_GENDER', 'NAME_FAMILY_STATUS', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
dummy_columns = ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE']
categorical_df = df_application_record[categorical_columns]
categorical_df = categorical_df.apply(lambda x: pd.factorize(x)[0])
categorical_df = pd.DataFrame(categorical_df)
df_application_record[categorical_columns] = categorical_df
df_application_record = pd.get_dummies(df_application_record, columns=dummy_columns)
print(df_application_record['APPROVED'].value_counts())
df_application_record.to_csv('dataset.csv', index=False)
print(df_application_record.columns)
print(df_application_record.head())


# %% [markdown]
# # 创建数据集

# %%
# scaler = MinMaxScaler()
# df_application_record['AMT_INCOME_TOTAL']=scaler.fit_transform(df_application_record['AMT_INCOME_TOTAL'].values.reshape(-1, 1))
# df_application_record['DAYS_EMPLOYED']=scaler.fit_transform(df_application_record['DAYS_EMPLOYED'].values.reshape(-1, 1))
# df_application_record['MONTHS_BALANCE']=scaler.fit_transform(df_application_record['MONTHS_BALANCE'].values.reshape(-1, 1))
# scaler = StandardScaler()
# df_application_record['CNT_FAM_MEMBERS']=scaler.fit_transform(df_application_record['CNT_FAM_MEMBERS'].values.reshape(-1, 1))
# df_application_record['AGE']=scaler.fit_transform(df_application_record['AGE'].values.reshape(-1, 1))

negative_data_orgin = df_application_record[df_application_record['APPROVED']==0]
negative_data = negative_data_orgin.drop(['APPROVED', 'ID','CODE_GENDER'], axis = 1)

X = df_application_record.drop(['APPROVED', 'ID','CODE_GENDER'], axis = 1)
y = df_application_record['APPROVED']
X = np.array(X,dtype=float)
y = np.array(y, dtype=int)

X_train, X_test, y_train, y_test = train_test_split(X, y)

scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.fit_transform(X_test)

negative_data = scalar.fit_transform(np.array(negative_data,dtype=float))
negative_label_list = np.zeros(len(negative_data))


# 创建 RandomUnderSampler 对象
undersampler = RandomUnderSampler(sampling_strategy='majority')

# 使用 RandomUnderSampler 来生成平衡的训练集
X_train_under_random, y_train_under_random = undersampler.fit_resample(X_train, y_train)

# 创建RandomOverSampler对象
oversampler = RandomOverSampler(sampling_strategy='minority')

# 使用RandomOverSampler来生成平衡的训练集
X_train_over_random, y_train_over_random = oversampler.fit_resample(X_train, y_train)

# 创建TomekLinks对象
undersampler = TomekLinks()

# 使用TomekLinks来生成平衡的训练集
X_train_under_tomelinks, y_train_under_tomelinks = undersampler.fit_resample(X_train, y_train)

# 创建SMOTE对象
smote = SMOTE(sampling_strategy='minority',random_state=42)

# 使用SMOTE来生成平衡的训练集
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# %%
data_list = [
             (X_train, y_train, "Original Data"),
            #  (X_train_over_random,y_train_over_random, "Over-sampled Data"),
             (X_train_under_random,y_train_under_random, "Under-sampled Data"),
            #  (X_train_under_tomelinks,y_train_under_tomelinks, "Tomelinks Data"),
             (X_train_smote,y_train_smote, "SMOTE Data")
            ]


# %% [markdown]
# # 分类

# %%
performance_data= []

# %% [markdown]
# ## LightGBM

# %%
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metrics': 'binary_logloss',
    'learning_rate': 0.05,
    'reg_lambda': 1.,
    'reg_alpha': .1,
    'max_depth': 5,
    'n_estimators': 100,
    'colsample_bytree': .5,
    'min_child_samples': 100,
    'subsample': .9,
    'importance_type': 'gain',
    'random_state': 71,
    'num_leaves': 32,
    'force_col_wise': True,
    'scale_pos_weight': 1,
    'bagging_freq': 5,
}


# %%
index = 0

performance_data = []
for X_train_processed, y_train_processed, method_name in data_list:
    lgb_model = lgb.LGBMClassifier(**params, verbose=-1)

    lgb_model.fit(X_train_processed, y_train_processed)

    print(len(negative_data))
    y_pred = lgb_model.predict(X_test)
    y_prob = lgb_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    negative_data_pred = lgb_model.predict(negative_data)
    negative_accuracy = accuracy_score(negative_label_list, negative_data_pred)

    print(
        classification_report(negative_label_list,
                              negative_data_pred,
                              zero_division=1))

    performance_data.append({
        'Classification Method' : 'LightGBM',
        'Data Process Method': method_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Negative Accuracy': negative_accuracy,
    })
    index += 1


# %% [markdown]
# ## RandomForest

# %%
for X_train_processed, y_train_processed, method_name in data_list:
    rfc = RandomForestClassifier(n_estimators=1000, max_features=12)
    rfc.fit(X_train_processed, y_train_processed)
    predictions = rfc.predict(X_test)
    print(f"Classification Report for {method_name} on Test Data:")
    print(classification_report(y_test, predictions))
    negative_predictions = rfc.predict(negative_data)
    print(f"Classification Report for {method_name} on Negative Data:")
    print(
        classification_report(negative_label_list,
                              negative_predictions,
                              zero_division=1))
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    performance_data.append({
        'Classification Method': 'Random Forest',
        'Data Process Method': method_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Negative Accuracy': negative_accuracy,
    })

# %% [markdown]
# ## AdaBoost for Desicion Tree

# %%
for X_train_processed, y_train_processed, method_name in data_list:
    # 使用决策树桩作为弱分类器，也可以选择其他弱分类器
    base_classifier = DecisionTreeClassifier(max_depth=1)

    # 使用AdaBoost分类器
    adaboost = AdaBoostClassifier(base_classifier,
                                  n_estimators=1000,
                                  algorithm='SAMME',
                                  random_state=42)

    # 训练模型
    adaboost.fit(X_train_processed, y_train_processed)

    # 在测试集上进行预测和评估
    predictions_test = adaboost.predict(X_test)
    print(f"Classification Report for {method_name} on Test Data:")
    print(classification_report(y_test, predictions_test))

    # 在负样本数据上进行预测和评估
    predictions_negative = adaboost.predict(negative_data)
    print(f"Classification Report for {method_name} on Negative Data:")
    print(
        classification_report(negative_label_list,
                              predictions_negative,
                              zero_division=1))
    accuracy = accuracy_score(y_test, predictions_test)
    precision = precision_score(y_test, predictions_test)
    recall = recall_score(y_test, predictions_test)
    f1 = f1_score(y_test, predictions_test)
    performance_data.append({
        'Classification Method': 'AdaBoost for Decision Tree',
        'Data Process Method': method_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Negative Accuracy': negative_accuracy,
    })

# %% [markdown]
# ## SVM

# %%
for X_train_processed, y_train_processed, method_name in data_list:
   # 使用 SVM 替代 RandomForestClassifier
   svm_model = SVC()
   svm_model.fit(X_train_processed, y_train_processed)

   # SVM 在测试集上的分类报告
   predictions_svm_test = svm_model.predict(X_test)
   print(f"Method: {method_name} - SVM Classification Report on Test Data:")
   print(classification_report(y_test, predictions_svm_test))

   # SVM 在负样本上的分类报告
   predictions_svm_negative = svm_model.predict(negative_data)
   print(
      f"Method: {method_name} - SVM Classification Report on Negative Data:")
   print(
      classification_report(negative_label_list,
                           predictions_negative,
                           zero_division=1))
   accuracy = accuracy_score(y_test, predictions_svm_test)
   precision = precision_score(y_test, predictions_svm_test)
   recall = recall_score(y_test, predictions_svm_test)
   f1 = f1_score(y_test, predictions_svm_test)
   performance_data.append({
       'Classification Method': 'SVM',
       'Data Process Method': method_name,
       'Accuracy': accuracy,
       'Precision': precision,
       'Recall': recall,
       'F1 Score': f1,
       'Negative Accuracy': negative_accuracy,
   })


# %% [markdown]
# ## AdaBoost for SVM

# %%
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report

# for X_train_processed, y_train_processed, method_name in data_list:
#     # 使用支持向量机作为弱分类器
#     base_classifier = SVC(kernel='linear', C=1.0)

#     # 使用AdaBoost分类器
#     adaboost = AdaBoostClassifier(base_classifier,
#                                   n_estimators=1000,
#                                   algorithm='SAMME',
#                                   random_state=42)

#     # 训练模型
#     adaboost.fit(X_train_processed, y_train_processed)

#     # 在测试集上进行预测和评估
#     predictions_test = adaboost.predict(X_test)
#     print(f"Classification Report for {method_name} on Test Data:")
#     print(classification_report(y_test, predictions_test))

#     # 在负样本数据上进行预测和评估
#     predictions_negative = adaboost.predict(negative_data)
#     print(f"Classification Report for {method_name} on Negative Data:")
#     print(
#         classification_report(negative_label_list,
#                               predictions_negative,
#                               zero_division=1))
#     accuracy = accuracy_score(y_test, predictions_test)
#     precision = precision_score(y_test, predictions_test)
#     recall = recall_score(y_test, predictions_test)
#     f1 = f1_score(y_test, predictions_test)
#     performance_data.append({
#         'Classification Method': 'AdaBoost for SVM',
#         'Data Process Method': method_name,
#         'Accuracy': accuracy,
#         'Precision': precision,
#         'Recall': recall,
#         'F1 Score': f1,
#         'Negative Accuracy': negative_accuracy,
#     })

# %%
# index = 0
# for X_train_processed, y_train_processed, method_name in data_list:
#     svm_model = SVC()
#     train_feature = train_features[index]
#     test_feature = test_features[index]
#     svm_model.fit(train_feature, y_train_processed)
#     predictions = svm_model.predict(test_feature)
#     print(classification_report(y_test,predictions))
#     predictions = svm_model.predict(negative_data_list[index])
#    print(classification_report(negative_label_list, predictions_negative, zero_division=1))
#     index = index + 1

# %% [markdown]
# # 特征提取

# %%
model_list = []
for X_train_processed, _, method_name in data_list:
   print("Training for the method: " + method_name)
   model = train_vae(X_train=X_train_processed,X_test=X_test, progress=False,num_epoch=100).eval()
   model_list.append(model)

total_list = [ t+ (data,)  for t, data in zip(data_list, model_list)]

index = 0
train_features = []
test_features = []
negative_data_list = []
for X_train_processed, _, method_name, __ in total_list:
    train_features.append(model_list[index].encoder(torch.Tensor(X_train_processed).to(device)).cpu().detach().numpy())
    test_features.append(model_list[index].encoder(torch.Tensor(X_test).to(device)).detach().cpu().numpy())
    negative_data_list.append(model_list[index].encoder(torch.Tensor(negative_data).to(device)).detach().cpu().numpy())
    index+=1

   
# for X_train_processed, _, method_name, model in model_list:
#     model.eval()
#     with torch.no_grad():
#         encoded_data = model.encoder(torch.Tensor(X_train_processed).to(device))
#         encoded_data = encoded_data.cpu().numpy()
#         tsne = TSNE(n_components=2)
#         reduced_data = tsne.fit_transform(encoded_data)

# plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
# plt.title("VAE Visualization")
# plt.show()

# %% [markdown]
# ## LightGBM for VAE data

# %%
index = 0

performance_data = []
data_df = pd.DataFrame(columns=['Method', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Negative Accuracy'])

for X_train_processed, y_train_processed, method_name, _ in total_list:
    lgb_model = lgb.LGBMClassifier(**params, verbose=-1)

    lgb_model.fit(train_features[index], y_train_processed)

    print(len(negative_data))
    y_pred = lgb_model.predict(test_features[index])
    y_prob = lgb_model.predict_proba(test_features[index])[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(classification_report(y_test, y_pred))
    negative_data_pred = lgb_model.predict(negative_data_list[index])
    negative_accuracy = accuracy_score(negative_label_list, negative_data_pred)

    print(
        classification_report(negative_label_list,
                              negative_data_pred,
                              zero_division=1))

    performance_data.append({
        'Classification Method': 'LightGBM for VAE data',
        'Data Process Method': method_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Negative Accuracy': negative_accuracy,
    })
    index += 1

data_df = pd.DataFrame(performance_data)
print(data_df)


# %% [markdown]
# ## SVM for VAE data

# %%
index = 0
for X_train_processed, y_train_processed, method_name in data_list:
    svm_model = SVC()
    train_feature = train_features[index]
    test_feature = test_features[index]
    svm_model.fit(train_feature, y_train_processed)
    predictions = svm_model.predict(test_feature)
    print(classification_report(y_test, predictions))
    negative_predictions= svm_model.predict(negative_data_list[index])
    print(
        classification_report(negative_label_list,
                              negative_predictions,
                              zero_division=1))
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    negative_accuracy = accuracy_score(negative_label_list, negative_predictions)
    performance_data.append({
        'Classification Method':
        'SVM for VAE data',
        'Data Process Method':
        method_name,
        'Accuracy':
        accuracy,
        'Precision':
        precision,
        'Recall':
        recall,
        'F1 Score':
        f1,
        'Negative Accuracy':
        negative_accuracy,
    })
    index = index + 1


# %% [markdown]
# ## AdaBoost for Desicion Tree for VAE data

# %%
index = 0
for X_train_processed, y_train_processed, method_name in data_list:

    train_feature = train_features[index]
    test_feature = test_features[index]
    # 使用决策树桩作为弱分类器，也可以选择其他弱分类器
    base_classifier = DecisionTreeClassifier(max_depth=1)

    # 使用AdaBoost分类器
    adaboost = AdaBoostClassifier(base_classifier,
                                  n_estimators=1000,
                                  algorithm='SAMME',
                                  random_state=42)

    # 训练模型
    adaboost.fit(train_feature, y_train_processed)

    # 在测试集上进行预测和评估
    predictions_test = adaboost.predict(test_feature)
    print(f"Classification Report for {method_name} on Test Data:")
    print(classification_report(y_test, predictions_test))

    # 在负样本数据上进行预测和评估
    predictions_negative = adaboost.predict(negative_data_list[index])
    print(f"Classification Report for {method_name} on Negative Data:")
    print(
        classification_report(negative_label_list,
                              predictions_negative,
                              zero_division=1))
    accuracy = accuracy_score(y_test, predictions_test)
    precision = precision_score(y_test, predictions_test)
    recall = recall_score(y_test, predictions_test)
    f1 = f1_score(y_test, predictions_test)
    negative_accuracy = accuracy_score(negative_label_list, predictions_negative)
    performance_data.append({
        'Classification Method': 'AdaBoost for Decision Tree',
        'Data Process Method': method_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Negative Accuracy': negative_accuracy,
    })
    index += 1

# %%
data_df = pd.DataFrame(performance_data)
# 找到最佳性能数据的行
best_accuracy_row = data_df[data_df['Accuracy'] == data_df['Accuracy'].max()]
best_precision_row = data_df[data_df['Precision'] == data_df['Precision'].max()]
best_recall_row = data_df[data_df['Recall'] == data_df['Recall'].max()]
best_f1_score_row = data_df[data_df['F1 Score'] == data_df['F1 Score'].max()]
best_negative_accuracy_row = data_df[data_df['Negative Accuracy'] == data_df['Negative Accuracy'].max()]

# 输出最佳性能数据
print("best Accuracy Classification Method:", best_accuracy_row['Classification Method'].values[0])
print("Best Accuracy Data Process Method:", best_accuracy_row['Data Process Method'].values[0])
print("Best Accuracy Value:", best_accuracy_row['Accuracy'].values[0])

print("best Accuracy Classification Method:", best_precision_row['Classification Method'].values[0])
print("Best Precision Data Process Method:", best_precision_row['Data Process Method'].values[0])
print("Best Precision Value:", best_precision_row['Precision'].values[0])

print("best Accuracy Classification Method:", best_recall_row['Classification Method'].values[0])
print("Best Recall Data Process Method:", best_recall_row['Data Process Method'].values[0])
print("Best Recall Value:", best_recall_row['Recall'].values[0])

print("best Accuracy Classification Method:", best_f1_score_row['Classification Method'].values[0])
print("Best F1 Score Data Process Method:", best_f1_score_row['Data Process Method'].values[0])
print("Best F1 Score Value:", best_f1_score_row['F1 Score'].values[0])


print("best Accuracy Classification Method:", best_negative_accuracy_row['Classification Method'].values[0])
print("Best Negative Accuracy Data Process Method:", best_negative_accuracy_row['Data Process Method'].values[0])
print("Best Negative Accuracy Value:", best_negative_accuracy_row['Negative Accuracy'].values[0])
data_df.to_csv('performance_data.csv', index=False)

# %% [markdown]
# # Ensemble


