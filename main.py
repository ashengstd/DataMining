# %% [markdown]
# # 引入必要库

# %%
# 引入必要库
import pandas as pd
import numpy as np
import torch
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.over_sampling import RandomOverSampler, SMOTE
from matplotlib import pyplot as plt
from model.vae_pipeline import train_vae
from model.anomaly_detection_pipeline import train_vae_anomaly_detection

# 选择设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# # 数据准备

# %%
df_application_record = pd.read_csv("dataset/application_record.csv")
df_credit_record = pd.read_csv("dataset/credit_record.csv")

# %%
# 丢弃重复ID数据
df_application_record = df_application_record.drop_duplicates(subset = 'ID', keep = False)

# %%
# 调整数据框，以便仅使用一致的ID进行处理
df_application_record = df_application_record[df_application_record['ID'].isin(df_credit_record['ID'])]
df_credit_record = df_credit_record[df_credit_record['ID'].isin(df_application_record['ID'])]

# %% [markdown]
# # 数据清洗

# %%
# 生成标签用于柱状图
label_dict = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, 'X': -1, 'C': 1, '0': 1}
df_credit_record['APPROVED'] = df_credit_record['STATUS'].map(label_dict)
df_credit_record = df_credit_record[df_credit_record['APPROVED'] != -1]

# %%
# 合并数据
df_application_record = df_application_record.merge(df_credit_record, on='ID')

# %%
df_application_record = df_application_record[df_application_record['MONTHS_BALANCE']==-4]
df_application_record.loc[df_application_record["NAME_INCOME_TYPE"]=="Pensioner","OCCUPATION_TYPE"] = "Pension"
df_application_record.loc[df_application_record["NAME_INCOME_TYPE"]=="Commercial associate","OCCUPATION_TYPE"] = "Commercial associate"
df_application_record.loc[df_application_record["NAME_INCOME_TYPE"]=="State servant","OCCUPATION_TYPE"] = "State servant"
df_application_record.loc[df_application_record["NAME_INCOME_TYPE"]=="Student","OCCUPATION_TYPE"] = "Student"
df_application_record = df_application_record.dropna()

# %%
df_application_record['Work_Time'] = -(df_application_record['DAYS_EMPLOYED'])//365

df_application_record = df_application_record.drop(df_application_record[df_application_record['Work_Time']>50].index)
df_application_record = df_application_record.drop(df_application_record[df_application_record['Work_Time']<0].index)
df_application_record = df_application_record.drop(columns=['STATUS'])
df_application_record.drop(['DAYS_EMPLOYED'],axis=1,inplace=True)


# %%
baseline_date = pd.to_datetime('2023-01-01')
df_application_record['BIRTH_DATE'] = baseline_date + pd.to_timedelta(df_application_record['DAYS_BIRTH'], unit='D')
df_application_record['AGE'] = (baseline_date - df_application_record['BIRTH_DATE']).dt.days // 365
df_application_record = df_application_record.drop(columns=['DAYS_BIRTH','BIRTH_DATE'])

# %%
onehot = False
if onehot:
    categorical_columns = ['CODE_GENDER', 'NAME_FAMILY_STATUS', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
    dummy_columns = ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE']
else:
    categorical_columns = ['CODE_GENDER', 'NAME_FAMILY_STATUS', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE']
    dummy_columns = []
categorical_df = df_application_record[categorical_columns]
categorical_df = categorical_df.apply(lambda x: pd.factorize(x)[0])
categorical_df = pd.DataFrame(categorical_df)
df_application_record[categorical_columns] = categorical_df
df_application_record = pd.get_dummies(df_application_record, columns=dummy_columns)
df_application_record.to_csv('dataset.csv', index=False)

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
data_dict = {
    'Original Data': (X_train, y_train),
    'Over-sampled Data': (X_train_over_random, y_train_over_random),
    'Under-sampled Data': (X_train_under_random, y_train_under_random),
    'Tomelinks Data': (X_train_under_tomelinks, y_train_under_tomelinks),
    'SMOTE Data': (X_train_smote, y_train_smote)
}

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
for method_name, (X_train_processed, y_train_processed) in data_dict.items():
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
        'Classification Method': 'LightGBM',
        'Data Process Method': method_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Negative Accuracy': negative_accuracy,
    })

# %% [markdown]
# ## RandomForest

# %%
for method_name, (X_train_processed, y_train_processed) in data_dict.items():
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
# ## AdaBoost for Decision Tree

# %%
for method_name, (X_train_processed, y_train_processed) in data_dict.items():
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
for method_name, (X_train_processed, y_train_processed) in data_dict.items():
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
# for method_name, (X_train_processed, y_train_processed) in data_dict.items():
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

# %% [markdown]
# # 特征提取

# %%
model_dict = {}

for method_name, (X_train_processed, _) in data_dict.items():
    print("Training for the method: " + method_name)
    model = train_vae(X_train=X_train_processed, X_test=X_test, progress=False, num_epoch=250).eval()
    model_dict[method_name] = model

total_dict = {}

for method_name, (X_train_processed, _) in data_dict.items():
    model = model_dict[method_name]
    total_dict[method_name] = (X_train_processed, _, model)

# %%
for method_name, (X_train_processed, _, model) in total_dict.items():
    model.eval()
    with torch.no_grad():
        encoded_data = model.encoder(torch.Tensor(X_train_processed).to(device))
        encoded_data = encoded_data.cpu().numpy()
        tsne = TSNE(n_components=2)
        reduced_data = tsne.fit_transform(encoded_data)

    plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
    plt.title(f"VAE Visualization for {method_name}")
    plt.show()

# %% [markdown]
# ## LightGBM for VAE data

# %%
for method_name, (X_train_processed, y_train_processed, model) in total_dict.items():
    lgb_model = lgb.LGBMClassifier(**params, verbose=-1)

    lgb_model.fit(model.encoder(torch.Tensor(X_train_processed).to(device)).cpu().detach().numpy(), y_train_processed)

    y_pred = lgb_model.predict(model.encoder(torch.Tensor(X_test).to(device)).detach().cpu().numpy())

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(classification_report(y_test, y_pred))
    negative_data_pred = lgb_model.predict(model.encoder(torch.Tensor(negative_data).to(device)).detach().cpu().numpy())
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


# %% [markdown]
# ## SVM for VAE data

# %%
for method_name, (X_train_processed, y_train_processed, model) in total_dict.items():
    svm_model = SVC()
    train_feature = model.encoder(torch.Tensor(X_train_processed).to(device)).cpu().detach().numpy()
    test_feature = model.encoder(torch.Tensor(X_test).to(device)).detach().cpu().numpy()
    svm_model.fit(train_feature, y_train_processed)
    predictions = svm_model.predict(test_feature)
    print(classification_report(y_test, predictions))
    negative_predictions = svm_model.predict(model.encoder(torch.Tensor(negative_data).to(device)).detach().cpu().numpy())
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
        'Classification Method': 'SVM for VAE data',
        'Data Process Method': method_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Negative Accuracy': negative_accuracy,
    })

# %% [markdown]
# ## AdaBoost for Decision Tree for VAE data

# %%
for method_name, (X_train_processed, y_train_processed, model) in total_dict.items():
    train_feature = model.encoder(torch.Tensor(X_train_processed).to(device)).cpu().detach().numpy()
    test_feature = model.encoder(torch.Tensor(X_test).to(device)).detach().cpu().numpy()

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
    predictions_negative = adaboost.predict(model.encoder(torch.Tensor(negative_data).to(device)).detach().cpu().numpy())
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

# %% [markdown]
# # Ensemble

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

# %% [markdown]
# ## Stacking

# %%
# 使用决策树桩作为弱分类器，也可以选择其他弱分类器
base_classifier = DecisionTreeClassifier(max_depth=1)

# 使用AdaBoost分类器
adaboost = AdaBoostClassifier(base_classifier,
                              n_estimators=1000,
                              algorithm='SAMME',
                              random_state=42)

X_train_processed = total_dict['Under-sampled Data'][0]

# 训练模型
adaboost.fit(X_train_under_random, y_train_under_random)

# 在测试集上进行预测和评估
predictions_ada = adaboost.predict(X_test)
negative_accuracy_1 = adaboost.predict(negative_data)

# %%
# 使用LightGBM分类器
lgb_model = lgb.LGBMClassifier(**params, verbose=-1)

# 训练模型
lgb_model.fit(X_train_under_random, y_train_under_random)

# 在测试集上进行预测和评估
predictions_lgbm = lgb_model.predict(X_test)
negative_accuracy_2 = lgb_model.predict(negative_data)

# %%
# 创建元模型
meta_model = RandomForestClassifier(n_estimators=100, random_state=42)

# 将基模型的预测结果作为特征
X_ensemble = np.column_stack((predictions_ada, predictions_lgbm))
negative_predictions = np.column_stack((negative_accuracy_1,negative_accuracy_2))

# 划分数据集
X_train_ensemble, X_test_ensemble, y_train_ensemble, y_test_ensemble = train_test_split(X_ensemble, y_test, test_size=0.2, random_state=42)

# 训练元模型
meta_model.fit(X_train_ensemble, y_train_ensemble)

# 预测
ensemble_predictions = meta_model.predict(X_test_ensemble)
negative_predictions = meta_model.predict(negative_predictions)

# 计算新的指标
accuracy_ensemble = accuracy_score(y_test_ensemble, ensemble_predictions)
negative_accuracy_ensemble = accuracy_score(negative_label_list, negative_predictions)
precision_ensemble = precision_score(y_test_ensemble, ensemble_predictions)
recall_ensemble = recall_score(y_test_ensemble, ensemble_predictions)
f1_ensemble = f1_score(y_test_ensemble, ensemble_predictions)

# 打印新的指标
print("Ensemble Model Metrics:")
print(f"Accuracy: {accuracy_ensemble:.4f}")
print(f"Precision: {precision_ensemble:.4f}")
print(f"Recall: {recall_ensemble:.4f}")
print(f"F1 Score: {f1_ensemble:.4f}")
print(f"Negative Accuracy: {negative_accuracy_ensemble:.4f}")

# 打印分类报告
print("Classification Report for Ensemble Model:")
print(classification_report(y_test_ensemble, ensemble_predictions))
performance_data.append({
    'Classification Method': 'Stacking Ensemble Model for AdaBoost and LightGBM',
    'Data Process Method': 'Under-sampled Data',
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1,
    'Negative Accuracy': negative_accuracy_ensemble,
})

# %% [markdown]
# ## Voting

# %%
# 创建投票分类器
voting_model = VotingClassifier(estimators=[('adaboost', adaboost), ('lgbm', lgb_model)], voting='hard')

# 训练投票模型
voting_model.fit(X_train_under_random, y_train_under_random)

# 预测
ensemble_predictions = voting_model.predict(X_test)
negative_predictions = voting_model.predict(negative_data)

# 计算新的指标
accuracy_ensemble = accuracy_score(y_test, ensemble_predictions)
negative_accuracy_ensemble = accuracy_score(negative_label_list, negative_predictions)
precision_ensemble = precision_score(y_test, ensemble_predictions)
recall_ensemble = recall_score(y_test, ensemble_predictions)
f1_ensemble = f1_score(y_test, ensemble_predictions)

# 打印新的指标
print("Ensemble Model Metrics:")
print(f"Accuracy: {accuracy_ensemble:.4f}")
print(f"Precision: {precision_ensemble:.4f}")
print(f"Recall: {recall_ensemble:.4f}")
print(f"F1 Score: {f1_ensemble:.4f}")
print(f"Negative Accuracy: {negative_accuracy_ensemble:.4f}")
performance_data.append({
    'Classification Method': 'Voting Ensemble Model for AdaBoost and LightGBM',
    'Data Process Method': 'Under-sampled Data',
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1,
    'Negative Accuracy': negative_accuracy_ensemble,
})

# 打印分类报告
print("Classification Report for Ensemble Model:")
print(classification_report(y_test, ensemble_predictions))


# %% [markdown]
# # Anomaly Detection

# %% [markdown]
# ## VAE

# %%
Anomal_dict = {}
Anomal_model_list = []

for method_name, (X_train_processed, _) in data_dict.items():
    print("Training for the method: " + method_name)
    model = train_vae_anomaly_detection(X_train=X_train_processed, X_test=X_test, progress=True, num_epoch=250).eval()
    Anomal_model_list.append(model)
    Anomal_dict[method_name] = (X_train_processed, _, model)

# %%
for method_name, (X_train_processed, _, model) in Anomal_dict.items():
    predictions = model.predict_anomaly(
        torch.Tensor(X_test).to(device),
        threshold=0.05).detach().cpu().numpy()
    negative_data_pred = model.predict_anomaly(
        torch.Tensor(negative_data).to(device),
        threshold=0.05).detach().cpu().numpy()

    # 计算评估指标
    report_test = classification_report(y_test, predictions, output_dict=True)
    report_negative = classification_report(negative_label_list, negative_data_pred, output_dict=True, zero_division=1)

    # 提取指标
    accuracy = accuracy_score(y_test, predictions)
    precision = report_test['1']['precision']
    recall = report_test['1']['recall']
    f1 = report_test['1']['f1-score']
    negative_accuracy = accuracy_score(negative_label_list, negative_data_pred)

    print(classification_report(y_test, predictions))
    print(
        classification_report(negative_label_list,
                              negative_data_pred,
                              zero_division=1))
    # 添加到 performance_data 列表中
    performance_data.append({
        'Classification Method': 'Anomaly Detection with VAE',
        'Data Process Method': method_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Negative Accuracy': negative_accuracy,
    })

# %%
data_df = pd.DataFrame(performance_data)
data_df = data_df.round({'Accuracy': 3, 'Precision': 3, 'Recall': 3, 'F1 Score': 3, 'Negative Accuracy': 3})

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
data_df.to_csv('output/performance_data.csv', index=False)


