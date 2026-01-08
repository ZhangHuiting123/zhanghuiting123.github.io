---
title: "医疗数据住院死亡风险预测分析"
collection: portfolio
type: "Data Analysis"
permalink: /portfolio/medical-risk-prediction
date: 2025-12-26
excerpt: "基于13258条医疗数据，用逻辑回归建立死亡风险预测模型，ROC-AUC达0.6879"
header:
  teaser: /images/portfolio/medical-risk-prediction/basic_statistics.png
tags:
- 医疗数据分析
- 风险预测
- 逻辑回归
- 特征重要性
- 临床建议
- 随机森林
- 数据预处理
tech_stack:
- name: Python
- name: Pandas
- name: Scikit-learn
- name: Matplotlib
- name: Seaborn
layout: "default"
---

## 项目背景
本项目基于13,258条医疗数据，分析患者年龄（age_month）和5项实验室指标，建立住院死亡风险预测模型，为临床提供风险评估工具。

## 核心实现
### 数据预处理
```python
# 数据读取
df = pd.read_csv('medical_data.csv')

# 异常值处理
df_clean = df.copy()
age_median = df_clean[df_clean['age_month'] != -1]['age_month'].median()
df_clean.loc[df_clean['age_month'] == -1, 'age_month'] = age_median

# 缺失值处理
lab_cols = [col for col in df_clean.columns if col.startswith('lab_')]
imputer = SimpleImputer(strategy='median')
df_clean[lab_cols] = imputer.fit_transform(df_clean[lab_cols])

# 数据集划分
X = df_clean.drop('HOSPITAL_EXPIRE_FLAG', axis=1)
y = df_clean['HOSPITAL_EXPIRE_FLAG']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

