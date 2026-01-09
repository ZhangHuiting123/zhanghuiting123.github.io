---
title: "医疗数据住院死亡风险预测分析"
collection: portfolio
type: "Machine Learning"
permalink: /portfolio/medical-mortality-prediction
date: 2026-01-08 # 请替换为实际项目日期
excerpt: "基于13,258条医疗数据建立住院死亡风险预测模型，辅助临床风险评估与决策"
header:
  teaser: /images/portfolio/medical-mortality-prediction/basic_statistics.png
tags:
  - 医疗数据分析
  - 死亡风险预测
  - 机器学习
tech_stack:
  - name: Python
  - name: Scikit-learn
  - name: Pandas
  - name: Matplotlib
  - name: Seaborn
---

## 项目背景  
本项目针对13,258条住院患者数据，分析年龄（月）及5项实验室指标与死亡风险的关联，构建可解释的预测模型，为临床提供高风险患者识别工具，降低不良事件发生率。


## 核心实现  
### 1. 数据预处理  
处理异常值、缺失值并划分数据集：  
```python
# 年龄异常值替换（-1→中位数）
age_median = df_clean[df_clean['age_month'] != -1]['age_month'].median()
df_clean.loc[df_clean['age_month'] == -1, 'age_month'] = age_median

# 实验室指标缺失值填充（中位数策略）
lab_cols = [col for col in df_clean.columns if col.startswith('lab_')]
imputer = SimpleImputer(strategy='median')
df_clean[lab_cols] = imputer.fit_transform(df_clean[lab_cols])

# 分层抽样划分数据集（8:2）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 逻辑回归模型（平衡类别权重）
lr_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
lr_model.fit(X_train_scaled, y_train)

# 随机森林模型
rf_model = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
rf_model.fit(X_train_scaled, y_train)

# 模型评估函数
def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    return {'accuracy': accuracy, 'recall': recall, 'roc_auc': roc_auc}
