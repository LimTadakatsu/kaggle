import numpy as np #數學運算
import pandas as pd #表格計算
import matplotlib
import matplotlib.pyplot as plt #繪圖 
import seaborn as sns #進階繪圖(尚未使用過
import warnings #錯誤管理(尚未使用

#看過都還沒使用過
from sklearn.ensemble import RandomForestClassifier 
#含有多棵決策樹的分類方法，結果由眾數決定
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve, train_test_split
#Cross-Validation 選擇好的特徵 
#StratifiedKFold 切分訓練集，避免過擬合
#也可以用GridSearchCV直接找到最佳參數
#learning_curve 特徵曲線，判斷模型是否過擬合了
#train_test_split 將合併的資料處理完後重新切成測試與訓練集


from sklearn.preprocessing import LabelEncoder #數據清洗 #對非連續數值或文本編碼
from sklearn.feature_selection import RFECV #交叉驗證預防過度擬合

# for display dataframe #顯示繪製圖片 
from IPython.display import display
from IPython.display import display_html

warnings.filterwarnings('ignore') #不顯示警告資訊

#讀取資料，以前用過
df_test = pd.read_csv('C:/Users/wan/Desktop/python/reference/pra/test.csv')
df_train = pd.read_csv('C:/Users/wan/Desktop/python/reference/pra/train.csv')
df_data = df_train.append(df_test)

##相關性分析

# Sex vs Survived
print(df_train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print()
# Class vs Survived，存活率高到低，P1,P2,P3  
print(df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print()
# Sex and Class vs Survived
print(df_train[['Sex', 'Pclass', 'Survived']].groupby(['Sex', 'Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print()
# SibSp vs Survived
print(df_train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print()
# Parch vs Survived
print(df_train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print()
# Family vs Survived
df_train['Family'] = df_train['SibSp'] + df_train['Parch']
print(df_train[['Family', 'Survived']].groupby(['Family'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print()
# Embark vs Survived
print(df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print()

def FeatureCorreate(datasets,dropData):
    sns.set(context="paper", font="monospace")
    sns.set(style="white")
    f, ax = plt.subplots(figsize=(10,6))
    train_corr = datasets.drop(dropData,axis=1).corr()
    sns.heatmap(train_corr, ax=ax, vmax=.9, square=True)
    ax.set_xticklabels(train_corr.index, size=15)
    ax.set_yticklabels(train_corr.columns[::-1], size=15)
    ax.set_title('Feature Corr', fontsize=20)
print("Train Features")
FeatureCorreate(df_train,'PassengerId')
plt.show()