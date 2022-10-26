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

def display_side_by_side(*args): #做特徵工程顯示資料時用到  #在ti_feature.py檔中會用到
    html_str='' #不確定""的用途
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)

warnings.filterwarnings('ignore') #不顯示警告資訊
# %matplotlib inline #讓繪出的圖能正常顯示 #直接改用plt.show()比較熟悉
sns.set(font_scale=1.56) #設定seabon字體顯示大小

#讀取資料，以前用過
df_test = pd.read_csv('C:/Users/wan/Desktop/python/reference/pra/test.csv')
df_train = pd.read_csv('C:/Users/wan/Desktop/python/reference/pra/train.csv')
df_data = df_train.append(df_test)


#查看資料 EDA Exploratory Data(探索性分析)
#選擇模型 抗噪強的(svm,knn,隨機森林)

#其他補充(廣泛性：對小資料集來說三個都可以，但隨機森林可以平行化 預處理：svm,knn分別超平面跟鄰近投票，但不如隨機森林用gini不純度簡便)

def DatasetsInfo(train_data,test_data): #檢查資料是否齊全、種類、型別
    df_train.info()
    print("-" * 40+"1")
    df_test.info()
DatasetsInfo(df_train,df_test)

print(df_train.describe()) #找出中位數 最大最小值等等 平均數 標準差
print("-" * 40+"2")
print(df_train.describe(include=['O']))

#找出缺值比例(數據)
def DatasetMissingPercentage(data):
    return pd.DataFrame({'DataMissingPercentage':data.isnull().sum() * 100 / len(df_train)})

DatasetMissingPercentage(df_train)
DatasetMissingPercentage(df_test)
print(DatasetMissingPercentage(df_train))
print(DatasetMissingPercentage(df_test))

#找出缺值比例(非數據或是文本)
def DatasetUniquePercentage(data):
    return pd.DataFrame({'percent_unique':data.apply(lambda x: x.unique().size/x.size*100)})

DatasetUniquePercentage(df_train)
DatasetUniquePercentage(df_test)

print(DatasetUniquePercentage(df_train))
print(DatasetUniquePercentage(df_test))

##
##分析資料


print('Id 沒有重複.') if df_train.PassengerId.nunique() == df_train.shape[0] else print('oops')
print('訓練測試有分類好.') if len(np.intersect1d(df_train.PassengerId.values, df_test.PassengerId.values))== 0 else print('oops')#0表示train,test dataset資料一致


#查看資料是否有nan並設置datasetHasNan flag   #nan為不是數字的數字(例如除以0的結果)
if df_train.count().min() == df_train.shape[0] and df_testset.count().min() == df_testset.shape[0] :
    print('缺值可補.') 
else:
    nas = pd.concat([df_train.isnull().sum(), df_test.isnull().sum()], axis=1, keys=['Train Dataset', 'Test Dataset']) 
    print('有Nan存在')
    print(nas[nas.sum(axis=1) > 0])

