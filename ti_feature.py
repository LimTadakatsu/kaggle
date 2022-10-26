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
df_data = df_train.append(df_test) #train跟test加在一起處理

#轉換性別文字為數字  #map看起來是建立字典
df_data['Sex_Code'] = df_data['Sex'].map({'female' : 1, 'male' : 0}).astype('int')
#再把訓練與測試集分開
df_train = df_data[:len(df_train)]
df_test = df_data[len(df_train):]
#定義預測與預測結果資料集
Y = df_train['Survived'] #幫passengerID標上1 0的Survived標示
X = df_train.drop(labels=['Survived','PassengerId'],axis=1)
#基準模型，比他低可能加入太多噪聲的特徵，或過擬合了  #選用sex pclass 當基準模型的原因不清楚
Base = ['Sex_Code','Pclass']
Base_Model = RandomForestClassifier(random_state=2,n_estimators=250,min_samples_split=20,oob_score=True)
Base_Model.fit(X[Base], Y)
print('Base oob score :%.5f' %(Base_Model.oob_score_)) #out of bag (OOB效果好所用資源少，是一種好的驗證方式) #%.5f 輸出小數,即保留小數點後5位數字

df_data.info()

#######
# there is some bugs in log-scale of boxplot. 
# alternatively, we transform x into log10(x) for visualization.
#######這段以後再處理

fig, ax = plt.subplots( figsize = (18,7) )
df_data['Log_Fare'] = (df_data['Fare']+1).map(lambda x : np.log10(x) if x > 0 else 0)
sns.boxplot(y='Pclass', x='Log_Fare',hue='Survived',data=df_data, orient='h' #sns.boxplot箱線圖
                ,ax=ax,palette="Set3")
ax.set_title(' Log_Fare & Pclass vs Survived ',fontsize = 20) #設置圖表標題
pd.pivot_table(df_data,values = ['Fare'], index = ['Pclass'], columns= ['Survived'] ,aggfunc = 'median' ).round(3) #樞紐分析表幫助閱讀

print(pd.pivot_table(df_data,values = ['Fare'], index = ['Pclass'], columns= ['Survived'] ,aggfunc = 'median' ).round(3))
# plt.show()

#只有一個缺失，填中位數就可以了，還要適當切分票價避免過擬合
# Filling missing values
df_data['Fare'] = df_data['Fare'].fillna(df_data['Fare'].median())

# 用pandas qcut以累積百分比來切分 
df_data['FareBin_4'] = pd.qcut(df_data['Fare'], 4) #切四分切五分切六份來驗證  ####下面會發現切六份效果最好
df_data['FareBin_5'] = pd.qcut(df_data['Fare'], 5)
df_data['FareBin_6'] = pd.qcut(df_data['Fare'], 6)

label = LabelEncoder()
df_data['FareBin_Code_4'] = label.fit_transform(df_data['FareBin_4'])
df_data['FareBin_Code_5'] = label.fit_transform(df_data['FareBin_5'])
df_data['FareBin_Code_6'] = label.fit_transform(df_data['FareBin_6'])

# cross tab
df_4 = pd.crosstab(df_data['FareBin_Code_4'],df_data['Pclass'])
df_5 = pd.crosstab(df_data['FareBin_Code_5'],df_data['Pclass'])
df_6 = pd.crosstab(df_data['FareBin_Code_6'],df_data['Pclass'])

def display_side_by_side(*args): #做特徵工程顯示資料時用到  #在ti_feature.py檔中會用到
    html_str='' #不確定""的用途
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)

display_side_by_side(df_4,df_5,df_6)

####################################################
# plots #########繪圖沒修好
fig, [ax1, ax2, ax3] = plt.subplots(1, 3,sharey=True)
fig.set_figwidth(18)
for axi in [ax1, ax2, ax3]:
    axi.axhline(0.5,linestyle='dashed', c='black',alpha = .3)
g1 = sns.factorplot(x='FareBin_Code_4', y="Survived", data=df_data,kind='bar',ax=ax1)
g2 = sns.factorplot(x='FareBin_Code_5', y="Survived", data=df_data,kind='bar',ax=ax2)
g3 = sns.factorplot(x='FareBin_Code_6', y="Survived", data=df_data,kind='bar',ax=ax3)


# close FacetGrid object
plt.close(g1.fig)
plt.close(g2.fig)
plt.close(g3.fig)
# plt.show()

#####文字沒出來

####################################################

# fare切割完再分一次訓練與測試集
df_train = df_data[:len(df_train)]
df_test = df_data[len(df_train):]
# Training set and labels
X = df_train.drop(labels=['Survived','PassengerId'],axis=1)
Y = df_train['Survived']
# show columns
X.columns
print(X.columns)

#RFE選擇特徵，跟Chi square、或是information gain比可以考慮到特徵之間的交互作用
compare = ['Sex_Code','Pclass','FareBin_Code_4','FareBin_Code_5','FareBin_Code_6']
selector = RFECV(RandomForestClassifier(n_estimators=250,min_samples_split=20),cv=10,n_jobs=-1)
selector.fit(X[compare], Y)
print(selector.support_)
print(selector.ranking_)
print(selector.grid_scores_*100) #############出來的數字過多之後要修
#看來切6份比較好，但還要驗證

score_b4,score_b5, score_b6 = [], [], []
seeds = 10
for i in range(seeds):
    diff_cv = StratifiedKFold(n_splits=10,shuffle=True,random_state=i)
    selector = RFECV(RandomForestClassifier(random_state=i,n_estimators=250,min_samples_split=20),cv=diff_cv,n_jobs=-1)
    selector.fit(X[compare], Y)
    score_b4.append(selector.grid_scores_[2])
    score_b5.append(selector.grid_scores_[3])
    score_b6.append(selector.grid_scores_[4])
# to np.array
score_list = [score_b4, score_b5, score_b6]
for item in score_list:
    item = np.array(item*100)
# plot 繪圖區############上面的產出數字太多，繪出的圖太詭異要修
fig = plt.figure(figsize= (18,8) )
ax = plt.gca()
ax.plot(range(seeds), score_b4,'-ok',label='bins = 4')
ax.plot(range(seeds), score_b5,'-og',label='bins = 5')
ax.plot(range(seeds), score_b6,'-ob',label='bins = 6')
ax.set_xlabel("Seed #", fontsize = '14')
ax.set_ylim(0.783,0.815)
ax.set_ylabel("Accuracy", fontsize = '14')
ax.set_title('bins = 4 vs bins = 5 vs bins = 6', fontsize='20')
plt.legend(fontsize = 14,loc='upper right')


###OOB判斷切5分就好
b4, b5, b6 = ['Sex_Code', 'Pclass','FareBin_Code_4'], ['Sex_Code','Pclass','FareBin_Code_5'],\
['Sex_Code','Pclass','FareBin_Code_6']
b4_Model = RandomForestClassifier(random_state=2,n_estimators=250,min_samples_split=20,oob_score=True)
b4_Model.fit(X[b4], Y)
b5_Model = RandomForestClassifier(random_state=2,n_estimators=250,min_samples_split=20,oob_score=True)
b5_Model.fit(X[b5], Y)
b6_Model = RandomForestClassifier(random_state=2,n_estimators=250,min_samples_split=20,oob_score=True)
b6_Model.fit(X[b6], Y)
print('b4 oob score :%.5f' %(b4_Model.oob_score_),'   LB_Public : 0.7790')
print('b5 oob score :%.5f '%(b5_Model.oob_score_),' LB_Public : 0.79425')
print('b6 oob score : %.5f' %(b6_Model.oob_score_), '  LB_Public : 0.77033')


#ticket補值
df_train['Ticket'].describe()

#有人有相同車票，可能是家人朋友

# Family_size
df_data['Family_size'] = df_data['SibSp'] + df_data['Parch'] + 1
#觀察相同票根‘那些人的姓名、票價、艙位、家庭人數  
deplicate_ticket = []
for tk in df_data.Ticket.unique():
    tem = df_data.loc[df_data.Ticket == tk, 'Fare']
    #print(tem.count())
    if tem.count() > 1:
        #print(df_data.loc[df_data.Ticket == tk,['Name','Ticket','Fare']])
        deplicate_ticket.append(df_data.loc[df_data.Ticket == tk,['Name','Ticket','Fare','Cabin','Family_size','Survived']])
deplicate_ticket = pd.concat(deplicate_ticket)
deplicate_ticket.head(14)

#可以從票跟姓名看出那天上船情況，接下來Family_size 分類，1朋友，>1家人

df_fri = deplicate_ticket.loc[(deplicate_ticket.Family_size == 1) & (deplicate_ticket.Survived.notnull())].head(7)
df_fami = deplicate_ticket.loc[(deplicate_ticket.Family_size > 1) & (deplicate_ticket.Survived.notnull())].head(7)
display(df_fri,df_fami)
print('people keep the same ticket: %.0f '%len(deplicate_ticket))
print('friends: %.0f '%len(deplicate_ticket[deplicate_ticket.Family_size == 1]))
print('families: %.0f '%len(deplicate_ticket[deplicate_ticket.Family_size > 1]))

# 1.過濾出重複的票根 : if( len(df_grp) > 1)
# 2.如果群組中有人生還 則定義 Connected_Survival = 1 : if(smax == 1.0):
# 3.沒有人生還，則定義Connected_Survival = 0 : if( smin == 0.0):
# 4.剩下的沒有生還資訊，定義Connected_Survival = 0.5 : 程式碼第一行 df_data['Connected_Survival'] = 0.5

# the same ticket family or friends
df_data['Connected_Survival'] = 0.5 # default 
for _, df_grp in df_data.groupby('Ticket'):
    if (len(df_grp) > 1):
        for ind, row in df_grp.iterrows():
            smax = df_grp.drop(ind)['Survived'].max()
            smin = df_grp.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                df_data.loc[df_data['PassengerId'] == passID, 'Connected_Survival'] = 1
            elif (smin==0.0):
                df_data.loc[df_data['PassengerId'] == passID, 'Connected_Survival'] = 0
#print
print('people keep the same ticket: %.0f '%len(deplicate_ticket))
print("people have connected information : %.0f" 
      %(df_data[df_data['Connected_Survival']!=0.5].shape[0]))
df_data.groupby('Connected_Survival')[['Survived']].mean().round(3)

# connected information連結關係(0 or 1 )

df_train = df_data[:len(df_train)]
df_test = df_data[len(df_train):]
# Training set and labels
X = df_train.drop(labels=['Survived','PassengerId'],axis=1)
Y = df_train['Survived']
connect = ['Sex_Code','Pclass','FareBin_Code_5','Connected_Survival']
connect_Model = RandomForestClassifier(random_state=2,n_estimators=250,min_samples_split=20
                                       ,oob_score=True)
connect_Model.fit(X[connect], Y)
print('connect oob score :%.5f' %(connect_Model.oob_score_))


##age補值


df_data.info()

#觀察缺失值分佈  
df_data['Has_Age'] = df_data['Age'].isnull().map(lambda x : 0 if x == True else 1)
fig, [ax1, ax2] = plt.subplots(1, 2)
fig.set_figwidth(18)
ax1 = sns.countplot(df_data['Pclass'],hue=df_data['Has_Age'],ax=ax1)
ax2 = sns.countplot(df_data['Sex'],hue=df_data['Has_Age'],ax=ax2)
pd.crosstab(df_data['Has_Age'],df_data['Sex'],margins=True).round(3)

# Masks
Mask_Has_Age_P12_Survived = ( (df_data.Has_Age == 1) & (df_data.Pclass != 3 ) & (df_data.Survived == 1) )
Mask_Has_Age_P12_Dead = ( (df_data.Has_Age == 1) & (df_data.Pclass != 3 ) & (df_data.Survived == 0) )
# Plot
fig, ax = plt.subplots( figsize = (15,9) )
ax = sns.distplot(df_data.loc[Mask_Has_Age_P12_Survived, 'Age'],kde=False,bins=10,norm_hist=True,label='Survived') 
ax = sns.distplot(df_data.loc[Mask_Has_Age_P12_Dead, 'Age'],kde=False,bins=10,norm_hist=True,label='Dead')
ax.legend()
ax.set_title('Age vs Survived in Pclass = 1 and  2',fontsize = 20)

# extracted title using name
df_data['Title'] = df_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
df_data['Title'] = df_data['Title'].replace(['Capt', 'Col', 'Countess', 'Don',
                                               'Dr', 'Dona', 'Jonkheer', 
                                                'Major','Rev','Sir'],'Rare') 
df_data['Title'] = df_data['Title'].replace(['Mlle', 'Ms','Mme'],'Miss')
df_data['Title'] = df_data['Title'].replace(['Lady'],'Mrs')
df_data['Title'] = df_data['Title'].map({"Mr":0, "Rare" : 1, "Master" : 2,"Miss" : 3, "Mrs" : 4 })
Ti = df_data.groupby('Title')['Age'].median()
# Ti

Ti_pred = df_data.groupby('Title')['Age'].median().values
df_data['Ti_Age'] = df_data['Age']
# Filling the missing age
for i in range(0,5):
 # 0 1 2 3 4 5
    df_data.loc[(df_data.Age.isnull()) & (df_data.Title == i),'Ti_Age'] = Ti_pred[i]
df_data['Ti_Age'] = df_data['Ti_Age'].astype('int')
df_data['Ti_Minor'] = ((df_data['Ti_Age']) < 16.0) * 1
# splits again beacuse we just engineered new feature
df_train = df_data[:len(df_train)]
df_test = df_data[len(df_train):]
# Training set and labels
X = df_train.drop(labels=['Survived','PassengerId'],axis=1)
Y = df_train['Survived']
minor = ['Sex_Code','Pclass','FareBin_Code_5','Connected_Survival','Ti_Minor']
minor_Model = RandomForestClassifier(random_state=2,n_estimators=250,min_samples_split=20,oob_score=True)
minor_Model.fit(X[minor], Y)
print('minor oob score :%.5f' %(minor_Model.oob_score_))

X.info()

#產生csv上傳
# Base_Model基準模型:RandomForestClassifier，只對性別，階級編碼  
X_Submit = df_test.drop(labels=['PassengerId'],axis=1)
Base_pred = Base_Model.predict(X_Submit[Base])
submit = pd.DataFrame({"PassengerId": df_test['PassengerId'],"Survived":Base_pred.astype(int)})
submit.to_csv("submit_Base.csv",index=False)
#fare切區間之後  
X_Submit = df_test.drop(labels=['PassengerId'],axis=1)
b5_pred = b5_Model.predict(X_Submit[b5])
submit = pd.DataFrame({"PassengerId": df_test['PassengerId'], "Survived":b5_pred.astype(int)})
submit.to_csv("submit_b5.csv",index=False)
#增加family之後
X_Submit = df_test.drop(labels=['PassengerId'],axis=1)
connect_pred = connect_Model.predict(X_Submit[connect])
submit = pd.DataFrame({"PassengerId": df_test['PassengerId'],"Survived":connect_pred.astype(int)})
submit.to_csv("submit_connect.csv",index=False)
#處理age之後
X_Submit = df_test.drop(labels=['PassengerId'],axis=1)
minor_pred = minor_Model.predict(X_Submit[minor])
submit = pd.DataFrame({"PassengerId": df_test['PassengerId'],"Survived":minor_pred.astype(int)})
submit.to_csv("submit_minor.csv",index=False)