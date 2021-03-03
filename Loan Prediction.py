#!/usr/bin/env python
# coding: utf-8

# In[1]:


pwd


# In[2]:


#changing pwd
import os
os.chdir('D:\python')


# In[3]:


pwd


# In[4]:


#importing dataset(train dataset)
import pandas as pd 
df=pd.read_csv("train_ctrUa4K.csv")


# In[5]:


#check information of datset
df.info()


# In[6]:


df.head()


# In[7]:


#df.count()


# In[8]:


#to check the null values
df.isnull().sum()


# In[9]:


#to check the value count of the column
df['Self_Employed'].value_counts()


# In[10]:


#replacinf the null values with na
df['Self_Employed'].fillna('Not Specified',inplace=True)


# In[11]:


#df.head()


# In[12]:


#df.describe()


# In[13]:


df['Dependents'].value_counts(dropna=False)


# In[14]:


df['Dependents'].replace('3+',3,inplace=True)
df['Dependents'].fillna(0,inplace=True)


# In[15]:


df['Dependents'] = df['Dependents'].astype(int) #to chnge int


# In[16]:


df['LoanAmount'].fillna(df['LoanAmount'].mean(),inplace=True)
df['Loan_Amount_Term'].fillna(360,inplace=True)


# In[17]:


df['Credit_History'].fillna('Not Specified', inplace=True)


# In[18]:


df['Gender'].fillna('Male',inplace=True)
df['Married'].fillna('Yes', inplace=True)


# In[19]:


#df['Married'].value_counts(dropna=False)


# In[20]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[21]:


#to ignore warning
import warnings
warnings.filterwarnings('ignore')


# In[22]:


plt.figure(figsize=(16,6))
sns.boxplot(data=df)


# In[23]:


Q1=df.quantile(0.25)
print(Q1)


# In[24]:


Q3=df.quantile(0.75)
print(Q3)


# In[25]:


IQR= Q3-Q1
print(IQR)


# In[ ]:





# In[26]:


#max range whiskr outlyers
Q3+1.5*IQR


# In[27]:


#min range whiskr outlyers
Q1-1.5*IQR


# In[28]:


#how to check data pointers to delete(optional)
df[df['ApplicantIncome']> 10171]


# In[29]:


df[df['CoapplicantIncome']> 5743]


# In[30]:


df.describe()


# In[31]:


df.loc[df['ApplicantIncome'] > 10171, 'ApplicantIncome'] = 5403


# In[32]:


df.loc[df['CoapplicantIncome'] > 5743, 'CoapplicantIncome'] = 1621


# In[33]:


plt.figure(figsize=(16,6))
sns.boxplot(data=df)


# In[34]:


df['ApplicantIncome'].max()


# In[35]:


df.corr()


# In[36]:


#converting into numerical features
df['Loan_Status'].replace('Y',1,inplace = True)
df['Loan_Status'].replace('N',0,inplace = True)


# In[37]:


df['Education'].value_counts()


# In[38]:


sns.heatmap(df.corr(), annot=True)


# In[39]:


#label encoding
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df['Education_en'] = le.fit_transform(df['Education'])
df['Gender_en'] = le.fit_transform(df['Gender'])
df['Married_en'] = le.fit_transform(df['Married'])
df['Self_Employed_en'] = le.fit_transform(df['Self_Employed'])


# In[40]:


df['Education'].value_counts()


# In[41]:


df.head()


# In[42]:


#pd.get_dummies(df['Gender'])


# In[43]:


# One hot encoding
df = pd.get_dummies(df, columns = ['Gender'], prefix = 'G')
df = pd.get_dummies(df, columns = ['Credit_History'], prefix = 'CH')
df = pd.get_dummies(df, columns = ['Property_Area'], prefix = 'PA')


# In[44]:


df.head()


# In[45]:


sns.countplot(df['Loan_Status'])


# In[46]:


from imblearn.over_sampling import SMOTE

oversample = SMOTE()
X = df[['ApplicantIncome', 'CoapplicantIncome','LoanAmount','Loan_Amount_Term','Education_en','Married_en', 
         'Self_Employed_en', 'G_Female', 'G_Male', 'CH_0.0',
         'CH_1.0', 'CH_Not Specified', 'PA_Rural', 'PA_Semiurban', 'PA_Urban']]

y = df[['Loan_Status']]

X_os, y_os = oversample.fit_resample(X,y)

# X = df[x_var]
# y = df['Loan_Status']


# In[47]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X_os, y_os, test_size=0.3, random_state=1)


# In[48]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform (X_test)


# In[49]:


# from sklearn.preprocessing import normalize

# X_train_nm = normalize(X_train)
# X_test_nm = normalize(X_test)


# In[50]:


#X_test_sc


# In[51]:


#X_test_sc=pd.DataFrame(X_train_nm,columns=[X_os.columns])


# In[52]:


#X_test_sc


# In[53]:


# #from sklearn.tree import DecisionTreeClassifier
# from sklearn import tree
# clf = tree.DecisionTreeClassifier(max_depth=4,random_state=1)
# clf = clf.fit(X_train_sc, y_train)
# y_train_accuracy = clf.predict(X_test_sc)


# In[54]:


#len(X_test), len(X_test_sc)


# In[55]:


# from sklearn.metrics import classification_report
# print(classification_report(y_test,y_train_accuracy))


# In[56]:


# from sklearn.ensemble import RandomForestClassifier

# # rfr = RandomForestRegressor(n_estimators = 10 ,n_jobs = 5, verbose = 3)

# # rfr.fit(X_train, y_train)
# # rfr_pred = rfr.predict(X_test)


# In[57]:


#rfr = RandomForestClassifier(random_state=0)


# In[58]:


#rfr = rfr.fit(X_train_sc, y_train)


# In[59]:


#rfr_acc= rfr.predict(X_test_sc)


# In[60]:


#print(classification_report(y_test, rfr_acc))


# In[61]:


from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier(random_state=0)


# In[62]:


gbc=gbc.fit(X_train_sc, y_train)


# In[63]:


gbc_acc= gbc.predict(X_test_sc)


# In[64]:


print(classification_report(y_test, gbc_acc))


# In[65]:


#gbc.feature_importances_


# In[66]:


# #pd.DataFrame(gbc.feature_importances)

# importance = gbc.feature_importances_

# #create a feature list from the original dataset(list of colums)
# feature_list = list(X_train.columns)

# #create a list of tuples
# feature_importance = sorted(zip(importance,feature_list), reverse=True)

# #create two lists from the previous list of tuples
# imp_f= pd.DataFrame(feature_importance,columns=['importance','feature'])
# importance = list(imp_f['importance'])
# feature= list(imp_f['feature'])


# In[67]:


#print(imp_f)


# In[68]:


# X_new= df[['CH_1.0' ,'ApplicantIncome' ,'CH_Not Specified' , 'LoanAmount' ,'CoapplicantIncome','PA_Semiurban','G_Male','PA_Urban','Loan_Amount_Term'
#           ,'PA_Rural','G_Female']]
# y_new = df['Loan_Status']


# In[69]:


#X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.3, random_state=1)


# In[70]:


# sc = StandardScaler()
# X_train_sc = sc.fit_transform(X_train)
# X_test_sc = sc.transform (X_test)


# In[71]:


# from sklearn.ensemble import GradientBoostingClassifier
# gbc=GradientBoostingClassifier(random_state=0)
# gbc=gbc.fit(X_train_sc, y_train)
# gbc_acc= gbc.predict(X_test_sc)


# In[72]:


#print(classification_report(y_test, gbc_acc))


# In[73]:


#confusion matrics

#from sklearn.metrics import plot_confusion_matrix


# In[74]:


#plot_confusion_matrix(y_test, gbc_acc)
#plot_confusion_matrix(gbc,X_test_sc,y_test)


# In[75]:


#hyper parameter tuning on our XGBoost classifier

# from sklearn.model_selection import GridSearchCV

# clf=DecisionTreeClassifier(random_state = 1)
# param_gr={'max_depth':[2,3,4,5,6,7,8,9,10],'criterion':['entropy','gini'],'min_samples_split':[2,3,4,5]}
# grid_search=GridSearchCV(clf,param_gr,cv=5,scoring='recall',refit= True, n_jobs=4,return_train_score=True)

# grid_search=grid_search.fit(X_train,y_train)
# pred=grid_search.predict(X_test)


# In[76]:


#print(classification_report(y_test,pred))


# In[77]:


# #grid_search.best_params_
# #from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import GridSearchCV

# # param_gr={'max_depth':[2,3,4,5,6,7,8,9,10],'criterion':['entropy','gini'],'min_samples_split':[2,3,4,5]}
# # grid_search=GridSearchCV(clf,param_gr,cv=5,scoring='recall',refit= True, n_jobs=4,return_train_score=True)
# from sklearn.model_selection import GridSearchCV
# new_gr = {'max_depth':[2,3,4,5,6,7,8,9,10],'loss':['deviance', 'exponential'], 'learning_rate':[0.1,0.2,0.3,0.4,0.5], 'n_estimators':range(60,101,10),
#          'max_features':['auto', 'sqrt', 'log2']}
# grid_search=GridSearchCV(gbc,new_gr,cv=5,scoring='accuracy', n_jobs=4,return_train_score=True)


# In[78]:


# grid_search =grid_search.fit (X_train_sc, y_train)
# pred=grid_search.predict(X_test_sc)


# In[79]:


#print(classification_report(y_test,pred))


# In[80]:


#grid_search.best_params_


# In[101]:


gb=GradientBoostingClassifier(loss = 'exponential',max_depth = 2, max_features= 'auto',n_estimators= 80)
gb=gb.fit(X_train_sc, y_train)
gb_acc=gb.predict(X_test_sc)


# In[102]:


print(classification_report(y_test,gb_acc))


# In[114]:


Loan_Status=gb.predict(X)


# In[ ]:


# from sklearn.svm import SVC
# #Support Vector Classification(SVC)
# #Support Vector Machine

# svm_s=SVC(random_state=0, kernel='rbf')
# svm_s=svm_s.fit(X_train_sc, y_train)
# svm_s_acc= svm_s.predict(X_test_sc)


# In[ ]:


# print(classification_report(y_test,svm_s_acc))


# In[ ]:


# from sklearn.linear_model import LogisticRegression
# LOR = LogisticRegression(random_state=0).fit(X_train,y_train)
# LOR_acc=LOR.predict(X_test_sc)


# In[ ]:


#print(classification_report(y_test, LOR_acc))


# In[83]:


df1= pd.read_csv('test_lAUu6dG.csv')


# In[84]:


df1.head()


# In[85]:


df1.isnull().sum()


# In[86]:


df1['Gender'].fillna('Male',inplace=True)
df1['Married'].fillna('Yes', inplace=True)


# In[87]:


df1['Dependents'].replace('3+',3,inplace=True)
df1['Dependents'].fillna(0,inplace=True)


# In[88]:


df1['Self_Employed'].fillna('Not Specified',inplace=True)


# In[89]:


df1['LoanAmount'].fillna(df1['LoanAmount'].mean(),inplace=True)
df1['Loan_Amount_Term'].fillna(360,inplace=True)


# In[90]:


df1['Credit_History'].fillna('Not Specified', inplace=True)


# In[91]:


le = preprocessing.LabelEncoder()

df1['Education_en'] = le.fit_transform(df1['Education'])
df1['Gender_en'] = le.fit_transform(df1['Gender'])
df1['Married_en'] = le.fit_transform(df1['Married'])
df1['Self_Employed_en'] = le.fit_transform(df1['Self_Employed'])


# In[92]:


# One hot encoding
df1= pd.get_dummies(df1, columns = ['Gender'], prefix = 'G')
df1= pd.get_dummies(df1, columns = ['Credit_History'], prefix = 'CH')
df1= pd.get_dummies(df1, columns = ['Property_Area'], prefix = 'PA')


# In[93]:


df1.head()


# In[94]:


df1.columns


# In[95]:


X=df1[['Married_en','LoanAmount', 'Education_en', 'Self_Employed_en','Dependents','ApplicantIncome','CoapplicantIncome', 'G_Female', 'G_Male', 'CH_0.0', 'CH_1.0',
'CH_Not Specified', 'PA_Rural', 'PA_Semiurban', 'PA_Urban']]


# In[96]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train_sc=sc.fit_transform(X_train)
X_test1_sc=sc.transform(X)


# In[97]:


X_test1_sc.shape


# In[112]:


# gb1=GradientBoostingClassifier(random_state=0)
# #(learning_rate = 0.1,loss = 'exponential',max_depth = 5, max_features= 'sqrt',n_estimators= 80)
# gb1.fit(X_train_sc, y_train)
# Loan_Status=gb1.predict(X_test1_sc)


# svm_s=SVC(random_state=0, kernel='rbf')
# svm_s=svm_s.fit(X_train_sc, y_train)
# Loan_Status= svm_s.predict(X_test1_sc)


# In[ ]:


print(Loan_Status)


# In[ ]:


df1.head()


# In[115]:


Loan_Status=pd.DataFrame(Loan_Status,columns=['Loan_Status'])


# In[116]:


Loan_Status.head()


# In[117]:


Loan_Status=pd.concat([df1['Loan_ID'],Loan_Status],axis=1)


# In[118]:


Loan_Status['Loan_Status'].replace(1,'Y',inplace = True)
Loan_Status['Loan_Status'].replace(0,'N',inplace = True)


# In[119]:


Loan_Status.head()


# In[120]:


Loan_Status.to_csv('Submission1.csv',index=None)


# In[121]:


Loan_Status['Loan_Status'].value_counts()


# In[ ]:




