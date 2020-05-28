#FMCG TEAM_3
#IMPORTING THE LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, norm
from scipy import stats

#LOAD THE DATASET
train_data = pd.read_csv("C:\\Users\\hp\\Desktop\\project\\train1.csv")
test_data = pd.read_csv("C:\\Users\\hp\\Desktop\\project\\test1.csv")


train_data.head(10) 
train_data.shape
train_data.drop(["Unnamed: 0"], axis = 1, inplace = True)
print(train_data.keys())
train_data.dtypes #datatype of colunms in train data
train_data.info
 
#CHECKING THE NULL VALUES
train_data.isnull()      #NO NULL VALUES
train_data.isnull().sum()

#CHECKING DUPLICATES AND DRPPPING THEM
train_data.drop_duplicates(keep='first',inplace=True)   #NO DUPLICATES
train_data.columns

test_data.head(10)
test_data.shape
test_data.drop(["ID"], axis = 1, inplace = True)
test_data.info
print(test_data.keys())
test_data.dtypes #datatype of all columns

#ENCODING  (using this we have remove string part and kept integer in following colunms)
train_data['PROD_CD'] = train_data['PROD_CD'].str.replace(r'\D', '').astype(int)
train_data['SLSMAN_CD'] = train_data['SLSMAN_CD'].str.replace(r'\D', '').astype(int)
train_data['TARGET_IN_EA'] = train_data['TARGET_IN_EA'].str.replace(r'\D', '').astype(int)
train_data['ACH_IN_EA'] = train_data['ACH_IN_EA'].str.replace(r'\D', '').astype(int)

test_data['PROD_CD'] = test_data['PROD_CD'].str.replace(r'\D', '').astype(int)
test_data['SLSMAN_CD'] = test_data['SLSMAN_CD'].str.replace(r'\D', '').astype(int)
test_data['TARGET_IN_EA'] = test_data['TARGET_IN_EA'].str.replace(r'\D', '').astype(int)

types_train = train_data.dtypes  # all are integers
types_test = test_data.dtypes #datatype of all columns

#CHECK FOR CORRELATION
sns.pairplot((train_data),hue='ACH_IN_EA')
#sns.paiplot((train_data),hue='TARGET_IN_EA')

corr=train_data.corr()
corr
sns.heatmap(corr,annot=True)

#by heatmap we can see that target and achivement have good correlation that is 0.719321
#and month and year do not have good correlation that is -0.98

"""

                    ID   PROD_CD    ...      TARGET_IN_EA  ACH_IN_EA
ID            1.000000  0.004423    ...         -0.020581  -0.008335
PROD_CD       0.004423  1.000000    ...         -0.038098  -0.018343
SLSMAN_CD     0.999510 -0.000161    ...         -0.020497  -0.008000
PLAN_MONTH    0.009756  0.051039    ...         -0.062976   0.154697
PLAN_YEAR    -0.008052 -0.041916    ...          0.059697  -0.161209
TARGET_IN_EA -0.020581 -0.038098    ...          1.000000   0.719321
ACH_IN_EA    -0.008335 -0.018343    ...          0.719321   1.000000

"""

#def norm_func(i):
    #x=(i-i.mean())/(i.std())
   # return(x)

#GROUPBY IS USED FOR GRUOPING OF PARTICULAR PARAMETERS 
train_data.groupby(['PROD_CD','PLAN_MONTH'])['PROD_CD'].count()
train_data.groupby(['SLSMAN_CD','PLAN_MONTH'])['SLSMAN_CD'].count()
train_data.groupby(['SLSMAN_CD','PLAN_MONTH','PLAN_YEAR'])['SLSMAN_CD'].count()

#CENTER TENDENCY
np.mean(train_data.PROD_CD)  
np.mean(train_data.SLSMAN_CD)  
np.mean(train_data.PLAN_MONTH)  
np.mean(train_data.PLAN_YEAR)  
np.mean(train_data.TARGET_IN_EA)  
np.mean(train_data.ACH_IN_EA)  

#STANDARD DEVIATION
np.std(train_data) 
#VARIANCE
np.var(train_data) 
# SKWENESS
skew(train_data) 
# KURTOSIS
kurtosis(train_data) 
   
#HISTOGRAM
plt.hist(train_data['PROD_CD']);plt.title('Histogram of PROD_CD'); plt.xlabel('PROD_CD'); plt.ylabel('Frequency')
plt.hist(train_data['SLSMAN_CD'], color = 'coral');plt.title('Histogram of SLSMAN_CD'); plt.xlabel('SLSMAN_CD'); plt.ylabel('Frequency')
plt.hist(train_data['PLAN_MONTH'], color= 'orange');plt.title('Histogram of PLAN_MONTH'); plt.xlabel('PLAN_MONTH'); plt.ylabel('Frequency')
plt.hist(train_data['TARGET_IN_EA'], color= 'brown');plt.title('Histogram of TARGET_IN_EA'); plt.xlabel('TARGET_IN_EA'); plt.ylabel('Frequency')
plt.hist(train_data['ACH_IN_EA'], color = 'violet');plt.title('Histogram of ACH_IN_EA'); plt.xlabel('ACH_IN_EA'); plt.ylabel('Frequency')

#BARPLOT
sns.barplot(x="TARGET_IN_EA", y="ACH_IN_EA", hue="PROD_CD", data=train_data)
plt.ylabel("ACH_IN_EA")
plt.title("ACHIVEMENT BASED ON TARGET")

#BOXPLOT
sns.boxplot(train_data["PROD_CD"])
sns.boxplot(train_data["SLSMAN_CD"])
sns.boxplot(train_data["PLAN_MONTH"])
sns.boxplot(train_data["PLAN_YEAR"])
sns.boxplot(train_data["TARGET_IN_EA"])
sns.boxplot(train_data["ACH_IN_EA"])

#SCATTERPLOT
sns.scatterplot(x='PROD_CD', y='ACH_IN_EA', data=train_data).set_title('Scatterplot of ACH_IN_EA & PROD_CD')
sns.scatterplot(x='SLSMAN_CD', y='ACH_IN_EA', data=train_data).set_title('Scatterplot of ACH_IN_EA & SLSMAN_CD')
sns.scatterplot(x='PLAN_MONTH', y='ACH_IN_EA', data=train_data).set_title('Scatterplot of ACH_IN_EA & PLAN_MONTH')
sns.scatterplot(x='PLAN_YEAR', y='ACH_IN_EA', data=train_data).set_title('Scatterplot of ACH_IN_EA & PLAN_YEAR')
sns.scatterplot(x='TARGET_IN_EA', y='ACH_IN_EA', data=train_data).set_title('Scatterplot of ACH_IN_EA & TARGET_IN_EA')

#COUNTPLOT
sns.countplot(train_data["PROD_CD"])
sns.countplot(train_data["SLSMAN_CD"])
sns.countplot(train_data["PLAN_MONTH"])
sns.countplot(train_data["PLAN_YEAR"])
sns.countplot(train_data["TARGET_IN_EA"])

#UNIQUE VALUES
train_data.PROD_CD.unique()               
train_data.PROD_CD.value_counts()                    
train_data.SLSMAN_CD.unique()
train_data.SLSMAN_CD.value_counts()
train_data.PLAN_YEAR.unique()
train_data.PLAN_YEAR.value_counts()
train_data.PLAN_MONTH.unique()
train_data.PLAN_MONTH.value_counts()
train_data.TARGET_IN_EA.unique()
train_data.TARGET_IN_EA.value_counts()
train_data.ACH_IN_EA.unique()
train_data.ACH_IN_EA.value_counts()

df_fmcg = pd.DataFrame(train_data)

fig= df_fmcg.groupby(['PROD_CD','PLAN_MONTH'])['TARGET_IN_EA'].sum().unstack().plot(figsize =(14,8),linewidth = 2)
#In above plot we can see the relation between product_CD, Target and Month.
fig= df_fmcg.groupby(['SLSMAN_CD','PLAN_MONTH'])['TARGET_IN_EA'].sum().unstack().plot(figsize =(14,8),linewidth = 2)
#In above plot we can see the relation between SLSMAN_CD, Target and Month.

#pd.crosstab(train_data.PROD_CD,train_data.SLSMAN_CD).plot(kind="bar")
#pd.crosstab(train_data.PROD_CD,train_data.PLAN_MONTH).plot(kind="bar")
pd.crosstab(train_data.PROD_CD,train_data.PLAN_YEAR).plot(kind="bar")
#In above crosstab we can se values of Product_CD with respect to years.

#distribution plot
#By distribution plot we can see how much our data is normally distributed
sns.distplot(train_data['PROD_CD'], fit=norm, kde=False)
sns.distplot(train_data['SLSMAN_CD'], fit=norm, kde=False, color = 'coral')
sns.distplot(train_data['PLAN_MONTH'], fit=norm, kde=False, color = 'skyblue')
sns.distplot(train_data['PLAN_YEAR'], fit=norm, kde=False, color = 'orange')
sns.distplot(train_data['TARGET_IN_EA'], fit=norm, kde=False, color = 'brown')
sns.distplot(train_data['ACH_IN_EA'], fit=norm, kde=False, color = 'violet')

#arrays 
prod = np.array(train_data['PROD_CD'])
salesman = np.array(train_data['SLSMAN_CD'])
month = np.array(train_data['PLAN_MONTH'])
year = np.array(train_data['PLAN_YEAR'])
target = np.array(train_data['TARGET_IN_EA'])
achieved = np.array(train_data['ACH_IN_EA'])

# Normal Probability distribution 
#As we know data is not normally Distribution so we process and form the Normal Probability Distribution of data column wise.

# ACHIEVED
x_ach = np.linspace(np.min(achieved), np.max(achieved))
y_ach = stats.norm.pdf(x_ach, np.mean(x_ach), np.std(x_ach))
plt.plot(x_ach, y_ach,); plt.xlim(np.min(x_ach), np.max(x_ach));plt.xlabel('achieved');plt.ylabel('Probability');plt.title('Normal Probability Distribution of achieved')

# Product_code
x_prod = np.linspace(np.min(prod), np.max(prod))
y_prod = stats.norm.pdf(x_prod, np.mean(x_prod), np.std(x_prod))
plt.plot(x_prod, y_prod, color = 'coral'); plt.xlim(np.min(x_prod), np.max(x_prod));plt.xlabel('prod_cd');plt.ylabel('Probability');plt.title('Normal Probability Distribution of prod_cd')

# train_dataman_code
x_sale = np.linspace(np.min(salesman), np.max(salesman))
y_sale = stats.norm.pdf(x_sale, np.mean(x_sale), np.std(x_sale))
plt.plot(x_sale, y_sale, color = 'coral'); plt.xlim(np.min(x_sale), np.max(x_prod));plt.xlabel('Sale_cd');plt.ylabel('Probability');plt.title('Normal Probability Distribution of sales_cd')

# target
x_target = np.linspace(np.min(target), np.max(target))
y_target = stats.norm.pdf(x_target, np.mean(x_target), np.std(x_target))
plt.plot(x_target, y_target, color = 'coral'); plt.xlim(np.min(x_target), np.max(x_target));plt.xlabel('target');plt.ylabel('Probability');plt.title('Normal Probability Distribution of target')

# Unsquish the pie.
train_data['PLAN_MONTH'].value_counts().head(10).plot.pie()
train_data['PLAN_YEAR'].value_counts().head(10).plot.pie()
plt.gca().set_aspect('equal')

#By all the plots and graphs we have come to conclusion that highest achivement is 232,000 
#by saleman code i.e SLSMAN_CD 94 in month of november 2019 product_CD 31 

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X = train_data.iloc[:,:6]  #independent columns
y = train_data.iloc[:,-1]    #target column i.e price range

#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=5)
fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['imp','importance']  #naming the dataframe columns

featureScores
"""
            imp    importance
0       PROD_CD  7.657698e+04
1     SLSMAN_CD  2.116982e+05
2    PLAN_MONTH  2.996838e+04
3     PLAN_YEAR  1.313697e+00
4  TARGET_IN_EA  4.555223e+08
5     ACH_IN_EA  6.650893e+08

"""
#By using feature selection we get the importance of particular column in data.
#so by feature selection we get that product_CD and Achivement is having more importance respectively followed by target
#column with very less importance can be drop.




##
data19 = pd.DataFrame(train_data.loc[(train_data.PLAN_YEAR==2019)|(train_data.PLAN_MONTH==10)])

datadrop2=pd.DataFrame(data19.loc[(data19.ACH_IN_EA ==0)&(data19.TARGET_IN_EA ==0)])

data20 = pd.DataFrame(train_data.loc[(train_data.PLAN_YEAR==2020)|(train_data.PLAN_MONTH==1)])



'''from sklearn import preprocessing
le = preprocessing.LabelEncoder()
data19["PLAN_MONTH"] = le.fit_transform(data19["PLAN_MONTH"])



data19['PROD_CD'] = data19['PROD_CD'].str.replace(r'\D', '').astype(int)
data19['SLSMAN_CD'] = data19['SLSMAN_CD'].str.replace(r'\D', '').astype(int)
data19['TARGET_IN_EA'] =data19['TARGET_IN_EA'].str.replace(r'\D', '').astype(int)
data19['ACH_IN_EA'] = data19['ACH_IN_EA'].str.replace(r'\D', '').astype(int)'''

data19.drop(["PLAN_YEAR"], inplace = True, axis = 1)

id=data19[(data19['ACH_IN_EA']==0) & (data19['TARGET_IN_EA']==0)].index
data19.drop(id,inplace=True)


from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict

colnames=list(data19.columns)
print(colnames)

predictor=colnames[0:4]
predictor
target=colnames[-1]
target
train,test =train_test_split(data19,test_size=0.3,random_state=0)

train_x=train[predictor]
train_y=train[target]
test_x=test[predictor]
test_y=test[target]###till here

# =============================================================================
#    RandomForestRegressor

# =============================================================================
from sklearn.ensemble import RandomForestRegressor


rf=RandomForestRegressor()

model=rf.fit(train_x, train_y)

y_pred = model.predict(test_x)
y_pred1 = model.predict(train_x)

y_train=pd.DataFrame(train_y)
y_test=pd.DataFrame(test_y)

y_test['y_pred']=y_pred
error1 = y_test['ACH_IN_EA'] - y_test['y_pred']
y_test['error']=error1
corr_matrix2 = y_test.corr()
err = y_test['error']
#sns.distplot(y_test['error'])
#sns.pairplot(y_test['error'])


from sklearn.metrics import r2_score
rf_test_R2= r2_score(test_y, y_pred)
rf_train_R2= r2_score(train_y, y_pred1)
 ###  r2 = 85.44%
 
 
from sklearn import ensemble
from sklearn.multioutput import MultiOutputRegressor

rf_multioutput = MultiOutputRegressor(ensemble.RandomForestRegressor())
rf_multioutput.fit(train_x, y_train)
rf_train_RMSE=np.mean((rf_multioutput.predict(train_x) - y_train)**2, axis=0)
rf_train_RMSE = np.sqrt(rf_train_RMSE)

rf_multioutput.fit(test_x, y_test)
rf_test_RMSE=np.mean((rf_multioutput.predict(test_x) - y_test)**2, axis=0)
rf_test_RMSE= np.sqrt(rf_test_RMSE)



from sklearn.metrics import r2_score
rf_train_pred = rf_multioutput.predict(train_x)
rf_test_pred = rf_multioutput.predict(test_x)
rf_train_R223 = r2_score(y_train, rf_train_pred)#0.98035
rf_test_R224 = r2_score(y_test, rf_test_pred)#0.97590


