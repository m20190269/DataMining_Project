# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 12:24:32 2019

@author: laura
"""


import pandas as pd
import seaborn as sb
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import statistics as st

data=pd.read_csv('C:\\Users\\laura\\OneDrive\\Documentos\\MAA\\Data_mining\\project\\A2Z_Insurance.csv')

#look to column values
data.columns.values

#we don't nedd the Customer Id column:
data.drop(columns=['Customer Identity'], inplace=True)

#confirm data types
data.dtypes

#we have 3 categorical variables bu only one is setted as object
#data[['Geographic Living Area', 'Has Children (Y=1)']] =data[['Geographic Living Area', 'Has Children (Y=1)']].astype(str)
#nao podemos ja trocar os data types pq se nao os NaN deasaprecem!

#looking for each numeric's column distribution
dist = data.describe()
#we get two strange values here:

#a first policy's year of 53784 and a birthday in 1028 but the lowest values in firts policy's is 1974, lets detect their indexes:
data[data['First Policy´s Year']==53784].index
data[data['Brithday Year']==1028].index

#lets drop those two and look for dist again:  ??? droping or erase the value and fill it later?
data.drop(index=7195, inplace=True)
data.drop(index=9294, inplace=True)

data.reset_index(drop=True, inplace=True)

dist = data.describe()
#now everithing looks  consistent in dist 

#we do detected some cases of first policys year before the birth year, lets count these episodes in total:
conta=0
for i in range(data.shape[0]):
    if data.iloc[i,0]-data.iloc[i,1]<0:
        conta+=1    
conta
# it do happends 1997 times in our data base

#now lets take in account our missing data, whats the percentage of missing data in our data base?
(data.isnull().values.sum()/(data.shape[0]*data.shape[1]))*100

#whats the max number of empty cells in one row?    
max(data.isnull().sum(axis=1))
#looks that there's no nedd to dop the column as 4/13*100 ~ 30% of the row, so not significant and we have no need to drop rows

#lets count these the number of Nan for each column:
data.isna().sum() 

#lets also take in account with columns have in fact true zeros, lets count them per column:
numerical_columns=['First Policy´s Year', 'Brithday Year',
                   'Gross Monthly Salary',
                   'Customer Monetary Value', 'Claims Rate', 'Premiums in LOB: Motor',
                   'Premiums in LOB: Household', 'Premiums in LOB: Health',
                   'Premiums in LOB:  Life', 'Premiums in LOB: Work Compensations']
count_zeros={}
for x in numerical_columns:
    zeros=len(data.loc[:,x][data.loc[:,x]==0])
    count_zeros.update({x : zeros})
numbr_zeros=pd.DataFrame.from_dict(count_zeros, orient='index')

#here we can see that the only premium with true zeros is the only one that don't has Nan values, therefore we will assume that
#the other's premiuns Nan's are in matter in fact zeros that weren't filled. For the other variables we will define other rules

#lets see correlations among the data to help us decide how will we fill the other data:
corr = data.corr()

#lets produce a heatmat, a simplest visualization of correlation between variables
plt.figure(figsize=(16, 6))
sb.heatmap(corr, annot=True)

#treating premiuns:
df_temp=data[['Premiums in LOB: Motor','Premiums in LOB: Household', 'Premiums in LOB: Health',
              'Premiums in LOB:  Life', 'Premiums in LOB: Work Compensations']]

df_temp=df_temp.fillna(0)

data[['Premiums in LOB: Motor','Premiums in LOB: Household', 'Premiums in LOB: Health',
       'Premiums in LOB:  Life', 'Premiums in LOB: Work Compensations']] = df_temp

#lets see correlations among the data to help us decide how will we fill the other data:
corr = data.corr()

#lets produce a heatmat, a simplest visualization of correlation between variables
plt.figure(figsize=(16, 6))
sb.heatmap(corr, annot=True)

#we can see some correlations as birthday and salary and in CMV and claims rate

#next we will say the things that we considered for filling the missing data:

#first some simple imputations:
temp_data=data.loc[:,['First Policy´s Year', 'Educational Degree', 'Geographic Living Area']]

from sklearn.impute import SimpleImputer

#fillup the 1st policy year with column median, looking for his distribution is very 
#centered (median~mean) (close to a normal distribution) and the interval amplitude's and standart deviation
#are also low (more then in age), so most of the poupulation is concentrated near the observations's median/mean
#therefore we will fill up those missing values with the colum median
#age has the same distributions's behavior as 1st policy's year but a bigger std but filling those missing
#column's values with the median could lead us to errors like 1st policy before birth and
#we don't want that, so we will come up with other strategy for age

# First Policy's Year Column 
imp_median = SimpleImputer(missing_values=np.nan, strategy='median')

temp_data.loc[:,'First Policy´s Year'] = imp_median.fit_transform(temp_data[['First Policy´s Year']]).ravel()

#looking for geographic living are and educational degree seemed logical for us, that those
#columns could be filled up with de mode (mos common value)
imp_mf = SimpleImputer(missing_values=np.nan, strategy='most_frequent') 

temp_data.loc[:,'Geographic Living Area'] = imp_mf.fit_transform(temp_data[['Geographic Living Area']]).ravel()
temp_data.loc[:,'Educational Degree'] = imp_mf.fit_transform(temp_data[['Educational Degree']]).ravel()

data.loc[:,['First Policy´s Year', 'Educational Degree', 'Geographic Living Area']]=temp_data

#about the age column we wanted to take in account that most of population made the 1st policy in same period. As
#in age we also see that behavior that could, in some way, lead us to the assumption that most of the population
#makes the firts policy after some certain time after birth, lets see these difference distribution (1st policy-age):
diff = data.iloc[:,0]-data.iloc[:,1]
diff=diff.dropna()
dist1=diff.describe()
#we can see that diff is centered as (median~mean~18), we can also see that we have a big std (18)... but the 
#we will consider that the aproximation to the normal distribution is enought for this imputation:

temp_data1 = data.loc[:,['First Policy´s Year', 'Brithday Year']]

#so we will separete our trainning set in 2 the complete and incomplete data 
data_incomplete = temp_data1[temp_data1.iloc[:,1].isna()]
data_complete = temp_data1[~temp_data1.index.isin(data_incomplete.index)]

#calculating median of difference from complete dataata
diff0 = data_complete.iloc[:,0]-data_complete.iloc[:,1]
median_diff = st.median(diff0)

data_incomplete.iloc[:,1]=data_incomplete.iloc[:,0]-median_diff

data_join=pd.concat([data_complete, data_incomplete], 
                    axis=0, 
                    ignore_index=False, 
                    verify_integrity=False, sort=False).sort_index()

data.loc[:,['First Policy´s Year', 'Brithday Year']]=data_join

data.isna().sum()

#we decided that a good method would be to regress Gross Monthly Salary in function of Educational Degree  and Brithday Year
#as GMS shows a hight correlation with age and we considered that educational degree would produce good resultes with age

#first transform Educational Degree to dummy variable

my_data_to_regress=pd.DataFrame(data.loc[:,['Brithday Year', 'Educational Degree', 'Gross Monthly Salary']])

from sklearn.preprocessing import OneHotEncoder
#define encoder
educ_ohe = OneHotEncoder()
#apply encoder
#for each different numeric label we will put as a dummy variable
Education = educ_ohe.fit_transform(my_data_to_regress.loc[:,'Educational Degree'].values.reshape(-1,1)).toarray()

#change header
my_data_to_label_OneHot_s = pd.DataFrame(Education, columns = ['educ_'+str(int(i)) for i in range(1,Education.shape[1]+1)])

new_data_regress = pd.concat([my_data_to_regress, my_data_to_label_OneHot_s], axis=1).drop(columns='Educational Degree')

#regress the data
from sklearn.neighbors import KNeighborsRegressor

#separate in complete and incomplete set to train the model
my_data_to_reg_incomplete = new_data_regress[new_data_regress.loc[:,'Gross Monthly Salary'].isna()]
my_data_to_reg_complete = new_data_regress[~new_data_regress.index.isin(my_data_to_reg_incomplete.index)]

my_regressor = KNeighborsRegressor(10, 
                                   weights ='distance', 
                                   metric = 'euclidean')

#trainning with complete set
neigh = my_regressor.fit(my_data_to_reg_complete.loc[:,['Brithday Year','educ_1', 'educ_2', 'educ_3', 'educ_4']],
                         my_data_to_reg_complete.loc[:,['Gross Monthly Salary']])

imputed_GMS = neigh.predict(my_data_to_reg_incomplete.drop(columns = ['Gross Monthly Salary']))

temp_df = pd.DataFrame(imputed_GMS.reshape(-1,1),index=list(my_data_to_reg_incomplete.index), columns = ['Gross Monthly Salary'])
my_data_to_reg_incomplete = my_data_to_reg_incomplete.drop(columns=['Gross Monthly Salary'])

my_data_to_reg_incomplete = pd.concat([my_data_to_reg_incomplete, temp_df], 
                                      axis=1, 
                                      ignore_index=False, 
                                      verify_integrity=False)

data_join1=pd.concat([my_data_to_reg_complete, my_data_to_reg_incomplete], 
                    axis=0, 
                    ignore_index=False, 
                    verify_integrity=False, sort=False).sort_index()

data.loc[:,'Gross Monthly Salary']=data_join1.loc[:,'Gross Monthly Salary']


#at last we need to predict Has Children (Y=1), we will use Brithday Year and Premiums in LOB:  Life
from sklearn.neighbors import KNeighborsClassifier

my_data = data.loc[:,['Brithday Year','Has Children (Y=1)', 'Premiums in LOB:  Life']]

#we will separate the Has Children by incomplete info and complete info, to train the model
my_data_incomplete = my_data.loc[my_data.loc[:,'Has Children (Y=1)'].isna()]
my_data_complete = my_data[~my_data.index.isin(my_data_incomplete.index)]

my_data_complete.loc[:,'Has Children (Y=1)']= my_data_complete.loc[:,'Has Children (Y=1)'].astype('category')

#definning our classification model
clf = KNeighborsClassifier(3, 
                           weights='distance',
                           metric = 'euclidean')

#trainning our model
trained_model = clf.fit(my_data_complete.loc[:,['Brithday Year','Premiums in LOB:  Life']], 
                        my_data_complete.loc[:,'Has Children (Y=1)'])

#predicting the missing value
imputed_child = trained_model.predict(my_data_incomplete.drop(columns = ['Has Children (Y=1)']))

#adding missing values to original data
temp_dff = pd.DataFrame(imputed_child.reshape(-1,1),index=list(my_data_incomplete.index), columns = ['Has Children (Y=1)'])
my_data_incomplete = my_data_incomplete.drop(columns=['Has Children (Y=1)'])

my_data_incomplete = pd.concat([my_data_incomplete, temp_dff], 
                              axis=1, 
                              ignore_index=False, 
                              verify_integrity=False)

data_join2=pd.concat([my_data_complete, my_data_incomplete], 
                    axis=0, 
                    ignore_index=False, 
                    verify_integrity=False, sort=False).sort_index()

data.loc[:,'Has Children (Y=1)']=data_join2.loc[:,'Has Children (Y=1)']

data.isna().sum()
dist_f=data.describe()
data.dtypes

#################################################################################################
mean_diff = st.median(diff)

a = np.zeros(shape=(data.shape[0],1))


for i in range(data.shape[0]):
    if np.isnan(data.iloc[i,2]):
        a[i] = data.iloc[i,1] - mean_diff
    else:
        a[i] = data.iloc[i,2]
data.dtypes
data[['Brithday Year']]=a

data.isna().sum()
 

