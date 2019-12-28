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

#nao podemos ja trocar os data types pq se nao os NaN deasaprecem!

#looking for each numeric's column distribution
dist = data.describe()
#we get two strange values here:

#a first policy's year of 53784 and a birthday in 1028 but the lowest values in firts policy's is 1974, lets detect their indexes:
data[data['First Policy´s Year']==53784].index
data[data['Brithday Year']==1028].index

#erase the value and fill it later
data.loc[data[data['First Policy´s Year']==53784].index[0], 'First Policy´s Year']=np.nan
data.loc[data[data['Brithday Year']==1028].index[0], 'Brithday Year']=np.nan

dist = data.describe()
#now everithing looks  consistent in dist 

#we do detected some cases of first policys year before the birth year, lets count these episodes in total:
sum(data.iloc[:,0]<data.iloc[:,1]) #1997 cases

#we will assume that the birthday year is wrong, dsetting as nan the values in birthday year
data.iloc[list(data[data.iloc[:,0]<data.iloc[:,1]].index),1]=np.nan
# it do happends 1997 times in our data base

#now lets take in account our missing data, whats the percentage of missing data in our data base?
(data.isna().values.sum()/(data.shape[0]*data.shape[1]))*100

#whats the max number of empty cells in one row?    
max(data.isna().sum(axis=1))
#looks that there's no nedd to dop the column as 5/13*100 ~ 38% of the row, we condider that we have no need to drop rows

#lets count these the number of Nan for each column:
data.isna().sum() 

#we have 20% of our birthday column with missing data, taking also in account the dependency of birthday Year and 
#First Policy's Year column this could lead us to some big predictions errors envolving the impact on variable's distribution
#and lost of sence among variables.
#In that way we will consider that First Policy's Year is enought for our analysis, taking account in the future the 
#policy time that makes a lot of sence for our customer's analysis'.

data.drop(columns=['Brithday Year'], inplace=True)

#lets also take in account with columns have in fact true zeros, lets count them per column:
numerical_columns=['First Policy´s Year','Gross Monthly Salary',
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
corr1 = data.corr()

#lets produce a heatmat, a simplest visualization of correlation between variables
plt.figure(figsize=(16, 6))
sb.heatmap(corr1, annot=True)

#we can see some correlations as birthday and salary and in CMV and claims rate

#next we will say the things that we considered for filling the missing data:

#first some simple imputations:
temp_data=data.loc[:,['First Policy´s Year', 'Educational Degree', 'Geographic Living Area']]

from sklearn.impute import SimpleImputer
#for First Policy as there are just some missing values, we will fill up these missing values with column median
#in that way there's not much impact in population's distribution
imp_median = SimpleImputer(missing_values=np.nan, strategy='median')

temp_data.loc[:,'First Policy´s Year'] = imp_median.fit_transform(temp_data[['First Policy´s Year']]).ravel()

#looking for geographic living are and educational degree seemed logical for us, that those
#columns could be filled up with de mode (most common value)
imp_mf = SimpleImputer(missing_values=np.nan, strategy='most_frequent') 

temp_data.loc[:,'Geographic Living Area'] = imp_mf.fit_transform(temp_data[['Geographic Living Area']]).ravel()
temp_data.loc[:,'Educational Degree'] = imp_mf.fit_transform(temp_data[['Educational Degree']]).ravel()

data.loc[:,['First Policy´s Year', 'Educational Degree', 'Geographic Living Area']]=temp_data.copy()


data.isna().sum()

#we decided that a good method would be to regress Gross Monthly Salary in function of Educational Degree and 'Premiums in LOB:  Life'
#as GMS shows some correlation with 'Premiums in LOB:  Life' and we considered that educational degree would produce good resultes with it

#first transform Educational Degree to dummy variable

my_data_to_regress=pd.DataFrame(data.loc[:,['Premiums in LOB:  Life', 'Educational Degree', 'Gross Monthly Salary']])

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
neigh = my_regressor.fit(my_data_to_reg_complete.loc[:,['Premiums in LOB:  Life','educ_1', 'educ_2', 'educ_3', 'educ_4']],
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

data.loc[:,'Gross Monthly Salary']=data_join1.loc[:,'Gross Monthly Salary'].copy()



#at last we need to predict Has Children (Y=1), we will use 'Premiums in LOB: Motor' and Gross Monthly Salary because of correlation
from sklearn.neighbors import KNeighborsClassifier

data.groupby(by=['Has Children (Y=1)'])['Gross Monthly Salary'].mean()

my_data = data.loc[:,['Premiums in LOB: Motor', 'Has Children (Y=1)', 'Gross Monthly Salary']]

#we will separate the Has Children by incomplete info and complete info, to train the model
my_data_incomplete = my_data.loc[my_data.loc[:,'Has Children (Y=1)'].isna()]
my_data_complete = my_data[~my_data.index.isin(my_data_incomplete.index)]

my_data_complete.loc[:,'Has Children (Y=1)']= my_data_complete.loc[:,'Has Children (Y=1)'].astype('category').copy()

#definning our classification model
clf = KNeighborsClassifier(3, 
                           weights='distance',
                           metric = 'euclidean')

#trainning our model
trained_model = clf.fit(my_data_complete.loc[:,['Premiums in LOB: Motor','Gross Monthly Salary']], 
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

data.loc[:,'Has Children (Y=1)']=data_join2.loc[:,'Has Children (Y=1)'].copy()

data.isna().sum()
dist_f=data.describe()
data.dtypes


corr = data.corr()
plt.figure(figsize=(16, 6))
sb.heatmap(corr, annot=True)





###############              Tranform the Variables                   #################


data.rename(columns={'First Policy´s Year': 'Fidelity Time',
                     'Gross Monthly Salary':'Percentage of Salary'}, inplace=True)
#about the First Policy´s Year we thougth that we may reharange them in some way that they
#bring more value to us. So instead of first policy year we thought that would be more interest to know
#the years hat they are our costumer 

fidelity_time = 2016-data.iloc[:,0]

data.loc[:,'Fidelity Time']=fidelity_time

# instead of salary we createt 'Salary Proportion', 
#to get the information of how much of the costumer's salary is actually spended in premiuns.
premiuns_corrected=data.iloc[:,7:]
premiuns_corrected[premiuns_corrected<0]=0 #replacing negatives by 0

premiuns_total=premiuns_corrected.sum(axis=1)
anual_salary=data.iloc[:,2]*12
salary_prop=(premiuns_total/anual_salary)
salary_prop.idxmax()
data.loc[:,'Percentage of Salary']= (salary_prop*100)

#Considering the further clustering, makes more sence grouping people with the same consuming habbits
#and not thw ones who spend the same ammount in something, in that way we thought about considerind 
#premiuns proportions instead of premiuns ammounts:

premiuns=data.iloc[:,7:]
premiuns.replace(0,np.nan, inplace=True) #to not get 0/0 errors

props_premiuns=premiuns.divide(premiuns_total, axis=0)

props_premiuns.replace(np.nan,0, inplace=True)
 
data.iloc[:,7:]=props_premiuns*100

corr2 = data.corr()
plt.figure(figsize=(16, 6))
sb.heatmap(corr2, annot=True)
dist_f=data.describe()

#although we see high correlations inter-Premiuns, we do cnsider that all variables give 
#much value for our analysis 

#we can see a very higth (negative) correlation between CMV and Claims rate and they do are related:
# - CMV englobes all years 'profit' along with other variables, more higher better
# - claims rate are also like 'profit' beeing 'costs'/'sales' for last 2 years, more lower better
#that correlations indicates that in most of the cases these two variables when one increases the other decreases
#so they are pretty much alignt. In that way, as they are correlated and related and CMV also englobes profit from 
#last 2years, we will only consider CMV for our clusters






##################          Removing Outliers                   ###############

sb.set_style("ticks")

sb.pairplot(data[['Premiums in LOB: Motor','Premiums in LOB: Household', 'Premiums in LOB: Health',
                  'Premiums in LOB:  Life', 'Premiums in LOB: Work Compensations']],
            diag_kind='hist',
            kind='scatter',
            palette='hus1')
#looking for the plots:
# - householding, we can identify a outlier with low household (-50)
# - work compensasion a highervalues almost 80%, annother two highs and like 2 too lows 

sb.pairplot(data[['Fidelity Time', 'Percentage of Salary', 'Customer Monetary Value']],
            diag_kind='hist',
            kind='scatter',
            palette='hus1')
#looking for the plots:
# - percentage of salary we can verify a really higth value and a 2nd not so hight 
#but also kind away of the population
# - CMV also has some pretty low values
outlier_indexes=[]

sb.distplot(data[['Premiums in LOB: Household']], kde=False)
sb.boxplot(data[['Premiums in LOB: Household']], orient='h', whis=1.5) #1 outlier to remove (bad costumer)

outlier_indexes.append(data[data.iloc[:,8]<(-20)].index[0]) 

sb.distplot(data[['Premiums in LOB: Work Compensations']], kde=False)
sb.boxplot(data[['Premiums in LOB: Work Compensations']], orient='h', whis=1.5) 

for i in range(len(data[data.iloc[:,11]>55].index)):
    outlier_indexes.append(data[data.iloc[:,11]>55].index[i]) 

for i in range(len(data[data.iloc[:,11]<(-5)].index)):
    outlier_indexes.append(data[data.iloc[:,11]<(-5)].index[i])

sb.distplot(data[['Percentage of Salary']], kde=False)
sb.boxplot(data[['Percentage of Salary']], orient='h', whis=20) #lot of extreme values but 2 outliers

for i in range(len(data[data.iloc[:,2]>60].index)):
    outlier_indexes.append(data[data.iloc[:,2]>60].index[i])


sb.distplot(data[['Customer Monetary Value']], kde=False)
sb.boxplot(data[['Customer Monetary Value']], orient='h', whis=7) 
#one such extreme that we cant read correlctly the boxplot, remove the ones that we can see and create new viz

for i in range(len(data[data.iloc[:,5]<(-25000)].index)):
    outlier_indexes.append(data[data.iloc[:,5]<(-25000)].index[i]) 

outliers=pd.DataFrame(data.iloc[outlier_indexes,:])
#for now, remove the outliers from our data
data.drop(index=outlier_indexes, inplace=True)


#lets take a better view in CMV
sb.distplot(data[['Customer Monetary Value']], kde=False)
sb.boxplot(data[['Customer Monetary Value']], orient='h', whis=7)

outlier_indexes=[]

for i in range(len(data[data.iloc[:,5]<(-5000)].index)):
    outlier_indexes.append(data[data.iloc[:,5]<(-5000)].index[i])

for i in range(len(data[data.iloc[:,5]>(4000)].index)):
    outlier_indexes.append(data[data.iloc[:,5]>(4000)].index[i])

outliers=pd.concat([outliers, pd.DataFrame(data.iloc[outlier_indexes,:])], axis=0)
outliers.reset_index(drop=True, inplace=True)
#for now, remove the outliers from our data 
data.drop(index=outlier_indexes, inplace=True)
data.reset_index(drop=True, inplace=True)


#do differnt clusters and see what produces better results


############           Prepare for Clusters                    ##############      
 
data_use=data[['Fidelity Time', 'Educational Degree', 'Percentage of Salary',
               'Geographic Living Area', 'Has Children (Y=1)', 'Customer Monetary Value',
               'Premiums in LOB: Motor','Premiums in LOB: Household', 'Premiums in LOB: Health',
               'Premiums in LOB:  Life', 'Premiums in LOB: Work Compensations']].copy()

      
#let's separate the variables into Consumtion and Client:
 #outliers???      
data_client=data[['Fidelity Time', 'Educational Degree', 'Percentage of Salary',
                  'Geographic Living Area', 'Has Children (Y=1)', 'Customer Monetary Value']]
data_consum=data[['Premiums in LOB: Motor','Premiums in LOB: Household', 'Premiums in LOB: Health',
                  'Premiums in LOB:  Life', 'Premiums in LOB: Work Compensations']].copy()

#in client we have a mix of numeric values with categoricals, lets separate it for the clustets

catdata_client= data_client[['Educational Degree', 'Geographic Living Area',
                            'Has Children (Y=1)']].copy().astype('str')

numdata_client= data_client[['Fidelity Time', 'Percentage of Salary', 'Customer Monetary Value']].copy()

#lets start with consumption, first we need to normalize the data


############           Cluster Analysis                   ############## 


##########################################################################################################################
############################################### KMEANS + HC Client ########################################################


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
client_norm =scaler.fit_transform(numdata_client)
client_norm =pd.DataFrame(client_norm, columns=numdata_client.columns)


#for Cluster's construction we will use both k-means followed by hyerichal clustering

#first, run k-means for data reduction:

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=200, 
                random_state=0,
                n_init = 5,
                max_iter=600).fit(client_norm)  ##fitting / trainning in CA_Norm data

client_clusters = kmeans.cluster_centers_

client1_labels=pd.DataFrame(kmeans.labels_, columns=['kmeansLabels'])

scaler.inverse_transform(X= client_clusters) #see how they look without normalization

client_rednorm = pd.DataFrame(client_clusters, columns= client_norm.columns) #mantain the data normalized 

import scipy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster import hierarchy
from pylab import rcParams

# The final result will use the sklearn

import sklearn
from sklearn.cluster import AgglomerativeClustering
import sklearn.metrics as sm

plt.figure(figsize=(10,5))
plt.style.use('seaborn-whitegrid')

#computing dendogram to decide number of clusters

Z = linkage(client_rednorm,
            method ='ward' )
dendrogram(Z,
            truncate_mode='lastp',
            p=40,
            orientation = 'top',
            leaf_rotation=45.,
            leaf_font_size=10.,
            show_contracted=True,
            show_leaf_counts=True, color_threshold=50, above_threshold_color='k')

plt.title('Truncated Hierarchical Clustering Dendrogram')
plt.xlabel('Cluster Size')
plt.ylabel('Distance')
plt.show()

#looking for the dendogram we can see that the cut that maximizes the intra-clusters distances ir k=2
k=2
#defining our Hierichal Clustering model
Hclustering = AgglomerativeClustering(n_clusters=k,
                                      affinity='euclidean',
                                      linkage='ward')

# fitting the model to Ca-norm
client_HC = Hclustering.fit(client_rednorm) 

#labels say witch centroid belong the index
numclientHC_labels = pd.DataFrame(client_HC.labels_) 
numclientHC_labels.columns =  ['Labels']


clust200_labels= pd.DataFrame(pd.concat([pd.DataFrame(client_rednorm),
                                  numclientHC_labels], axis=1),
                        columns=['Fidelity Time', 'Percentage of Salary', 'Customer Monetary Value', 'Labels'])

clust200_labels.reset_index(inplace=True)


#we know that the centroids will be the mean of the points! as they are the central point of each group:
to_revert=clust200_labels.groupby(['Labels'])['Fidelity Time', 'Percentage of Salary', 'Customer Monetary Value'].mean()
# the data is stiil in the normalized format so we will Do the necessary transformations

clientHC_result = pd.DataFrame(scaler.inverse_transform(X=to_revert), columns=['Fidelity Time', 'Percentage of Salary', 'Customer Monetary Value',])

#getting observations labels using kmeans labels and HC labels

client_finalables=[]
for i in client1_labels.kmeansLabels:
    for j in clust200_labels.index:
        if i==j:
            client_finalables.append(clust200_labels.iloc[j,4])
            
#see clusters distribution:
numclient_labels=pd.DataFrame(client_finalables, columns=['KmeansHC'])

#joining to our data each label, fom kmodes and kmeans+HC algorithms
client_kmHC = pd.DataFrame(pd.concat([client_norm, numclient_labels], axis = 1),
                      columns=['Fidelity Time', 'Percentage of Salary', 'Customer Monetary Value', 'KmeansHC'])
            
kmHC_client_table=pd.pivot_table(client_kmHC, values='Fidelity Time', index='KmeansHC', aggfunc='count')

kmHC_client_table.columns=['Count']
##########################################################################################################################
############################################### KMODES Client #############################################################

#Applying k-modes to categorical data
import numpy as np
from kmodes.kmodes import KModes

#elbow graph to decide number of clusters
lista=[]
for k in range(1,6):
    km=KModes(n_clusters=k, init='random', n_init=30, verbose=1)
    km.fit_predict(catdata_client)
    lista.append(km.cost_)
    
sb.lineplot( x=range(1,6) ,y=lista)

#the biggest change in variance of the cost is seen in k=3
k=3

km = KModes(n_clusters=k, init='random', n_init=50, verbose=1) #after running some times and observing cat_counts, that was the best k

clusters = km.fit_predict(catdata_client)

# Print the cluster centroids
print(km.cluster_centroids_)
cat_centroids = pd.DataFrame(km.cluster_centroids_,
                             columns = ['Educational Degree', 'Geographic Living Area', 'Has Children (Y=1)'])

catclient_labels=pd.DataFrame(km.labels_)

unique, counts = np.unique(km.labels_, return_counts=True)

cat_counts = pd.DataFrame(np.asarray((unique, counts)).T, columns = ['Label','Count'])

cat_centroids = pd.concat([cat_centroids, cat_counts], axis = 1)

catclient_labels.columns=['Kmodes']

#cluster 1 and 2 from kmodes have half the members of first group. lets see is k=2 produces better results:

k=2

km2 = KModes(n_clusters=k, init='random', n_init=50, verbose=1) #after running some times and observing cat_counts, that was the best k

clusters2 = km2.fit_predict(catdata_client)

# Print the cluster centroids
print(km2.cluster_centroids_)
cat_centroids2 = pd.DataFrame(km2.cluster_centroids_,
                             columns = ['Educational Degree', 'Geographic Living Area', 'Has Children (Y=1)'])

catclient_labels2=pd.DataFrame(km2.labels_, columns=['Kmodes'])

unique2, counts2 = np.unique(km2.labels_, return_counts=True)

cat_counts2 = pd.DataFrame(np.asarray((unique2, counts2)).T, columns = ['Label','Count'])

cat_centroids2 = pd.concat([cat_centroids2, cat_counts2], axis = 1)


#we obtain in the end 2 clusters: clients living in zone 1 with obly High school education and 
#clients living in zone 4 with bsc/Msc education 

###################################################################################################################
###################################################################################################################

######### consumption
#normalize data
cons_norm =scaler.fit_transform(data_consum)
cons_norm =pd.DataFrame(cons_norm, columns=data_consum.columns)

#for Cluster's construction we will use both k-means followed by hyerichal clustering

#first, run k-means for data reduction:

kmeans = KMeans(n_clusters=200, 
                random_state=0,
                n_init = 5,
                max_iter=600).fit(cons_norm)  ##fitting / trainning in CA_Norm data

my_clusters = kmeans.cluster_centers_

consum1_labels=pd.DataFrame(kmeans.labels_, columns=['kmeansLabels'])

scaler.inverse_transform(X= my_clusters) #see how they look without normalization

cons_rednorm = pd.DataFrame(my_clusters, columns= cons_norm.columns) #mantain the data normalized 


plt.figure(figsize=(10,5))
plt.style.use('seaborn-whitegrid')

#computing dendogram to decide number of clusters

Z = linkage(cons_rednorm,
            method ='ward' )
dendrogram(Z,
            truncate_mode='lastp',
            p=40,
            orientation = 'top',
            leaf_rotation=45.,
            leaf_font_size=10.,
            show_contracted=True,
            show_leaf_counts=True, color_threshold=50, above_threshold_color='k')

plt.title('Truncated Hierarchical Clustering Dendrogram')
plt.xlabel('Cluster Size')
plt.ylabel('Distance')
plt.show()

#looking for the dendogram we can see that the cut that maximizes the intra-clusters distances ir k=5
k=5
#defining our Hierichal Clustering model
Hclustering = AgglomerativeClustering(n_clusters=k,
                                      affinity='euclidean',
                                      linkage='ward')

# fitting the model to Ca-norm
consum_HC = Hclustering.fit(cons_rednorm) 

#labels say witch centroid belong the index
consumHC_labels = pd.DataFrame(consum_HC.labels_, columns=['Labels']) 
#consum_labels.columns =  ['Labels']


consum200_labels= pd.DataFrame(pd.concat([pd.DataFrame(cons_rednorm),
                                  consumHC_labels], axis=1),
                        columns=['Premiums in LOB: Motor', 'Premiums in LOB: Household','Premiums in LOB: Health', 
                                 'Premiums in LOB:  Life', 'Premiums in LOB: Work Compensations', 'Labels'])

consum200_labels.reset_index(inplace=True)
    
#we know that the centroids will be the mean of the points! as they are the central point of each group:
consumHC_result_to_revert=consum200_labels.groupby(['Labels'])['Premiums in LOB: Motor', 'Premiums in LOB: Household','Premiums in LOB: Health', 
                          'Premiums in LOB:  Life', 'Premiums in LOB: Work Compensations'].mean()
# the data is stiil in the normalized format so we will Do the necessary transformations

consumHC_result = pd.DataFrame(scaler.inverse_transform(X=consumHC_result_to_revert), columns=['Premiums in LOB: Motor', 'Premiums in LOB: Household',
                            'Premiums in LOB: Health','Premiums in LOB:  Life', 'Premiums in LOB: Work Compensations',])

consum_finalables=[]
for i in consum1_labels.kmeansLabels:
    for j in consum200_labels.index:
        if i==j:
            consum_finalables.append(consum200_labels.iloc[j,6])



consum_labels=pd.DataFrame(consum_finalables, columns=['KmeansHC'])


#joining to our data each label, fom kmeans+HC algorithms
consum = pd.DataFrame(pd.concat([cons_norm, consum_labels], axis = 1),
                      columns=['Premiums in LOB: Motor', 'Premiums in LOB: Household',
                            'Premiums in LOB: Health','Premiums in LOB:  Life', 
                            'Premiums in LOB: Work Compensations', 'KmeansHC'])

#cointing how many clients are in each cluster
consum_table=pd.pivot_table(consum, values='Premiums in LOB: Motor', index='KmeansHC', aggfunc='count')
consum_table.columns=['Count']
#Here we get 2 pretty low clusters, (1)+(4), evean is this 2 join togetter by choosing k=4, wouldnt be enought to be next to
#the other clusters, anyway lets try k=4:

k=4
#defining our Hierichal Clustering model
Hclustering = AgglomerativeClustering(n_clusters=k,
                                      affinity='euclidean',
                                      linkage='ward')

# fitting the model to Ca-norm
consum_HC1 = Hclustering.fit(cons_rednorm) 

#labels say witch centroid belong the index
consumHC_labels1 = pd.DataFrame(consum_HC1.labels_, columns=['Labels']) 
#consum_labels.columns =  ['Labels']


consum200_labels1= pd.DataFrame(pd.concat([pd.DataFrame(cons_rednorm),
                                  consumHC_labels1], axis=1),
                        columns=['Premiums in LOB: Motor', 'Premiums in LOB: Household','Premiums in LOB: Health', 
                                 'Premiums in LOB:  Life', 'Premiums in LOB: Work Compensations', 'Labels'])

consum200_labels1.reset_index(inplace=True)
    
#we know that the centroids will be the mean of the points! as they are the central point of each group:
consumHC_result_to_revert1=consum200_labels1.groupby(['Labels'])['Premiums in LOB: Motor', 'Premiums in LOB: Household','Premiums in LOB: Health', 
                          'Premiums in LOB:  Life', 'Premiums in LOB: Work Compensations'].mean()
# the data is stiil in the normalized format so we will Do the necessary transformations

consumHC_result1 = pd.DataFrame(scaler.inverse_transform(X=consumHC_result_to_revert1), columns=['Premiums in LOB: Motor', 'Premiums in LOB: Household',
                            'Premiums in LOB: Health','Premiums in LOB:  Life', 'Premiums in LOB: Work Compensations',])

consum_finalables1=[]
for i in consum1_labels.kmeansLabels:
    for j in consum200_labels1.index:
        if i==j:
            consum_finalables1.append(consum200_labels1.iloc[j,6])



consum_labels1=pd.DataFrame(consum_finalables1, columns=['KmeansHC'])


#joining to our data each label, fom kmeans+HC algorithms
consum1 = pd.DataFrame(pd.concat([cons_norm, consum_labels1], axis = 1),
                      columns=['Premiums in LOB: Motor', 'Premiums in LOB: Household',
                            'Premiums in LOB: Health','Premiums in LOB:  Life', 
                            'Premiums in LOB: Work Compensations', 'KmeansHC'])

#cointing how many clients are in each cluster
consum_table1=pd.pivot_table(consum1, values='Premiums in LOB: Motor', index='KmeansHC', aggfunc='count')
consum_table1.columns=['Count']

#looking for the consum table we can see that the firts cluster (0) has a pretty low number of individuals
#however looking for our clusters (consumHC_results) the cluster 0 and 3 have almost the same consuming
#tendencies, both of them have the highest consumptuion in Household, and both consum the same proportion
#of their money in health (17%). Doing this join the cluster 0 will reprsent less than 15% of the population
#beiing not that significant

#determinate new observations labels
new_ConsumLabels=[]
for i in range(len(consum_labels1.KmeansHC)):
    if (consum_labels1.KmeansHC[i]==0 or consum_labels1.KmeansHC[i]==3) :
        new_ConsumLabels.append(2)
    else:
        new_ConsumLabels.append((consum_labels1.KmeansHC[i]-1))

#new clusters labels
new_consum200=consum200_labels1.copy()
for i in range(len(new_consum200.Labels)):
    if (new_consum200.iloc[i,6]==0 or new_consum200.iloc[i,6]==3):
        new_consum200.iloc[i,6]=2
    elif new_consum200.iloc[i,6]==1:
        new_consum200.iloc[i,6]=0
    elif new_consum200.iloc[i,6]==2:
        new_consum200.iloc[i,6]=1
    
        

consum_labels_final2=pd.DataFrame(new_ConsumLabels, columns=['KmeansHC'])

#joining to our data each label, fom kmeans+HC algorithms
consum2 = pd.DataFrame(pd.concat([cons_norm, consum_labels_final2], axis = 1),
                      columns=['Premiums in LOB: Motor', 'Premiums in LOB: Household',
                            'Premiums in LOB: Health','Premiums in LOB:  Life', 
                            'Premiums in LOB: Work Compensations', 'KmeansHC'])

#cointing how many clients are in each cluster
consum_table2=pd.pivot_table(consum2, values='Premiums in LOB: Motor', index='KmeansHC', aggfunc='count')
consum_table2.columns=['Count']
#lets determinate our new clusters (still normalized):
consum_torevert2=consum2.groupby(['KmeansHC'])['Premiums in LOB: Motor', 'Premiums in LOB: Household','Premiums in LOB: Health', 
                          'Premiums in LOB:  Life', 'Premiums in LOB: Work Compensations'].mean()

#final consumption clusters
consum_clusters2 = pd.DataFrame(scaler.inverse_transform(X=consum_torevert2), columns=['Premiums in LOB: Motor', 'Premiums in LOB: Household',
                            'Premiums in LOB: Health','Premiums in LOB:  Life', 'Premiums in LOB: Work Compensations',])

#######################################################################################################################

#######################3                                 SOM                             ###############################

########################################################################################################################
import urllib3
import joblib
import random

from sompy.sompy import SOMFactory
from sompy.visualization.plot_tools import plot_hex_map
import logging

cons_norm =scaler.fit_transform(data_consum)
cons_norm =pd.DataFrame(cons_norm, columns=data_consum.columns)

X = cons_norm.values

names = ['Premiums in LOB: Motor', 'Premiums in LOB: Household',
         'Premiums in LOB: Health', 'Premiums in LOB:  Life',
         'Premiums in LOB: Work Compensations']


som = SOMFactory().build(data = X,
               mapsize=(22,22),
               normalization = 'var',
               initialization='random',
               component_names=names,
               lattice='hexa',
               training = 'seq')

som.train(n_job=4,
         verbose='info',
         train_rough_len=30,
         train_finetune_len=100)


som_clusters = pd.DataFrame(som._data, columns = names)

som_labels = pd.DataFrame(som._bmu[0], columns=['SOM_labels'])
    
som_clusters_ = pd.concat([som_clusters,som_labels], axis = 1)

som_clusters_.columns = ['Premiums in LOB: Motor', 'Premiums in LOB: Household',
                        'Premiums in LOB: Health', 'Premiums in LOB:  Life',
                        'Premiums in LOB: Work Compensations', 'Lables']

som_consum_clusters_torevert=som_clusters_.groupby(['Lables'])['Premiums in LOB: Motor', 'Premiums in LOB: Household',
                        'Premiums in LOB: Health', 'Premiums in LOB:  Life',
                        'Premiums in LOB: Work Compensations'].mean()

from sompy.visualization.mapview import View2DPacked 
view2D  = View2DPacked(10,10,"", text_size=7)
view2D.show(som, col_sz=5, what = 'codebook',)
plt.show()


from sompy.visualization.mapview import View2D
view2D  = View2D(10,10,"", text_size=7)
view2D.show(som, col_sz=5, what = 'codebook',)
plt.show()


#HC clustering - deciding k
plt.figure(figsize=(10,5))
plt.style.use('seaborn-whitegrid')

Z = linkage(som_consum_clusters_torevert,
            method ='ward' )

dendrogram(Z,
            truncate_mode='lastp',
            p=40,
            orientation = 'top',
            leaf_rotation=45.,
            leaf_font_size=10.,
            show_contracted=True,
            show_leaf_counts=True, color_threshold=50, above_threshold_color='k')

plt.title('Truncated Hierarchical Clustering Dendrogram')
plt.xlabel('Cluster Size')
plt.ylabel('Distance')
plt.show()

k=3
Hclustering = AgglomerativeClustering(n_clusters=k,
                                      affinity='euclidean',
                                      linkage='ward')

# fitting the model to Ca-norm
consum_som_HC = Hclustering.fit(som_consum_clusters_torevert) 

#labels say witch centroid belong the index
clust100_consum_labels = pd.DataFrame(consum_som_HC.labels_, columns=['Labels']) 

clust100_consum_labels.reset_index(inplace=True)

consum_finalables=[]

for i in som_labels.SOM_labels:
    for j in clust100_consum_labels.index:
        if i==j:
            consum_finalables.append(clust100_consum_labels.iloc[j,1])


som_cons_clusters = pd.concat([cons_norm,pd.DataFrame(consum_finalables, columns=['Labels'])], axis = 1)

som_consum_torevert= som_cons_clusters.groupby(['Labels'])['Premiums in LOB: Motor', 'Premiums in LOB: Household','Premiums in LOB: Health', 
                          'Premiums in LOB:  Life', 'Premiums in LOB: Work Compensations'].mean()

#final consumption clusters
SOM_consum_clusters = pd.DataFrame(scaler.inverse_transform(X=som_consum_torevert), columns=['Premiums in LOB: Motor', 'Premiums in LOB: Household',
                            'Premiums in LOB: Health','Premiums in LOB:  Life', 'Premiums in LOB: Work Compensations',])

SOM_consum_table=pd.pivot_table(som_cons_clusters, values='Premiums in LOB: Motor', index='Labels', aggfunc='count')
SOM_consum_table.columns=['Count']

#very balanced

##########################################################################################################################

client_norm =scaler.fit_transform(numdata_client)
client_norm =pd.DataFrame(client_norm, columns=numdata_client.columns)

X = client_norm.values

names = ['Fidelity Time', 'Percentage of Salary', 'Customer Monetary Value']


som1 = SOMFactory().build(data = X,
               mapsize=(22,22),
               normalization = 'var',
               initialization='random',
               component_names=names,
               lattice='hexa',
               training = 'seq')

som1.train(n_job=4,
         verbose='info',
         train_rough_len=30,
         train_finetune_len=100)


som_client_clusters = pd.DataFrame(som1._data, columns = names)

som_client_labels = pd.DataFrame(som1._bmu[0], columns=['SOM_labels'])
    
som_client_clusters = pd.concat([som_client_clusters,som_client_labels], axis = 1)

som_client_clusters_torevert=som_client_clusters.groupby(['SOM_labels'])['Fidelity Time',
                                           'Percentage of Salary', 'Customer Monetary Value'].mean()


som_client_clusters.columns = ['Fidelity Time', 'Percentage of Salary',
                               'Customer Monetary Value', 'Lables']

view2D  = View2DPacked(10,10,"", text_size=7)
view2D.show(som1, col_sz=5, what = 'codebook',)
plt.show()

view2D  = View2D(10,10,"", text_size=7)
view2D.show(som1, col_sz=5, what = 'codebook',)
plt.show()

#HC clustering - deciding k
plt.figure(figsize=(10,5))
plt.style.use('seaborn-whitegrid')

Z = linkage(som_client_clusters_torevert,
            method ='ward' )

dendrogram(Z,
            truncate_mode='lastp',
            p=40,
            orientation = 'top',
            leaf_rotation=45.,
            leaf_font_size=10.,
            show_contracted=True,
            show_leaf_counts=True, color_threshold=50, above_threshold_color='k')

plt.title('Truncated Hierarchical Clustering Dendrogram')
plt.xlabel('Cluster Size')
plt.ylabel('Distance')
plt.show() 

k=3
Hclustering = AgglomerativeClustering(n_clusters=k,
                                      affinity='euclidean',
                                      linkage='ward')

# fitting the model to Ca-norm
client_som_HC = Hclustering.fit(som_client_clusters_torevert) 

#labels say witch centroid belong the index
clust100_client_labels = pd.DataFrame(client_som_HC.labels_, columns=['Labels']) 

clust100_client_labels.reset_index(inplace=True)

client_finalables=[]

for i in som_client_labels.SOM_labels:
    for j in clust100_client_labels.index:
        if i==j:
            client_finalables.append(clust100_client_labels.iloc[j,1])


client_som_clusters = pd.concat([client_norm,pd.DataFrame(client_finalables, columns=['Labels'])], axis = 1)

som_client_torevert=client_som_clusters.groupby(['Labels'])['Fidelity Time',
                                           'Percentage of Salary', 'Customer Monetary Value'].mean()

#final consumption clusters
SOM_client_clusters = pd.DataFrame(scaler.inverse_transform(X=som_client_torevert), 
                                   columns=['Fidelity Time', 'Percentage of Salary', 'Customer Monetary Value',])

SOM_client_table=pd.pivot_table(client_som_clusters, values='Fidelity Time', index='Labels', aggfunc='count')
SOM_client_table.columns=['Count']

#we have a cluster with just a few elements comparing with the otheres, however this group represents our best customers
#said that we will keep them to support our analysis



##############################################################################################################################
######################################################  DBSCAN  ##############################################################
##############################################################################################################################

from sklearn.cluster import DBSCAN
from sklearn import metrics

# CLIENT
client_norm =scaler.fit_transform(numdata_client)
client_norm =pd.DataFrame(client_norm, columns=numdata_client.columns)

#eps is the ratio of the circle to cach elements
db = DBSCAN(eps= 0.4,
            min_samples=10).fit(client_norm)

#for k=0.4 there's a significant reduction of noise, while increasing the next k's the reduction started to not be thet significant


labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0) #counting all different discarding -1 from count

unique_clusters, counts_clusters = np.unique(db.labels_, return_counts = True)
print(np.asarray((unique_clusters, counts_clusters)))


from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(client_norm)
pca_2d = pca.transform(client_norm)
for i in range(0, pca_2d.shape[0]):
    if db.labels_[i] == 0:
        c1 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='r',marker='+')
    elif db.labels_[i] == 1:
        c2 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='g',marker='o')
    elif db.labels_[i] == 2:
        c4 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='k',marker='v')
    elif db.labels_[i] == 3:
        c5 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='y',marker='s')
    elif db.labels_[i] == 4:
        c6 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='m',marker='p')
    elif db.labels_[i] == -1:
        c3 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='b',marker='*')

plt.legend([c1, c2, c3], ['Cluster 1', 'Cluster 2','Noise'])
plt.title('DBSCAN finds 2 clusters and noise')
plt.show()


#Consumtion:
cons_norm =scaler.fit_transform(data_consum)
cons_norm =pd.DataFrame(cons_norm, columns=data_consum.columns)

#eps is the ratio of the circle to cach elements
db = DBSCAN(eps= 0.5,
            min_samples=10).fit(cons_norm)

#for k=0.6 there's a significant reduction of noise, while increasing the next k's the reduction started to not be thet significant


labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0) #counting all different discarding -1 from count

unique_clusters, counts_clusters = np.unique(db.labels_, return_counts = True)
print(np.asarray((unique_clusters, counts_clusters)))


from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(client_norm)
pca_2d = pca.transform(client_norm)
for i in range(0, pca_2d.shape[0]):
    if db.labels_[i] == 0:
        c1 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='r',marker='+')
    elif db.labels_[i] == 1:
        c2 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='g',marker='o')
    elif db.labels_[i] == 2:
        c4 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='k',marker='v')
    elif db.labels_[i] == -1:
        c3 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='b',marker='*')

plt.legend([c1, c2, c3], ['Cluster 1', 'Cluster 2','Noise'])
plt.title('DBSCAN finds 2 clusters and noise')
plt.show()


#again the dbscan finds clusters with different dimentions and that's not the point at all 

############################################################################################################################
#####################################################  MEAN_SHIFT ######################################################

#lets try the mean-shift algoritm:

# CLIENT
client_norm =scaler.fit_transform(numdata_client)
client_norm =pd.DataFrame(client_norm, columns=numdata_client.columns)


from sklearn.cluster import MeanShift, estimate_bandwidth

to_MS = client_norm
# lets detect automatically the bandwidth c
my_bandwidth = estimate_bandwidth(client_norm,
                               quantile=0.2,
                               n_samples=2000)
#this estimates the bandith calculating the distances between all points, calculate the quantils and we choose it 
#n_samples is we selecting a sample to not +usae all the sample because is computacional too hard


ms = MeanShift(bandwidth=my_bandwidth, 
               bin_seeding=True)
#similiar resultes but we dont get the noise and with noise we have a idea of outliers
ms.fit(to_MS)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)


#Values
ms_client_clusters=scaler.inverse_transform(X=cluster_centers)

#Count
unique, counts = np.unique(labels, return_counts=True)

print(np.asarray((unique, counts)).T)

# CONSUMPTION
cons_norm =scaler.fit_transform(data_consum)
cons_norm =pd.DataFrame(cons_norm, columns=data_consum.columns)

to_MS = cons_norm
# The following bandwidth can be automatically detected using
my_bandwidth = estimate_bandwidth(cons_norm,
                               quantile=0.2,
                               n_samples=1000)
#this estimates the bandith calculating the distances between all points, calculate the quantils and we choose it 
#n_samples is we selecting a sample to not +usae all the sample because is computacional too hard


ms = MeanShift(bandwidth=my_bandwidth, 
               bin_seeding=True)
#similiar resultes but we dont get the noise and with noise we have a idea of outliers
#we can use it in dbscan
ms.fit(to_MS)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)


#Values
ms_consum_clusters=scaler.inverse_transform(X=cluster_centers)

#Count
unique, counts = np.unique(labels, return_counts=True)

print(np.asarray((unique, counts)).T)

#again we get a first group with most of the population (the ones that spend most of their money in motor)


##########################################################################################################################
#####################################################  KMEANS  ###########################################################
##########################################################################################################################

from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
#defining a function that automatically plots a sillouette graph

def sil_plot(k, norm_data, labels):
    n_clusters = k
    silhouette_avg = silhouette_score(norm_data, labels)
    
    print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
    
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(norm_data, labels)
    
    
    cluster_labels = labels
    y_lower = 100
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    
    #ax.set_xlim([-1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
    ax.set_ylim([0, norm_data.shape[0] + (n_clusters + 1) * 10])
    
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    
        ith_cluster_silhouette_values.sort()
    
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
    
        color = cm.rainbow(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color,
                          edgecolor=color, 
                          alpha=0.7)
    
        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
        # Compute the new y_lower for next plot
        y_lower = y_upper + 100  # 10 for the 0 samples
        
        ax.set_title("The silhouette plot for the various clusters.")
        ax.set_xlabel("The silhouette coefficient values")
        ax.set_ylabel("Cluster label")
    
        # The vertical line for average silhouette score of all the values
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")
        
    return plt.show()
###############################################################################################################################

## CONSUMPTION    

cons_norm =scaler.fit_transform(data_consum)
cons_norm =pd.DataFrame(cons_norm, columns=data_consum.columns)

#choosing best k
lista=[]
for k in range(1,6):
    kmeans = KMeans(n_clusters=k, 
                random_state=0,
                n_init = 5,
                max_iter=200).fit(cons_norm)
    lista.append(kmeans.inertia_)
    
sb.lineplot( x=range(1,6),y=lista)

k=2

km_cons = KMeans(n_clusters=k, 
                random_state=0,
                n_init = 5,
                max_iter=200).fit(cons_norm)  

cons_clusters = km_cons.cluster_centers_ 

km_cons_clusters = pd.DataFrame(scaler.inverse_transform(X=cons_clusters), columns= cons_norm.columns)

# Check the Clusters

#silluete graph, range between -1 and 1, where if negative maybe they are wrong assingn, if positive (near 1) they are closest to his pears
#it does like a average, does like a average: 1-(sum distance to his pears /sum distance to other nearest clust)

sil_plot(k,cons_norm, km_cons.labels_)

#1 with only positive valus -- is ok!
#other with negative values, here we can consider 2 optoions:
#    - they are 2 subclusters that need to be setted together
#    - we nedd an extra cluster - lets try k=3


#lets see the distribution of the elements 
km_cons_labels=pd.concat([cons_norm, pd.DataFrame(km_cons.labels_, columns=['Labels'])], axis=1)

km_cons_table=pd.pivot_table(km_cons_labels, values='Premiums in LOB: Motor', index='Labels', aggfunc='count')
km_cons_table.columns=['Count']

#with the sillouette graph we conclueded that or we needed to set togetter 2 or an extra cluster
#looking for this table results we see a clusters with a pretty low value, so lets try k=3

k=3

km_cons1 = KMeans(n_clusters=k, 
                random_state=0,
                n_init = 5,
                max_iter=200).fit(cons_norm)  

cons_clusters1 = km_cons1.cluster_centers_ 

km_cons_clusters1 = pd.DataFrame(scaler.inverse_transform(X=cons_clusters1), columns= cons_norm.columns)

# Check the Clusters
sil_plot(k, cons_norm, km_cons1.labels_)

#looks worst than first solution 


## CLIENT
client_norm =scaler.fit_transform(numdata_client)
client_norm =pd.DataFrame(client_norm, columns=numdata_client.columns)

#choosing k
lista=[]
for k in range(1,6):
    kmeans = KMeans(n_clusters=k, 
                random_state=0,
                n_init = 5,
                max_iter=200).fit(client_norm)
    lista.append(kmeans.inertia_)
    
sb.lineplot( x=range(1,6),y=lista)

k=3

km_client = KMeans(n_clusters=k, 
                random_state=0,
                n_init = 5,
                max_iter=200).fit(client_norm)  

client_clusters = km_client.cluster_centers_ 


scaler.inverse_transform(X= client_clusters)

km_client_clusters = pd.DataFrame(scaler.inverse_transform(X=client_clusters), columns= client_norm.columns)

# Check the Clusters
sil_plot(k, client_norm, km_client.labels_)

#2 with only positive valus -- is ok!
#other with negative values, here we can consider 2 optoions:
#    - they are 2 subclusters that need to be setted together
#    - we nedd an extra cluster


#lets see the distribution of the elements 
km_client_labels=pd.concat([client_norm, pd.DataFrame(km_client.labels_, columns=['Labels'])], axis=1)

km_client_table=pd.pivot_table(km_client_labels, values='Fidelity Time', index='Labels', aggfunc='count')
km_client_table.columns=['Count']

#with the sillouette graph we conclueded that or we needed to set togetter 2 or an extra cluster
#looking for this table results we see a clusters with a pretty low value, however this represents 
#our best customers, in that way we will keep it

###############################################################################################################
####################################  HC clustering  #######################################################3

#CLIENT
client_norm =scaler.fit_transform(numdata_client)
client_norm =pd.DataFrame(client_norm, columns=numdata_client.columns)

plt.figure(figsize=(10,5))
plt.style.use('seaborn-whitegrid')
Z = linkage(client_norm,
            method ='ward' )
dendrogram(Z,
            truncate_mode='lastp',
            p=40,
            orientation = 'top',
            leaf_rotation=45.,
            leaf_font_size=10.,
            show_contracted=True,
            show_leaf_counts=True, color_threshold=50, above_threshold_color='k')

plt.title('Truncated Hierarchical Clustering Dendrogram')
plt.xlabel('Cluster Size')
plt.ylabel('Distance')
#plt.axhline(y=50)
plt.show()

k =3


Hclustering = AgglomerativeClustering(n_clusters=k,
                                      affinity='euclidean',
                                      linkage='ward')  

#Replace the test with proper data
client_HC = Hclustering.fit(client_norm) # fitting the model to Ca-norm

client_HC_labels = pd.DataFrame(client_HC.labels_, columns =  ['Labels']) #attributes a centroid to each point



HC_client_labels= pd.DataFrame(pd.concat([pd.DataFrame(client_norm),
                                  client_HC_labels], axis=1),
                        columns=['Fidelity Time', 'Percentage of Salary', 'Customer Monetary Value', 'Labels'])

HC_client_table=pd.pivot_table(HC_client_labels, values='Fidelity Time', index='Labels', aggfunc='count')
HC_client_table.columns=['Count']

HC_client_to_revert=HC_client_labels.groupby(['Labels'])['Fidelity Time', 'Percentage of Salary', 'Customer Monetary Value'].mean()
HC_client_clusters = pd.DataFrame(scaler.inverse_transform(X=HC_client_to_revert), columns=['Fidelity Time', 'Percentage of Salary', 'Customer Monetary Value',])


#not very balanced lets recalculate for k=2 

k =2


Hclustering = AgglomerativeClustering(n_clusters=k,
                                      affinity='euclidean',
                                      linkage='ward')  

#Replace the test with proper data
client_HC1 = Hclustering.fit(client_norm) # fitting the model to Ca-norm

client_HC_labels1 = pd.DataFrame(client_HC1.labels_, columns =  ['Labels']) #attributes a centroid to each point



HC_client_labels1= pd.DataFrame(pd.concat([pd.DataFrame(client_norm),
                                  client_HC_labels1], axis=1),
                        columns=['Fidelity Time', 'Percentage of Salary', 'Customer Monetary Value', 'Labels'])

HC_client_table1=pd.pivot_table(HC_client_labels1, values='Fidelity Time', index='Labels', aggfunc='count')


HC_client_to_revert1=HC_client_labels1.groupby(['Labels'])['Fidelity Time', 'Percentage of Salary', 'Customer Monetary Value'].mean()
HC_client_clusters1 = pd.DataFrame(scaler.inverse_transform(X=HC_client_to_revert1), columns=['Fidelity Time', 'Percentage of Salary', 'Customer Monetary Value',])


## CONSUMPTION

cons_norm =scaler.fit_transform(data_consum)
cons_norm =pd.DataFrame(cons_norm, columns=data_consum.columns)

plt.figure(figsize=(10,5))
plt.style.use('seaborn-whitegrid')
Z = linkage(cons_norm,
            method ='ward' )
dendrogram(Z,
            truncate_mode='lastp',
            p=40,
            orientation = 'top',
            leaf_rotation=45.,
            leaf_font_size=10.,
            show_contracted=True,
            show_leaf_counts=True, color_threshold=50, above_threshold_color='k')

plt.title('Truncated Hierarchical Clustering Dendrogram')
plt.xlabel('Cluster Size')
plt.ylabel('Distance')
#plt.axhline(y=50)
plt.show()

k =3

Hclustering = AgglomerativeClustering(n_clusters=k,
                                      affinity='euclidean',
                                      linkage='ward')  

#Replace the test with proper data
cons_HC = Hclustering.fit(cons_norm) # fitting the model to Ca-norm

cons_HC_labels = pd.DataFrame(cons_HC.labels_, columns =  ['Labels']) #attributes a centroid to each point



HC_cons_labels= pd.DataFrame(pd.concat([pd.DataFrame(cons_norm),
                                  cons_HC_labels], axis=1),
                        columns=['Premiums in LOB: Motor', 'Premiums in LOB: Household',
       'Premiums in LOB: Health', 'Premiums in LOB:  Life',
       'Premiums in LOB: Work Compensations', 'Labels'])

HC_cons_table=pd.pivot_table(HC_cons_labels, values='Premiums in LOB: Motor', index='Labels', aggfunc='count')
HC_cons_table.columns=['Count']

HC_cons_to_revert=HC_cons_labels.groupby(['Labels'])['Premiums in LOB: Motor', 'Premiums in LOB: Household',
                                                     'Premiums in LOB: Health', 'Premiums in LOB:  Life',
                                                     'Premiums in LOB: Work Compensations'].mean()
HC_cons_clusters = pd.DataFrame(scaler.inverse_transform(X=HC_cons_to_revert), 
                                columns=['Premiums in LOB: Motor', 'Premiums in LOB: Household',
                                'Premiums in LOB: Health', 'Premiums in LOB:  Life',
                                'Premiums in LOB: Work Compensations',])

#the 2 first clusters have a pretty low value ompring to the last one 
k =2

Hclustering = AgglomerativeClustering(n_clusters=k,
                                      affinity='euclidean',
                                      linkage='ward')  

#Replace the test with proper data
cons_HC1 = Hclustering.fit(cons_norm) # fitting the model to Ca-norm

cons_HC_labels1 = pd.DataFrame(cons_HC1.labels_, columns =  ['Labels']) #attributes a centroid to each point



HC_cons_labels1= pd.DataFrame(pd.concat([pd.DataFrame(cons_norm),
                                  cons_HC_labels1], axis=1),
                        columns=['Premiums in LOB: Motor', 'Premiums in LOB: Household',
       'Premiums in LOB: Health', 'Premiums in LOB:  Life',
       'Premiums in LOB: Work Compensations', 'Labels'])

HC_cons_table1=pd.pivot_table(HC_cons_labels1, values='Premiums in LOB: Motor', index='Labels', aggfunc='count')
HC_cons_table1.columns=['Count']

HC_cons_to_revert1=HC_cons_labels1.groupby(['Labels'])['Premiums in LOB: Motor', 'Premiums in LOB: Household',
                                                     'Premiums in LOB: Health', 'Premiums in LOB:  Life',
                                                     'Premiums in LOB: Work Compensations'].mean()
HC_cons_clusters1 = pd.DataFrame(scaler.inverse_transform(X=HC_cons_to_revert1), 
                                columns=['Premiums in LOB: Motor', 'Premiums in LOB: Household',
                                'Premiums in LOB: Health', 'Premiums in LOB:  Life',
                                'Premiums in LOB: Work Compensations',])

#worst than the first solution

###############################################################################################################################
###############################################################################################################################
#########################################  CHOOSING THE BEST MODEL ############################################################
###############################################################################################################################
##############################################################################################################################

#function to calculate euclidian distance from each member to their clluster
def total_distance(clusters, data_w_labels):
    ''' labels columns has to be called 'Labels' '''
    distances=0
    for i in range(clusters.shape[0]):
        for j in range(data_w_labels.shape[0]):
            if i==data_w_labels.loc[j,'Labels']:
                for c in range(clusters.shape[1]):
                    distances+=(clusters.iloc[i,c]-data_w_labels.iloc[j,c])**2
    sqrt_distance=np.sqrt(distances)
    return (sqrt_distance)

#now it's time choose from our best models:
    
#CLIENT:
    #- SOM
    #- kmeans 
km_norm_client_clusters=pd.DataFrame(client_clusters)
client_kmHC.columns=['Fidelity Time', 'Percentage of Salary', 'Customer Monetary Value', 'Labels']
kmHC_client_distance=total_distance(to_revert, client_kmHC)  
SOM_client_distance=total_distance(som_client_torevert, client_som_clusters)
kmeans_client_distance=total_distance(km_norm_client_clusters, km_client_labels)  #the best!
HC_client_distance=total_distance(HC_client_to_revert, HC_client_labels)

    
#CONSUMPTION:
    #- kmeans + HC
    #- SOM
    #- kmeans
km_norm_cons_clusters=pd.DataFrame(cons_clusters)
consum2.columns=['Premiums in LOB: Motor', 'Premiums in LOB: Household',
                 'Premiums in LOB: Health', 'Premiums in LOB:  Life',
                 'Premiums in LOB: Work Compensations', 'Labels']

kmHC_consum_distance=total_distance(consum_torevert2, consum2) #the best!
SOM_consum_distance=total_distance(som_consum_torevert, som_cons_clusters)
kmeans_consum_distance=total_distance(km_norm_cons_clusters, km_cons_labels) 
HC_consum_distance=total_distance(HC_cons_to_revert, HC_cons_labels)



#now lets try to join this two models also with k-modes results:

km_client_finalables=pd.DataFrame(km_client.labels_, columns=['N_Client'])
consum_labels_final2.columns=['Consum']
catclient_labels2.columns=['C_Client']

#table joining numerical client with categorical client
all_data = pd.DataFrame(pd.concat([data_use, km_client_finalables, catclient_labels2], axis = 1),
                      columns=['Fidelity Time', 'Educational Degree', 'Percentage of Salary',
                               'Geographic Living Area', 'Has Children (Y=1)', 'Customer Monetary Value',
                               'N_Client', 'C_Client'])

#cointing how many clients are in each cluster
KM_HC_table_final=pd.pivot_table(all_data, values='Fidelity Time', index='C_Client', 
                     columns=['N_Client'], aggfunc='count')

KM_HC_table_final.columns=['n_client_0', 'n_client_1', 'n_client_2']
KM_HC_table_final.index=['c_client_0', 'c_client_1']


#table joining numerical client with categorical client and consuming

all_data = pd.DataFrame(pd.concat([data_use, km_client_finalables, consum_labels_final2, catclient_labels2], axis = 1),
                      columns=['Fidelity Time', 'Educational Degree', 'Percentage of Salary',
                               'Geographic Living Area', 'Has Children (Y=1)', 'Customer Monetary Value',
                               'Premiums in LOB: Motor','Premiums in LOB: Household', 'Premiums in LOB: Health',
                               'Premiums in LOB:  Life', 'Premiums in LOB: Work Compensations', 'N_Client', 'Consum', 'C_Client'])

#cointing how many clients are in each cluster
KM_HC_table_final2=pd.pivot_table(all_data, values='Fidelity Time', index='Consum', 
                     columns=['N_Client', 'C_Client'], aggfunc='count')

KM_HC_table_final2.columns=['client_00', 'client_01', 'client_10', 'client_11', 'client_20', 'client_21']
KM_HC_table_final2.index=['consum_0', 'consum_1', 'consum_2']

#only num client and consuming

all_data = pd.DataFrame(pd.concat([data_use, km_client_finalables, consum_labels_final2], axis = 1),
                      columns=['Fidelity Time', 'Educational Degree', 'Percentage of Salary',
                               'Geographic Living Area', 'Has Children (Y=1)', 'Customer Monetary Value',
                               'Premiums in LOB: Motor','Premiums in LOB: Household', 'Premiums in LOB: Health',
                               'Premiums in LOB:  Life', 'Premiums in LOB: Work Compensations', 'N_Client', 'Consum'])

#cointing how many clients are in each cluster
KM_HC_table_final3=pd.pivot_table(all_data, values='Fidelity Time', index='Consum', 
                     columns=['N_Client'], aggfunc='count')
KM_HC_table_final3.columns=['client_0', 'client_1', 'client_2']
KM_HC_table_final3.index=['consum_0', 'consum_1', 'consum_2']

#two groups have pretty low values (0,0) and (1,0), for all the possible moovings that they can do they represent very low % of the pop
#in that way there's no need to recalculate the clusters 
#good0 (0,0) or goes to clust1 or clust2 from client maintanning his consuming or we alter his consumming to clust 2 mainting his client tendency

#normalized data for the members that will moove:
client_good0=pd.DataFrame()
consum_good0=pd.DataFrame()
client_good1=pd.DataFrame()
consum_good1=pd.DataFrame()
for i in range(all_data.shape[0]):
    if all_data.iloc[i,11]==0 and all_data.iloc[i,12]==0:
        client_good0=client_good0.append(client_norm.iloc[i,:], ignore_index=True)
        consum_good0=consum_good0.append(cons_norm.iloc[i,:], ignore_index=True)
    elif all_data.iloc[i,11]==0 and all_data.iloc[i,12]==1:
        client_good1=client_good0.append(client_norm.iloc[i,:], ignore_index=True)
        consum_good1=consum_good0.append(cons_norm.iloc[i,:], ignore_index=True)


def moving(good, cluster, move):   
    """function that calculates the total euclidian distance of a group 'good' moove to the 'cluster(move)' """
    distances=0
    for j in range(cluster.shape[1]):
        for i in range(good.shape[0]):
            distances+=(cluster.iloc[move,j]-good.iloc[i,j])**2
    sqrt_distance=np.sqrt(distances)
    return (sqrt_distance)

#computing possible mooves to decide the best option for both groups
move00_01=moving(client_good0,km_norm_client_clusters, 1)+moving(consum_good0, consum_torevert2, 0) #the best
move00_02=moving(client_good0,km_norm_client_clusters, 2)+moving(consum_good0, consum_torevert2, 0) 
move00_20=moving(client_good0,km_norm_client_clusters, 0)+moving(consum_good0, consum_torevert2, 2) 

move10_11=moving(client_good1,km_norm_client_clusters, 1)+moving(consum_good1, consum_torevert2, 1) #the best
move10_12=moving(client_good1,km_norm_client_clusters, 2)+moving(consum_good1, consum_torevert2, 1) 
move10_20=moving(client_good1,km_norm_client_clusters, 0)+moving(consum_good1, consum_torevert2, 2)   
#both will moove to consuming(1), client(1):

#changing labels:
for i in range(all_data.shape[0]):
    if all_data.iloc[i,11]==0 and (all_data.iloc[i,12]==0 or all_data.iloc[i,12]==1):
        all_data.iloc[i,11]=1
        km_client_finalables.iloc[i,0]=1
        
KM_HC_table_final4=pd.pivot_table(all_data, values='Fidelity Time', index='Consum', 
                     columns=['N_Client'], aggfunc='count')
KM_HC_table_final4.columns=['client_0', 'client_1', 'client_2']
KM_HC_table_final4.index=['consum_0', 'consum_1', 'consum_2']


#our final clusters:
km_client_clusters
consum_clusters2

#labels:
km_client_finalables
consum_labels_final2

#consum,client = nr cluster
#2,0 = 0
#0,1 = 1
#1,1 = 2
#2,1 = 3
#0,2 = 4
#1,2 = 5
#2,2 = 6

labels=pd.concat([consum_labels_final2, km_client_finalables], axis=1)

new_labels=[]
for i in range(len(labels.Consum)):
    if labels.iloc[i,0]==2 and labels.iloc[i,1]==0:
        new_labels.append(0)
    elif labels.iloc[i,0]==0 and labels.iloc[i,1]==1:
        new_labels.append(1)
    elif labels.iloc[i,0]==1 and labels.iloc[i,1]==1:
        new_labels.append(2)
    elif labels.iloc[i,0]==2 and labels.iloc[i,1]==1:
        new_labels.append(3)
    elif labels.iloc[i,0]==0 and labels.iloc[i,1]==2:
        new_labels.append(4)
    elif labels.iloc[i,0]==1 and labels.iloc[i,1]==2:
        new_labels.append(5)
    elif labels.iloc[i,0]==2 and labels.iloc[i,1]==2:
        new_labels.append(6)
            
#for the decisoin tree we will only use numerical data, first we haven't done clustering with the categorical data, and then decison tree
#classivier doesnt perform well with categorical

num_data=data[['Fidelity Time', 'Percentage of Salary','Customer Monetary Value',
               'Premiums in LOB: Motor','Premiums in LOB: Household', 'Premiums in LOB: Health',
               'Premiums in LOB:  Life', 'Premiums in LOB: Work Compensations']].copy()

Affinity=pd.concat([num_data, pd.DataFrame(new_labels, columns=['Labels'])], axis=1)


from sklearn import preprocessing
from sklearn import metrics
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import graphviz 

x=Affinity[['Fidelity Time', 'Percentage of Salary','Customer Monetary Value',
            'Premiums in LOB: Motor','Premiums in LOB: Household', 'Premiums in LOB: Health',
            'Premiums in LOB:  Life', 'Premiums in LOB: Work Compensations']].copy()
y=Affinity[['Labels']].copy()


x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=3082, #len(y)*0.3
                                                    random_state=1) # 70% training and 30% tes

lista1=[]
lista2=[]
for i in range(3,20):
    clf = DecisionTreeClassifier(random_state=0, max_depth=i).fit(x_train, y_train)
    lista1.append(metrics.accuracy_score(y_test, clf.predict(x_test)))
    lista2.append(metrics.accuracy_score(y_train, clf.predict(x_train)))
    
sb.lineplot(x=range(3,20),y=(lista1))    
sb.lineplot(x=range(3,20),y=(lista2)) #max_depth=7


clf = DecisionTreeClassifier(random_state=0, max_depth=7).fit(x_train, y_train)

clf.feature_importances_ #most important is motor, then health and then fidelity time

dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=['Fidelity Time', 'Percentage of Salary','Customer Monetary Value',
                                               'Premiums in LOB: Motor','Premiums in LOB: Household', 'Premiums in LOB: Health',
                                               'Premiums in LOB:  Life', 'Premiums in LOB: Work Compensations'],
                                filled=True,
                                rounded=True,
                                special_characters=True)  
graph = graphviz.Source(dot_data)

outliers1=outliers[['Fidelity Time', 'Percentage of Salary','Customer Monetary Value',
                   'Premiums in LOB: Motor','Premiums in LOB: Household', 'Premiums in LOB: Health',
                   'Premiums in LOB:  Life', 'Premiums in LOB: Work Compensations']].copy()

outliers_labels=clf.predict(outliers1)
outliers.drop(columns=['Claims Rate'], inplace=True)

outliers_clf=pd.concat([outliers, pd.DataFrame(outliers_labels)], axis=1, ignore_index=True)
outliers_clf.columns=['Fidelity Time', 'Educational Degree', 'Percentage of Salary',
                      'Geographic Living Area', 'Has Children (Y=1)','Customer Monetary Value',
                      'Premiums in LOB: Motor','Premiums in LOB: Household', 'Premiums in LOB: Health',
                      'Premiums in LOB:  Life', 'Premiums in LOB: Work Compensations', 'Labels']

final_data0=pd.concat([data_use, y], axis=1)
final_data=pd.concat([final_data0,outliers_clf], axis=0, ignore_index=True)

last_table=pd.pivot_table(final_data, values='Fidelity Time', index='Labels', 
                          aggfunc='count')
last_table.index=['clust_0', 'clust_1', 'clust_2', 'clust_3', 'clust_4', 'clust_5', 'clust_6']
last_table.columns=['Count']
##########################################################################################################
#PIE PLOTS CATEGORICAL PROPORTIONS
categoricals=['Educational Degree', 'Geographic Living Area', 'Has Children (Y=1)']
i=2
for colum in categoricals:
    cl=pd.DataFrame(final_data[final_data.iloc[:,11]==i].loc[:,colum].value_counts()).T
    #cl=pd.DataFrame(data.loc[:,colum].value_counts()).T
    fig1, ax1 = plt.subplots()
    ax1.pie(cl, labels=cl.columns.values, startangle=90, autopct='%1.1f%%')
    plt.legend(fontsize=16, loc="lower left")
    plt.title(str(colum)+' in Cluster '+str(i), fontsize=20)
    #plt.title(str(colum), fontsize=30)
### PIE PLOTS

sizes=[]
for i in range(last_table.shape[0]):
    sizes.append([last_table.iloc[i,0],last_table.iloc[:,0].sum()])

for i in range(last_table.shape[0]):
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes[i], startangle=90, colors=['orange','grey'], autopct='%1.1f%%')
    plt.title('clust_'+str(i),  fontsize=16)

## Creating clusters    

clusters1=pd.DataFrame(consum_clusters2.iloc[2,:]).T
clusters1=clusters1.append([consum_clusters2]*2, ignore_index=True)

clusters2=pd.DataFrame(km_client_clusters.iloc[0,:]).T
clusters2=clusters2.append([km_client_clusters.iloc[1,:]]*3, ignore_index=True)
clusters2=clusters2.append([km_client_clusters.iloc[2,:]]*3, ignore_index=True)

clusters=pd.concat([clusters2,clusters1], axis=1, ignore_index=True)
avg_data=pd.DataFrame(final_data.iloc[:,[0,2,5,6,7,8,9,10]].mean(axis=0)).T
avg_data.columns=list(range(0,8))
clusters_F=pd.concat([clusters,avg_data], axis=0, ignore_index=True)
clusters_F.index=['clust_0', 'clust_1', 'clust_2', 'clust_3', 'clust_4', 'clust_5', 'clust_6', 'data_avg']
clusters_F.columns=['Fidelity', '% Salary', 'CMV', 'Motor', 'Household', 'Health', 'Life', 'Work Comp']

clusters_F_norm= scaler.fit_transform(clusters_F)
clusters_F_norm =pd.DataFrame(clusters_F_norm, columns=clusters_F.columns, index=clusters_F.index)

clust=pd.DataFrame(clusters_F.iloc[6,:]).T

## Mean comparion plots
i=6
plt.plot(clusters_F_norm.iloc[3,:],'bo', markersize=16, color='r', label='Cluster 3')
plt.plot(clusters_F_norm.iloc[0,:],'bo', markersize=16, color='c', label='Cluster 0')
plt.plot(clusters_F_norm.iloc[i,:],'bo', markersize=16, color='m', label='Cluster '+str(i))
plt.plot(clusters_F_norm.iloc[7,:],'bo', markersize=16, color='k', label='Data Average')
plt.legend(fontsize=14)
plt.xticks(fontsize=24)
plt.yticks(fontsize=20)


