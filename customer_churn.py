#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings


# importing required library which is used to import dataset
import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report

from sklearn.metrics import confusion_matrix
print("cell running succesfully")




# In[43]:


#importing dataset
newdf=pd.read_csv('telecom.csv')
newdf.head()


# In[ ]:


#NUll----> accuracy(--)# 

# data preprocessing 
# NULL value eliminate
# ammount : - 10 , 40, 20, 23, NULL, 23, 40
# drop(ammount) - > mean --->NULL
# regression , classification---------(ML Models --- > decision treee  , randomforest , ANN)
# (binary)classification   --->YES/NO
# yes(1) / no(0)
# one hot encoding
#payment = electronic mail , mailcheck,credit
#payment_electronic_bills --> yes , no , no
#payment_check-------------->no ,yes , no

#payment_credit_card------->no


# In[3]:


newdf.isnull().sum().sum()


# In[4]:


newdf.info()


# In[5]:


#total cherges column was detected as object we will change it to the numeric data
newdf['TotalCharges']=pd.to_numeric(newdf['TotalCharges'],errors='coerce')
newdf.info()


# In[44]:


newdf.dropna(inplace=True)


# In[7]:


#remove CustomerId column
newdf.drop(columns='customerID',inplace=True)
newdf.head()


# In[8]:


newdf.PaymentMethod.unique()


# In[9]:


#removing automatic 
newdf['PaymentMethod']=newdf['PaymentMethod'].str.replace(' (automatic)',  '',regex=False)
newdf.PaymentMethod.unique()


# In[10]:


#importing Libraries for Data Visualization
import matplotlib.pyplot as plt


# In[11]:


#create a figure
fig=plt.figure(figsize=(10,6))
ax=fig.add_subplot(111)

# proportion of observation of each class
prop_response=newdf['Churn'].value_counts(normalize=True)

#create a bar plot to showing percentage of churn 
prop_response.plot(kind='bar',ax=ax,color=['red','green'])



#set title and labels 
ax.set_title('Proportion of response of variable',fontsize=18, loc='left')

ax.set_xlabel('Churn',fontsize=14)
ax.set_ylabel('Proportion of observation')

ax.tick_params(rotation='auto')

#eliminate the frame from plot 
spine_names=('top','right','bottom','left')

for spine_name in spine_names:
    ax.spines[spine_name].set_visible(False)
  




# In[12]:


# def percentage_stacked_plot(column_to_plot, super_title):
#     number_of_columns=2
#     number_of_rows= math.ceil(len(column_to_plot)/2)
    
#     # to create a figure
#     fig=plt.figure(figsize=(12,5*no_of_rows))
#     fig.suptitle(super_title, fontsize=22,y=.95)
    
#     # loop to each column name to create a subplot
    
#     for index,column in enumerate(column_to_plot,1):
#         
        
#         #create a subplot
#         ax=fig.add_subplot(number_of_rows,number_ofcolumn,index)
#         prop_by_independent=pd.crosstab(newdf[column],newdf['Churn']).apply(lambda x: x/x.sum()*100,axis=1 )
#         prop_by_independent.plot(kind='bar',ax=ax,stacked=True,rot=0,color['red','green'])
        
#         #set the legend in the top of the table
        
#         ax.legend(loc='upper right',bbox_to_anchor=(0.62,0.5,0.5,0.5),title='Churn',fancybox=True)
        
#         #set Title and lables
        
#         ax.set_title('Proportion of observation by' +column,fontsize=16, loc='left')
        
        
#         ax.tick_params(rotation='auto')
        
#         #eliminate the frame from the slot
#         spine_names  =  ('top','right','bottom','left')
#         for spine_name in spine_names:
#             ax.spines[spine_name].set_visible(False)
    
    


# In[13]:


# demographic_columns=['gender','Seniorcitizenship','Partner','Dependents']

# percentage_stacked_plot(demographic_columns,'Demographic Information')


# In[14]:


def compute_mutual_info(categorical_serie):
    return sklearn.metrics.mutual_info_score(categorical_serie,newdf.Churn)

categorical_variables=newdf.select_dtypes(include=object).drop('Churn',axis=1)

feature_importance=categorical_variables.apply(compute_mutual_info).sort_values(ascending=False)

print(feature_importance)


# In[15]:


df_transform=newdf.copy()
lable_encoding_columns=['gender','Partner','Dependents','PaperlessBilling','PhoneService','Churn']

for column in lable_encoding_columns:
    if column=='gender':
        df_transform[column]= df_transform[column].map({'Female':1,'Male':0})
    else :
        df_transform[column]=df_transform[column].map({'Yes':1,'No':0})
df_transform.head()        


# In[16]:


one_hot_encoding_column=['MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaymentMethod']

df_transform=pd.get_dummies(df_transform,columns=one_hot_encoding_column)


# In[17]:



    
    
print("Running Succesfully")    
    
df_transform.dropna(inplace=True)

# In[18]:


# Machine learnig algo
# x----->df.drop(colom='churn) model --->feature,target x=df-churn , y=
# df --> feature--->target
# x = df-target
# y = target
# feature ---> x_train,x_test
# target ---> y_train , y_test;
# X_train , x----- = train_test_Split(X,Y ,testsize = 0.20 ,
# model --

X = df_transform.drop(columns='Churn')

Y = df_transform.loc[:,'Churn']

print(X.columns)

print(Y.name)


# In[19]:


# Split the data in training and testing
X_train, X_test , Y_train , Y_test= train_test_split(X,Y,test_size=0.20,random_state=40,shuffle=True)
print('succesfully runned !')


# In[20]:


#data normalization with sklearn

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[21]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X,Y)


# In[22]:


from sklearn import tree
plt.figure(figsize=(70,70))
tree.plot_tree(model,filled=False)
plt.show()
print("Success!")


# In[23]:


from sklearn.model_selection import cross_val_score


# In[24]:


dt_cv = DecisionTreeClassifier()
print("Cross_validation_Accuracy\n")
cv_dt = cross_val_score(dt_cv,X_train,Y_train, cv=5,scoring='accuracy').mean()
print('For Decision Trees Classifier :',round(cv_dt,3))



# In[25]:


rdf = RandomForestClassifier(criterion = 'entropy')


# In[26]:


dt_cv.fit(X_train, Y_train)
# XGBoostclassifier.fit(X_train, Y_train)
rdf.fit(X_train, Y_train)
# SVMclassifier.fit(X_train, Y_train)
# LOGISTICclassifier.fit(X_train, Y_train)
# KNNclassifier.fit(X_train, Y_train)
# GNBclassifier.fit(X_train, Y_train)
# BaggingClassifier.fit(X_train, Y_train)
# NeuralNetworkClassifier.fit(X_train, Y_train)
# CatBoost.fit(X_train, Y_train)
# GBM.fit(X_train, Y_train)


# In[27]:


dtpred = dt_cv.predict(X_test)
dtaccuracy = accuracy_score(Y_test,dtpred)
print("the accuracy of the decision tree classifier: ",dtaccuracy)


# In[28]:


rdfpred = rdf.predict(X_test) 
rdfaccuracy = accuracy_score(Y_test,rdfpred)
print("the accuracy of the random forest classifier: ",rdfaccuracy)


# In[29]:


#confusion matrix and classification report of the the models 
# Random Forest 
cmRF = confusion_matrix(Y_test,rdfpred)
cmRF


# In[30]:


print(classification_report(Y_test,rdfpred))


# In[31]:


# Decision  treete
cmdt = confusion_matrix(Y_test,dtpred)
print(cmdt)


# In[32]:


print(classification_report(Y_test,dtpred))


# In[33]:


# By applying K-Fold Cross Validation
accuracies = cross_val_score(estimator = dt_cv, X= X_train , y=Y_train,cv = 100)
print("The accuracy of decision tree :",accuracies.mean()*100)


# In[34]:


accuracies7 = cross_val_score(estimator = rdf, X = X_train, y = Y_train, cv=100)
print("Ac  curacy for Random Forest : {:.2f} %".format(accuracies7.mean()*100))


# In[35]:


# Predictor = my_model.predict([[1,0,1,0,0.00000,0,1,0.115,0.0125,1,1,1,0,0,1,0,0,0,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]


# save the final model 
import pickle
pickle.dump(rdf,open('rdf','wb'))

my_model = pickle.load(open('rdf','rb'))






# In[36]:





# In[37]:





# In[38]:
# Predictor = my_model([[1,0,1,0,0.00000,0,1,0.115,0.0125,1,1,1,0,0,1,0,0,0,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
#
#
# # In[39]:
# Predictor
# def prediction(Predictor):
#     if(Predictor[0]==[0]):
#         print("Customer is not churning :")
#     else:
#         print("Customer is churning: ")
# prediction(Predictor)

with open('my_model.pkl','wb') as files:
    pickle.dump(rdf,files)

# In[40]:


# new = np.array([1,0,1,0,0.000,1,0,0.115,0.00125,1,0,0,1,0,0,0,0,1,0]).reshape(1,-1)
# my_model.predict(new)


# In[45]:


print("ALL Cells Runing ")

# In[ ]:




