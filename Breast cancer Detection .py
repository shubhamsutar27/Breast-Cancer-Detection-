#!/usr/bin/env python
# coding: utf-8

# In[64]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('breastCancer.csv')
df


# In[3]:


df.info()


# In[4]:


df['clump_thickness'].value_counts()


# In[5]:


df['size_uniformity'].value_counts()


# In[6]:


df['shape_uniformity'].value_counts()


# In[7]:


df['marginal_adhesion'].value_counts()


# In[8]:


df['epithelial_size'].value_counts()


# In[9]:


df['bare_nucleoli'].value_counts()


# In[10]:


df['bland_chromatin'].value_counts()


# In[11]:


df['normal_nucleoli'].value_counts()


# In[12]:


df['mitoses'].value_counts()


# In[13]:


df['class'].value_counts()


# # EXPLORATORY DATA ANALYSIS:

# In[14]:


df.drop(columns=['id'],inplace=True)


# Id is not useful feature for breast cancer delection. Hence I am dropping it in the beginning itself.

# In[15]:


df.describe()


# In[16]:


df.isnull().sum()


# In[17]:


df.duplicated().sum()


# In[18]:


duplicate_rows = df[df.duplicated()]
duplicate_rows


# In[19]:


df.dtypes


# 'bare_nucleoli' Feature is having '?' charecter. I will replace it with NaN.

# In[20]:


# replace '?' with NaN
df['bare_nucleoli'] = df['bare_nucleoli'].replace('?', np.nan)


# In[21]:


df.isna().sum()


# In[22]:


# remove records with missing values
df = df.dropna(subset=['bare_nucleoli'])


# In[23]:


# Convert 'bare_nucleoli' feature to integer.
df['bare_nucleoli'] = pd.to_numeric(df['bare_nucleoli'])


# In[24]:


df.dtypes


# In[25]:


df.isna().sum()


# In[26]:


# creating a new data frame with counts of cancerous and non-cancerous cases
counts = df.groupby(['class','clump_thickness']).size().unstack(0)

counts.plot(kind='bar')
plt.xlabel('Clump Thickness')
plt.ylabel('Count')
plt.title('Number of cancerous and non cancerous cases by clump thickness')
plt.show()


# * Clump thickness more than 5 indicates that the patient has high chances of contracting the cancer disease. 
# * Clump thickness 9 and 10 suggests that cancer is developed in the patient. 

# In[27]:


counts = df.groupby(['class','size_uniformity']).size().unstack(0)

counts.plot(kind='bar')
plt.xlabel('Size Uniformity')
plt.ylabel('Count')
plt.title('Number of cancerous and non cancerous cases by Size uniformity')
plt.show()


# * More than 50% data is belonging to class 1 in size uniformity. It is very difficult to predict how size unformity is related to breast cancer with this data.
# * Still Roughly we can say than high value of size uniformity is more prone to contract breast cancer.

# In[28]:


counts = df.groupby(['class','shape_uniformity']).size().unstack(0)

counts.plot(kind='bar')
plt.xlabel('shape Uniformity')
plt.ylabel('Count')
plt.title('Number of cancerous and non cancerous cases by Shape uniformity')
plt.show()


# * Shape uniformity plot is very much similar to size uniformity. 
# * There are high chances that these two features are highly correlated.I will confirm it with correlation matrix. If it is, then dropping either of the feature would be a good choice.

# In[29]:


counts = df.groupby(['class','marginal_adhesion']).size().unstack(0)

counts.plot(kind='bar')
plt.xlabel('Marginal Adhesion')
plt.ylabel('Count')
plt.title('Number of cancerous and non cancerous cases by Marginal Adhesion')
plt.show()


# * More than 50% records having class 1 in marginal adhesion. If balanced records were given we could have understood the pattern in better way.
# * We can roughly say that higher marginal adhesion indicates patient is prone to contract breast cancer. 

# In[30]:


counts = df.groupby(['class','epithelial_size']).size().unstack(0)

counts.plot(kind='bar')
plt.xlabel('Epithelial Size')
plt.ylabel('Count')
plt.title('Number of cancerous and non cancerous cases by Epithelial Size')
plt.show()


# * Higher the Epithelial size , higher the chances of contracting the disease.

# In[31]:


counts = df.groupby(['class','bare_nucleoli']).size().unstack(0)

counts.plot(kind='bar')
plt.xlabel('Bare Nucleoli')
plt.ylabel('Count')
plt.title('Number of cancerous and non cancerous cases by Bare Nucleoli')
plt.show()


# In[32]:


counts = df.groupby(['class','bland_chromatin']).size().unstack(0)

counts.plot(kind='bar')
plt.xlabel('Bland Chromatin')
plt.ylabel('Count')
plt.title('Number of cancerous and non cancerous cases by Bland Chromatin')
plt.show()


# In[33]:


counts = df.groupby(['class','normal_nucleoli']).size().unstack(0)

counts.plot(kind='bar')
plt.xlabel('Normal Nucleoli')
plt.ylabel('Count')
plt.title('Number of cancerous and non cancerous cases by Normal Nucleoli')
plt.show()


# In[34]:


counts = df.groupby(['class','mitoses']).size().unstack(0)

counts.plot(kind='bar')
plt.xlabel('mitoses')
plt.ylabel('Count')
plt.title('Number of cancerous and non cancerous cases by Mitoses')
plt.show()


# In[35]:


# count the number of occurrences of each value in 'clump_thickness'
counts = df['clump_thickness'].value_counts()

# create a pie chart of the counts
plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
plt.title('Distribution of Clump Thickness')
plt.show()


# In[36]:


counts = df['size_uniformity'].value_counts()
plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
plt.title('Distribution of Size uniformity')
plt.show()


# In[37]:


counts = df['shape_uniformity'].value_counts()
plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
plt.title('Distribution of Shape uniformity')
plt.show()


# In[38]:


counts = df['marginal_adhesion'].value_counts()
plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
plt.title('Distribution of Marginal Adhesion')
plt.show()


# In[39]:


counts = df['epithelial_size'].value_counts()
plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
plt.title('Distribution of Epithelial Size')
plt.show()


# In[40]:


counts = df['bare_nucleoli'].value_counts()
plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
plt.title('Distribution of Bare Nucleoli')
plt.show()


# In[41]:


counts = df['bland_chromatin'].value_counts()
plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
plt.title('Distribution of Bland Chromatin')
plt.show()


# In[42]:


counts = df['normal_nucleoli'].value_counts()
plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
plt.title('Distribution of Normal Nucleoli')
plt.show()


# In[43]:


counts = df['mitoses'].value_counts()
plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
plt.title('Distribution of mitoses')
plt.show()


# In[44]:


corr_matrix = df.corr()
sns.heatmap(corr_matrix,annot=True,cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()


# * Correlation Heatmap shows that their is high correlation between size uniformity and shape uniformity. 
# * I will drop size uniformity and proceed further with model building.

# In[45]:


df.drop(columns=['size_uniformity'],inplace=True)
df


# # MODEL BUILDING:

# In[46]:


# Split the data into independent and dependent features.
X = df.drop('class',axis=1)
y = df['class']


# In[47]:


# Split the dataset into training and testing sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# ## Model-1 Logistic Regression:

# In[76]:


model1 = LogisticRegression()

# Fit the model on training data
model1.fit(X_train,y_train)

#Make predictions on testing data
y_pred = model1.predict(X_test)

#Evaluate the performance of the model
accuracy1 = accuracy_score(y_test,y_pred)
log_reg_f1 = f1_score(y_test, y_pred,pos_label=4)
report = classification_report(y_test,y_pred)
print('Accuracy: ',accuracy1)
print('Report:\n', report)


# ## Model-2 Support Vector Machine:

# ### Support vector machine with linear Kernel 

# In[77]:


model2 = SVC(kernel='linear')

# Fit the model on training data
model2.fit(X_train,y_train)

# Make predictions on testing data
y_pred = model2.predict(X_test)

#Evaluate the performance of the model
accuracy2 = accuracy_score(y_test,y_pred)
report = classification_report(y_test,y_pred)
svm_f1 = f1_score(y_test, y_pred,pos_label=4)
print('Accuracy: ',accuracy2)
print('Report:\n', report)


# ### Support Vector Machine with RBF Kernal

# In[50]:


rbf_model = SVC(kernel='rbf')
rbf_model.fit(X_train,y_train)
y_pred_rbf = rbf_model.predict(X_test)

accuracy_rbf = accuracy_score(y_test,y_pred_rbf)
print('Accuracy (RBF Kernel): ',accuracy_rbf)


# ### Support Vector Machine with polynomial Kernal 

# In[51]:


poly_model = SVC(kernel='poly',degree=3)
poly_model.fit(X_train,y_train)
y_pred_poly = poly_model.predict(X_test)

accuracy_poly = accuracy_score(y_test,y_pred_poly)
print('Accuracy (Polynomial kernel): ',accuracy_poly)


# ## Model-3 K-Nearest Neighbor:

# In[78]:


# Train the KNN model
model3 = KNeighborsClassifier(n_neighbors=5)
model3.fit(X_train,y_train)

# Evaluate the performance of the model
y_pred = model3.predict(X_test)
accuracy3 = accuracy_score(y_test,y_pred)
knn_f1 = f1_score(y_test, y_pred,pos_label=4)
report = classification_report(y_test,y_pred)
print('Accuracy: ',accuracy3)
print('Report:\n', report)


# ## Model-4 Decision Tree:

# In[79]:


# Train the decision tree model
model4 = DecisionTreeClassifier()
model4.fit(X_train,y_train)

# Evaluate the performance of the model
y_pred = model4.predict(X_test)
accuracy4 = accuracy_score(y_test,y_pred)
report = classification_report(y_test,y_pred)
dt_f1 = f1_score(y_test, y_pred,pos_label=4)
print('Accuracy: ',accuracy4)
print('Report:\n', report)


# ## Model-5 Random Forest:

# In[81]:


# Train the random forest model
model5 = RandomForestClassifier(n_estimators=100,random_state=42)
model5.fit(X_train,y_train)

# Evaluate the performance of the model
y_pred = model5.predict(X_test)
accuracy_rf = accuracy_score(y_test,y_pred)
report = classification_report(y_test,y_pred)
print('Accuracy: ',accuracy_rf)
print('Report:\n',report)


# ### Hyperparameter Tunning for Random Forest Classifier:

# In[55]:


param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}


# In[56]:


# Create a random forest model
rf = RandomForestClassifier(random_state=42)

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(rf, param_grid, cv=5)
grid_search.fit(X_train, y_train)


# In[57]:


# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)


# In[82]:


# Train a new random forest model with the best hyperparameters
rf = RandomForestClassifier(**best_params, random_state=42)
rf.fit(X_train, y_train)

# Evaluate the performance of the model
y_pred = rf.predict(X_test)
accuracy5 = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
rf_f1 = f1_score(y_test, y_pred,pos_label=4)
print("Accuracy:", accuracy5)
print("Report:\n", report)


# ## Model-6 Naive Bayes:

# In[83]:


# Create the Gaussian Naive Bayes model
model6 = GaussianNB()

#Fit the model to the training data
model6.fit(X_train,y_train)

# make the predictions on the testing data
y_pred = model6.predict(X_test)

# Evaluate the accuracy of the model
accuracy6 = accuracy_score(y_test,y_pred)
nb_f1 = f1_score(y_test, y_pred,pos_label=4)
print("Accuracy: {:.2f}%".format(accuracy6 * 100))
report = classification_report(y_test, y_pred)
print("Report:\n", report)


# In[84]:



# create a list of model names
model_names = ['Logistic Regression', 'SVM', 'KNN', 'Decision Tree', 'Random Forest', 'Naive Bayes']

# create a list of model accuracies
model_accuracies = [accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6]

# create a list of model F1-scores
model_f1_scores = [log_reg_f1, svm_f1, knn_f1, dt_f1, rf_f1, nb_f1]

# create a dictionary with the model names, accuracies, and F1-scores
model_summary = {'Model Name': model_names, 'Accuracy': model_accuracies, 'F1-Score': model_f1_scores}

# create a DataFrame from the dictionary
summary_df = pd.DataFrame(model_summary)

# display the DataFrame
summary_df


# In[ ]:




