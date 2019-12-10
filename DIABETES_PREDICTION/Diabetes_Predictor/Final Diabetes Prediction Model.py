# -*- coding: utf-8 -*-
"""
Created on Fri Aug  01 10:49:33 2019

@authors: ARIKOSI MICHAEL OKURUT, KABAHINDA ROVENCE, NYIKA MELEBY, MWANJE MIKE
"""

#%% [markdown]
# # Diagnosing Diabetes Mellitus (dm)
#%% [markdown]
# ## Summary
#
# In this project, we use Logistic Regression and K-Nearest Neighbors (KNN) to diagnose dm.Both were able to classify patients. Logistic Regression and, KNN with high accuracy (> 80%)
#
# KNN required class balancing, scaling, and model tuning to perform, while Logistic Regression was still accurate without tuning (note: still had to stratify the train test split).
#

#%% [markdown]
# There are three links you may find important:
# - [A set of chronic kidney disease (CKD) data and other biological factors](./chronic_kidney_disease_full.csv).

#
# **Real-world problem**: Develop a medical diagnosis test system for diabetes
#
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yellowbrick.classifier
from yellowbrick.classifier import ClassificationReport
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import scipy.stats as stats
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


#%%
df = pd.read_csv('./chronic_kidney.csv')
data = df


#%%
data.head()

#%% [markdown]
# # Data Preprocessing
# # Data Mapping label encoding

#%%
data['class'] = data['class'].map({'ckd':1,'notckd':0})
data['htn'] = data['htn'].map({'yes':1,'no':0})
data['dm'] = data['dm'].map({'yes':1,'no':0})
data['cad'] = data['cad'].map({'yes':1,'no':0})
data['appet'] = data['appet'].map({'good':1,'poor':0})
data['ane'] = data['ane'].map({'yes':1,'no':0})
data['pe'] = data['pe'].map({'yes':1,'no':0})
data['ba'] = data['ba'].map({'present':1,'notpresent':0})
data['pcc'] = data['pcc'].map({'present':1,'notpresent':0})
data['pc'] = data['pc'].map({'abnormal':1,'normal':0})
data['rbc'] = data['rbc'].map({'abnormal':1,'normal':0})


#%%
data['dm'].value_counts()

#%% [markdown]
# Factors that may increase your risk of Diabetes include:
#
# - Diabetes - su(blood sugar), dm (diabetes mellitus)
# - High blood pressure - BP
# - Heart and blood vessel (cardiovascular) disease
# - Smoking
# - Obesity
# - Being African-American, Native American or Asian-American
# - Family history of kidney disease
# - Abnormal kidney structure
# - Older age - age

#%%
plt.figure(figsize = (15,15))
sns.heatmap(data.corr(), annot = True, cmap = 'coolwarm') # looking for strong correlations with "class" row



#%% [markdown]
# # # Feature selection using pearsong correlation heatmap, relatively high correlation with dm of > 0.3 considered.
# So as not to lose so many features that might aid in the final prediction

selected = data.drop(['sg','sod','hemo','pcv','rbcc','appet','bp','pcc','ba','sc','wbcc','pot','cad','ane'],axis='columns')
sns.pairplot(selected, hue='dm', height=2.5);
#%%
selected.shape

selected.columns


#%%
selected.isnull().sum()


#%% [markdown]
# We would only have 158 rows remaining if we drop the na columns.  One downside is that we reduce the overall power of our model when we feed in less data, and another is that we dont know if the fact that those values are null is related in some way to an additional variable.  If the latter is the case, throwing out that data could potentially skew our data.
#
# I am going to drop in this case and see how the model performs.
#
# Generally speaking in situations where we are providing patients with a diagnosis, we want to err on the side of false positives. In this specific case, a false negative would be telling a patient that is dm-positive that they do not have dm, and the result could be catastrophic if the mistake is not caught. This would be a "worse" mistake than telling someone who is dm-negative that they have dm, as they would be brought in for further testing and find out that they are actually dm-negative

#%%
#Handling Null values is to be done by either backward filling or dropping all nulls

#Backwardfilling
#data12=selected.fillna(method = 'bfill', limit=10)

# Dropping Null values
data12 = selected
data12.dropna(inplace=True)


data12.isnull().sum() #Checking for any Null values
data12

# Depicts all features in the selected dataframe as histograms
pd.DataFrame.hist(data12,figsize =[15,15])

#%% [markdown]
# # Modeling

# Defining visualization functions
#Plotting Histograms
def plotHistogram(values,label,feature,title):
    sns.set_style("whitegrid")
    plotOne = sns.FacetGrid(values, hue=label,aspect=2)
    plotOne.map(sns.distplot,feature,kde=False)
    plotOne.set(xlim=(0, values[feature].max()))
    plotOne.add_legend()
    plotOne.set_axis_labels(feature, 'Proportion')
    plotOne.fig.suptitle(title)
    plt.show()

plotHistogram(data12,"dm",'bgr','Blood Glucose Random vs Diagnosis (Blue = Healthy; Orange = Diabetes)')


# Plotting Learning curves
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Plots a learning curve. http://scikit-learn.org/stable/modules/learning_curve.html
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    return plt


#%% [markdown]
# ## Logistic Regression Model

#%%
from sklearn.linear_model import LogisticRegression


#%%
logreg = LogisticRegression()

# Specifying the target variable (dm) and its estimators (other features except dm)
X = data12.drop("dm", axis=1)
y = data12['dm']


#%%
#Splitting the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X,y, stratify = y, shuffle = True)


#%%

logreg.fit(X_train,y_train)

#%%
# Pickle is used to serialize  (Save to disk memory ) and deserialize (Load from disk) the machine learning models
import pickle
#%%
# save the model to disk using pickle
pickle.dump(logreg, open('logreg_deployed.pkl', 'wb'))


#%%
test_pred = logreg.predict(X_test)
train_pred = logreg.predict(X_train)

#Plot learning curve for Logistic Regression model
plot_learning_curve(logreg, 'Learning Curve For Logistic regression', X_train,y_train, (0.60,1.1), 10)


#%%
#Test accuracy of the machine learning model
from sklearn.metrics import accuracy_score, confusion_matrix


#%%
print('Train Accuracy: ', accuracy_score(y_train, train_pred))
print('Test Accuracy: ', accuracy_score(y_test, test_pred))

#%%
tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()

print(f'True Neg: {tn}')
print(f'False Pos: {fp}')
print(f'False Neg: {fn}')
print(f'True Pos: {tp}')

# Further Detailed Classification using YellowBrick Classification report for Logistic Regression model
# Instantiate the classification model and visualizer
visualizer = ClassificationReport(logreg, classes=[1.0,0.0])
visualizer.fit(X_train, y_train) # Fit the training data to the visualizer
visualizer.score(X_test, y_test) # Evaluate the model on the test data
g = visualizer.poof() # Draw/show/poof the data

#%%
#testing deployed model
custom_data = [{'age': 68 ,
         'al': 0 ,
         'su': 0,
         'rbc': 1,
         'pc' : 1,
         'bgr': 100,
         'bu' : 54,
         'htn' : 0,
         'pe': 0,
         'class' :1
                 }]
labels =['age', 'al', 'su', 'rbc', 'pc', 'bgr', 'bu', 'htn', 'pe', 'class']

#   Creates DataFrame.
test = pd.DataFrame(custom_data)
test = test[labels]
test.columns
#test1 = logreg.fit(test)
predictor = logreg.predict(test)
print(predictor[0])


#%% [markdown]
# ## K-Nearest Neighbors Classifier
#%% [markdown]
# I am going to balance the classes here before using KNN. Logistic regression was able to make accurate predictions even when trained on unbalanced classes,
# but KNN is more sensitive to unbalanced classes

#%%
data12["dm"].value_counts()

# Splitting the dm classes i.e 0.0 and 1.0
dm_0 = data12[data12["dm"] == 0]
dm_1 = data12[data12["dm"] == 1]

#Oversampling the deficient class so as to balance them
selected_class_1_over = dm_1.sample(162, replace=True)
selected_test_1_over = pd.concat([dm_0,selected_class_1_over], axis = 0)
selected_test_1_over["dm"].value_counts()
#%%
# Specifying the target variable (dm) and its estimators (other features except dm)
X = selected_test_1_over.drop("dm", axis=1)
y = selected_test_1_over["dm"]


#%%
#Splitting the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 42)


#%%
# Scaling the training and test data for KNN using StandardScaler()
ss = StandardScaler()
ss.fit(X_train)
X_train = ss.transform(X_train)
X_test = ss.transform(X_test)


#%%
from sklearn.neighbors import KNeighborsClassifier


#%%
knn = KNeighborsClassifier()
params = {
    "n_neighbors":[3,5,7,9],
    "weights":["uniform","distance"],
    "algorithm":["ball_tree","kd_tree","brute"],
    "leaf_size":[25,30,35],
    "p":[1,2]
}

#Grid search CV is for hyper-tuning the estimators so as to have a better classication model.
gs = GridSearchCV(knn, param_grid=params)
model = gs.fit(X_train,y_train)
preds = model.predict(X_test)
accuracy_score(y_test, preds)

plot_learning_curve(knn, 'Learning Curve For K-Nearest Neighbour Classifier', X_train,y_train, (0.60,1.1), 10)

#%%
tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

print(f'True Neg: {tn}')
print(f'False Pos: {fp}')
print(f'False Neg: {fn}')
print(f'True Pos: {tp}')


#YELLOWBRICK CLASSIFICATION REPORT
# Instantiate the classification model and visualizer
visualizer = ClassificationReport(knn, classes=[1.0,0.0])
visualizer.fit(X_train, y_train) # Fit the training data to the visualizer
visualizer.score(X_test, y_test) # Evaluate the model on the test data
g = visualizer.poof() # Draw/show/poof the data

#%%
test_knn_pred = knn.predict(X_test)
train_knn_pred = knn.predict(X_train)

#%%
# Printing accuracy scores of the  test and train data of the model
print('Train Accuracy: ', accuracy_score(y_train, train_knn_pred))
print('Test Accuracy: ', accuracy_score(y_test, test_knn_pred))

#%%
import pickle
#%%
# save the model to disk using pickle
pickle.dump(model, open('knn_deployed.pkl', 'wb'))

#Testing out the deserialzed KNN model
#model12 = pickle.load(open('./knn_deployed.pkl', 'rb'))
#preds1 = model12.predict(X_test)
#accuracy_score(y_test, preds1)
