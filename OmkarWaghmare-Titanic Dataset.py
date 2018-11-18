
# coding: utf-8

# In[1]:


# linear algebra and mathematical operations
import numpy as np 

# data processing and manupulation library
import pandas as pd   

# data visualization
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from matplotlib import style

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB


# In[2]:


test_df = pd.read_csv("test.csv")   #pandas read csv function
train_df = pd.read_csv("train.csv") 


# In[3]:


#data explotration
train_df.info()  #The training-set has 891 examples and 11 features + the target variable (survived). 2 of the features are floats, 5 are integers and 5 are objects.
train_df.describe()      # we can see that a mean of 38% survived from the trainins set
train_df.head(8)   #first few members 


# In[4]:


#for perdiction we need to know what data is missing, so taking a closer look into it
total = train_df.isnull().sum().sort_values(ascending=False)
percent_1 = train_df.isnull().sum()/train_df.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data.head(5)


# In[5]:


#The Embarked feature has only 2 missing values, which can easily be filled.
#It will be much more tricky, to deal with the ‘Age’ feature, which has 177 missing values.
#The ‘Cabin’ feature needs further investigation,
#but it looks like that we might want to drop it from the dataset, since 77 % of it are missing.


# In[6]:


def bar_chart(feature):
    survived = train_df[train_df['Survived']==1][feature].value_counts()
    dead = train_df[train_df['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))

bar_chart('Sex')
bar_chart('Pclass')


# In[7]:


#data preprocessing
#will drop ‘PassengerId’,'embarked','sibs','parch','ticket','cabin' from the train set
train_df = train_df.drop(['PassengerId'], axis=1)
train_df = train_df.drop(['Embarked'], axis=1)
train_df = train_df.drop(['SibSp'], axis=1)
train_df = train_df.drop(['Parch'], axis=1)
train_df = train_df.drop(['Ticket'], axis=1)
train_df = train_df.drop(['Cabin'], axis=1)
train_df = train_df.drop(['Fare'], axis=1)

#test_df = test_df.drop(['PassengerId'], axis=1)
test_df = test_df.drop(['Embarked'], axis=1)
test_df = test_df.drop(['SibSp'], axis=1)
test_df = test_df.drop(['Parch'], axis=1)
test_df = test_df.drop(['Ticket'], axis=1)
test_df = test_df.drop(['Cabin'], axis=1)
test_df = test_df.drop(['Fare'], axis=1)



# In[8]:


train_df.info()


# In[9]:


test_df.info()


# In[10]:


#as machine learning algorithms are not that great with names, 
#we'll convert all the names into a numerical form so that the processing is improved and is easier

data = [train_df, test_df]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:
    # extract titles
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # replace titles with a more common title or as Rare
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    # convert titles into numbers
    dataset['Title'] = dataset['Title'].map(titles)
    # filling NaN with 0, to get safe
    dataset['Title'] = dataset['Title'].fillna(0)
train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)


# In[11]:


#next we'll have to insert missing values for the field of age
#as we've seen earlier that age has a lot of missing values, and is also an improtant contributer to our prediction,
#so it makes sense to take extra efforts for filling these missing values
#an array of random values will be used to fill in the missing data 

data = [train_df, test_df]

for dataset in data:
    mean = train_df["Age"].mean()
    std = test_df["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    # compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    # fill NaN values in Age column with random values generated
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train_df["Age"].astype(int)
train_df["Age"].isnull().sum()


# In[12]:


#convert sex feature into a numeric forn, as our machine learning algorithms do not do very well will object type attributes.
genders = {"male": 0, "female": 1}
data = [train_df, test_df]

for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)


# In[13]:


#CREATING CATEGORIES


# In[14]:


#It is important to have a limited number of entries for every attribute,
#from our data we can see that the 'AGE' factor is all over the place
#so categorizing it will make our work more easier and help in increasing the accuracy of the prediction.

data = [train_df, test_df]
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6
    
#now lets see how its distributed
train_df['Age'].value_counts()


# In[15]:


#Now our data is complete, cleaned and in process-able form
#we will start out Modelling now
#four models have been considered:-
# 1 - Logistic Regression
# 2 - KNN
# 3 - Naive Bayes
# 4 - Decision Tree

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()


# In[16]:


# 1 - Logistic Regressionimport warnings
logreg = LogisticRegression(solver='lbfgs') #SOLVER SPECIFIED AS logreg.predict throws a future warning
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)


# In[17]:


#2 - KNN

#KNN knn = KNeighborsClassifier(n_neighbors = 3) 
#knn.fit(X_train, Y_train)  
#Y_pred = knn.predict(X_test)  
#acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(X_train, Y_train)
y_pred_knn = clf.predict(X_test)
acc_knn = round(clf.score(X_train, Y_train) * 100, 2)


# In[18]:


#3 - Gaussian Naive Bayes

gaussian = GaussianNB() 
gaussian.fit(X_train, Y_train)  
Y_pred = gaussian.predict(X_test)  
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)


# In[19]:


#4 - Decision tree
decision_tree = DecisionTreeClassifier() 
decision_tree.fit(X_train, Y_train) 
Y_pred = decision_tree.predict(X_test)  
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)


# In[20]:


#which is the best model??


# In[21]:


results = pd.DataFrame({
    'Model': [ 'KNN', 'Logistic Regression', 
               'Naive Bayes',  
              'Decision Tree'],
    'Score': [acc_knn, acc_log, 
              acc_gaussian, 
              acc_decision_tree]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)


# In[ ]:


#AS WE CAN SEE RANDOM FOREST PERFORMES THE BEST OUT OF ALL THE MODELS

