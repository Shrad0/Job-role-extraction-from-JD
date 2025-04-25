#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[3]:


df = pd.read_csv(r'C:\Users\Shraddha\Downloads\job_title_des.csv')
df.head(5)


# In[4]:


df.info


# In[5]:


# Remove the unnecessary column
df = df.drop(columns=["Unnamed: 0"])


# In[6]:


df.head(5)


# In[7]:


# Check for duplicate or missing values
df.duplicated().sum()
df.isnull().sum()


# In[8]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[9]:


# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df["Job Description"], df["Job Title"], test_size=0.2, random_state=42)


# In[10]:


# Create a pipeline for text processing and classification
model_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),  # Convert text to numerical features
    ("classifier", MultinomialNB())  # Train a Naive Bayes classifier
])


# In[11]:


# Train the model
model_pipeline.fit(X_train, y_train)


# In[16]:


# Evaluate the model on the test set
y_pred = model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(accuracy)


# In[15]:


classification_rep = classification_report(y_test, y_pred)
print(classification_rep)


# In[19]:


# Logistic Regression Model
logistic_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
    ("classifier", LogisticRegression(max_iter=500))
])


# In[21]:


# Train Logistic Regression
logistic_pipeline.fit(X_train, y_train)
y_pred_logistic = logistic_pipeline.predict(X_test)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
print(accuracy_logistic)


# In[23]:


# Random Forest Model
random_forest_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])


# In[24]:


# Train Random Forest
random_forest_pipeline.fit(X_train, y_train)
y_pred_rf = random_forest_pipeline.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(accuracy_rf)


# In[25]:


# Compare accuracy of all models
{
    "Naïve Bayes Accuracy": accuracy,
    "Logistic Regression Accuracy": accuracy_logistic,
    "Random Forest Accuracy": accuracy_rf
}


# In[26]:


# Define a function to predict the job role based on job description
def predict_job_role(job_description):
    predicted_role = logistic_pipeline.predict([job_description])[0]
    return predicted_role


# In[27]:


# Example test case
test_description = "We are looking for a skilled Python developer with expertise in Django and REST API development."
predicted_role = predict_job_role(test_description)
predicted_role


# In[28]:


# Example test case
test_description = "We are looking for a Data Scientist to analyze large datasets and develop predictive models. Responsibilities include building machine learning models, performing data analysis, and visualizing insights using Python, Pandas, and TensorFlow. Strong knowledge of statistics and experience with cloud platforms like AWS or GCP is preferred."
predicted_role = predict_job_role(test_description)
predicted_role


# In[29]:


# Example test case
test_description = "As a Backend Software Engineer, you will design and develop scalable APIs and backend systems. You should have expertise in Python, Django, and SQL databases. Knowledge of Docker, Kubernetes, and cloud services is a plus."
predicted_role = predict_job_role(test_description)
predicted_role


# In[30]:


# Example test case
test_description = "The Business Analyst will work closely with stakeholders to analyze business requirements and optimize processes. Strong analytical skills, SQL, and experience with BI tools like Power BI or Tableau are required"
predicted_role = predict_job_role(test_description)
predicted_role


# In[ ]:




