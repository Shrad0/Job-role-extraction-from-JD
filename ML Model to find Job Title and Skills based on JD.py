#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import string
import nltk


# In[2]:


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[4]:


from transformers import BertTokenizer, BertModel


# In[5]:


import torch


# In[ ]:


pip install --upgrade torch transformers


# In[ ]:


pip show transformers


# In[ ]:


pip install --upgrade optree


# In[6]:


# Step 1: Load Dataset
df = pd.read_csv(r'C:\Users\Shraddha\Downloads\new_jobs.csv')
df.head()


# In[7]:


df.info


# In[8]:


# Step 2: Preprocessing
nltk.download("stopwords")
nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


# In[9]:


def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

df["Cleaned_Description"] = df["Job Description"].apply(preprocess_text)


# In[10]:


# Step 3: Convert Text to Features
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(df["Cleaned_Description"])


# In[11]:


# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df["Job Title"], test_size=0.2, random_state=42)


# In[12]:


# Step 5: Train Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
log_reg_acc = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {log_reg_acc:.2f}")


# In[13]:


# Step 6: Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {rf_acc:.2f}")


# In[15]:


df = df.dropna(subset=['Job Title'])


# In[17]:


print(df[['Job Description', 'Cleaned_Description']].head())


# In[19]:


print(X_tfidf.shape)


# In[20]:


print(y_train.value_counts())


# In[21]:


print(y_test.value_counts())


# In[22]:


log_reg.fit(X_train[:50], y_train[:50])
print(log_reg.predict(X_test[:10]))


# In[23]:


from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
print(f"Naïve Bayes Accuracy: {accuracy_score(y_test, y_pred_nb):.2f}")


# In[14]:


# Step 7: BERT Embeddings
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")


# In[24]:


def get_bert_embedding(text):
    tokens = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        output = bert_model(**tokens)
    return output.last_hidden_state.mean(dim=1).squeeze().numpy()

df["BERT_Embedding"] = df["Cleaned_Description"].apply(get_bert_embedding)


# In[25]:


# Convert BERT embeddings to feature matrix
X_bert = np.vstack(df["BERT_Embedding"].values)
X_train_bert, X_test_bert, y_train, y_test = train_test_split(X_bert, df["Job Title"], test_size=0.2, random_state=42)


# In[26]:


# Step 8: Train Logistic Regression on BERT Features
log_reg_bert = LogisticRegression(max_iter=1000)
log_reg_bert.fit(X_train_bert, y_train)
y_pred_bert = log_reg_bert.predict(X_test_bert)
bert_acc = accuracy_score(y_test, y_pred_bert)
print(f"BERT + Logistic Regression Accuracy: {bert_acc:.2f}")


# In[27]:


# Function to Predict Job Title and Skills for a Given JD
def predict_job_title_and_skills(jd):
    cleaned_jd = preprocess_text(jd)
    jd_tfidf = vectorizer.transform([cleaned_jd])
    predicted_title = log_reg.predict(jd_tfidf)[0]
    predicted_skills = df[df["Job Title"] == predicted_title]["Skills"].values[0]
    return predicted_title, predicted_skills


# In[28]:


# Example Usage
example_jd = "Develop and maintain machine learning models, work with data pipelines, and deploy AI solutions."
title, skills = predict_job_title_and_skills(example_jd)
print(f"Predicted Job Title: {title}")
print(f"Predicted Skills: {skills}")


# In[29]:


# Example Usage
example_jd = "We are looking for a Data Scientist to analyze large datasets and develop predictive models. Responsibilities include building machine learning models, performing data analysis, and visualizing insights using Python, Pandas, and TensorFlow. Strong knowledge of statistics and experience with cloud platforms like AWS or GCP is preferred."
title, skills = predict_job_title_and_skills(example_jd)
print(f"Predicted Job Title: {title}")
print(f"Predicted Skills: {skills}")


# In[30]:


# Example Usage
example_jd = "As a Backend Software Engineer, you will design and develop scalable APIs and backend systems. You should have expertise in Python, Django, and SQL databases. Knowledge of Docker, Kubernetes, and cloud services is a plus."
title, skills = predict_job_title_and_skills(example_jd)
print(f"Predicted Job Title: {title}")
print(f"Predicted Skills: {skills}")


# In[31]:


# Example Usage
example_jd = "The Business Analyst will work closely with stakeholders to analyze business requirements and optimize processes. Strong analytical skills, SQL, and experience with BI tools like Power BI or Tableau are required"
title, skills = predict_job_title_and_skills(example_jd)
print(f"Predicted Job Title: {title}")
print(f"Predicted Skills: {skills}")


# In[ ]:




