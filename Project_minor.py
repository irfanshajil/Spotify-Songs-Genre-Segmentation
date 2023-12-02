#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv(r"C:\Users\irfan\OneDrive\Desktop\Inter_proj\Project 2  - Spotify Songs’ Genre Segmentation\spotify dataset.csv")
data


# In[3]:


data.describe()


# In[4]:


data.isnull()


# In[5]:


data.isnull().sum()


# In[6]:


data.dropna(inplace=True)


# In[7]:


data.isnull().sum()


# In[8]:


sns.histplot(data['track_popularity'],)
plt.title("Distribution of Track Popularity")
plt.xlabel("Popularity")
plt.ylabel("Frequency")
plt.show()


# In[9]:


sns.countplot(x='playlist_genre', data=data)
plt.title('Genre Distribution')
plt.show()


# In[10]:


columns_for_plot = data.iloc[:, [3, 15, 16, 17, 18, 19, 20, 21]]


# In[11]:


plt.figure(figsize=(14, 10))
sns.heatmap(columns_for_plot.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[12]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# In[13]:


columns_for_plot = data.iloc[:, [3, 15, 16, 17, 18, 19, 20, 21]]


# In[14]:


features = data[['danceability', 'energy', 'valence']]


# In[15]:


scaler = StandardScaler()
features_std = scaler.fit_transform(features)


# In[16]:


kmeans = KMeans(n_clusters=4, random_state=42)
data['cluster'] = kmeans.fit_predict(features_std)


# In[17]:


plt.figure(figsize=(12, 8))
sns.scatterplot(x='danceability', y='energy', hue='playlist_genre', data=data, palette='viridis', legend='full')
plt.title('Clusters based on Playlist Genres')
plt.xlabel('Danceability')
plt.ylabel('Energy')
plt.show()


# In[18]:


plt.figure(figsize=(12, 8))
sns.scatterplot(x='danceability', y='energy', hue='playlist_genre', data=data, palette='viridis', legend='full')
plt.title('Clusters based on Playlist Names')
plt.xlabel('Danceability')
plt.ylabel('Energy')
plt.show()


# In[19]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# In[20]:


ip = data.select_dtypes(include=['number', 'bool'])
op = data['playlist_genre']


# In[21]:


xtrain, xtest, ytrain, ytest = train_test_split(ip, op, train_size=0.8, random_state=42)


# In[25]:


rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(xtrain, ytrain)


# In[26]:


predictions = rfc.predict(xtest)


# In[27]:


accuracy = accuracy_score(ytest, predictions)


# In[28]:


print(f"Accuracy: {accuracy:.2%}")

