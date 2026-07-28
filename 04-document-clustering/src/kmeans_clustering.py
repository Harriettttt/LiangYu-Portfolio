#!/usr/bin/env python
# coding: utf-8

# In[76]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
import re
import nltk
from nltk.corpus import stopwords
import string
import matplotlib.pyplot as plt


# In[102]:


from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.manifold import TSNE


# In[4]:


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


# In[308]:


import pandas as pd
# data = pd.read_csv('movie.csv')
data = pd.read_csv('article.csv')
# data = pd.read_csv('news.csv')
data.head()


# In[108]:


def preprocess_text(text):
    # Text Cleaning
    translator = str.maketrans('', '', string.punctuation)
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '',text)
    text_no_punct = text.translate(translator)
    text_lower = text_no_punct.lower()
    # Tokenization
    tokens = word_tokenize(text_lower)
    # Stop Word Removal
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Stemming or Lemmatization 
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    # Join the tokens back into a single string
    preprocessed_text = ' '.join(lemmatized_tokens)

    return preprocessed_text


# In[309]:


#data['cleaned_text'] = data['review'].apply(preprocess_text) #this is for the movie review dataset
data['cleaned_text'] = data['Text'].apply(preprocess_text) #this is for the article dataset
# data['cleaned_text'] = data['Excerpt'].apply(preprocess_text) #this is for the news dataset


# In[312]:


# Encode the sentiment labels (positive -> 1, negative -> 0)
label_encoder = LabelEncoder()
# data['encoded_label'] = label_encoder.fit_transform(data['sentiment'])#this is for the movie review dataset
data['encoded_label'] = label_encoder.fit_transform(data['Category'])#this is for the article dataset
# data['encoded_label'] = label_encoder.fit_transform(data['Category'])#this is for the news dataset


# In[313]:


data.head()


# In[314]:


X = data['cleaned_text']
y = data['encoded_label']


# In[323]:


# Feature Extraction using Bag of Words (BoW)
bow_vectorizer = CountVectorizer(max_features=1000)  # Using the 1000 most frequent words
X_bow = bow_vectorizer.fit_transform(X)
# Standardize the data (since PCA is sensitive to scaling)
scaler = StandardScaler(with_mean=False)  
X_scaled = scaler.fit_transform(X_bow)


# In[328]:


tfidf_vectorizer = TfidfVectorizer(max_features=1000,ngram_range=(1, 2))  
X_tfidf = tfidf_vectorizer.fit_transform(X)
# Standardize the data (since PCA is sensitive to scaling)
scaler = StandardScaler(with_mean=False)  
X_scaled = scaler.fit_transform(X_tfidf)


# In[329]:


# Dimensionality Reduction using PCA (3D)
pca = PCA(n_components=3)  # Reduce to 3 dimensions
X_pca = pca.fit_transform(X_scaled.toarray())

# 3D Visualization of the real distribution 
fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111, projection='3d')

# Plot the data points in 3D, using sentiment to color the points
sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=data['encoded_label'], cmap='coolwarm', marker='o', s=50)

# Set labels and title
ax.set_title('3D PCA Visualization of News Data Distribution')
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('PCA Component 3')
plt.colorbar(sc, label='Label')

# Show plot
plt.show()


# In[330]:


num_clusters = len(np.unique(y))

from sklearn.cluster import KMeans
model = KMeans(n_clusters=num_clusters, random_state = 11)
model.fit(X_pca)
clusters = model.labels_.tolist()


# In[320]:


num_clusters


# In[331]:


from scipy.optimize import linear_sum_assignment

# build the cost matrix
num_classes = len(np.unique(y))
cost_matrix = np.zeros((num_classes, num_classes), dtype=int)

y_numpy = np.array(y)
clusters_numpy = np.array(clusters)

# fill the cost matrix
for i in range(num_classes):
    for j in range(num_classes):
        # Calculate the intersections between true labels 'i' and cluster 'j'
        cost_matrix[i, j] = np.sum((y_numpy == i) & (clusters_numpy == j))
        
# Use the Hungarian algorithm to find the optimal solution
row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)

# Assign new labels to the clusters based on the optimal matching
new_cluster_labels = np.zeros_like(clusters_numpy)
for i, j in zip(row_ind, col_ind):
    new_cluster_labels[clusters_numpy == j] = i


# In[332]:


# calculate confusion matrix
cm = confusion_matrix(y, new_cluster_labels)

# calculate Precision、Recall and F1-score
precision = precision_score(y, new_cluster_labels, average='weighted')  
recall = recall_score(y, new_cluster_labels, average='weighted')
f1 = f1_score(y, new_cluster_labels, average='weighted')

# print the result
print(f'Confusion Matrix:\n{cm}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')


# In[144]:


import seaborn as sns


# In[322]:


# visualiza the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=np.unique(clusters), yticklabels=np.unique(clusters))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# In[116]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




