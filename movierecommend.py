#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies.head()


# In[4]:


credits.head()


# In[5]:


credits.head(1)['cast'].values


# In[6]:


movies.shape


# In[7]:


credits.shape


# In[8]:


movies = movies.merge(credits,on='title')


# movies.head(

# In[9]:


movies.head(1)


# In[10]:


movies['original_language'].value_counts()


# In[11]:


#genres
#id
#keywords
#title
#overview
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[12]:


movies.head()


# In[13]:


movies.isnull().sum()


# In[14]:


movies.dropna(inplace=True)


# In[15]:


movies.duplicated().sum()


# In[16]:


movies.iloc[0].genres


# In[17]:


movies.iloc[1].genres


# In[18]:


#'[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
#['Action','Adventure','Fantasy','SciFi']


# In[19]:


import ast 
def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L    
    


# In[20]:


movies['genres'] = movies['genres'].apply(convert)


# In[21]:


movies.head()


# In[22]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[23]:


movies.head()


# In[24]:


movies['cast'][0]


# In[25]:


import ast 
def convert3(obj):
    L=[]
    counter = 0
    for i in ast.literal_eval(obj):
        if counter !=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L    
    


# In[26]:


movies['cast'] = movies['cast'].apply(convert3)


# In[27]:


movies.head()


# In[28]:


movies['crew'][0]


# In[29]:


import ast 
def fetch_director(obj):
    L=[]
    
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L    
    


# In[30]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[31]:


movies['crew'][0]


# In[32]:


movies.head()


# In[33]:


movies['overview'][0]


# In[34]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[35]:


movies.head()


# In[36]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[37]:


movies.head()


# In[38]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[39]:


movies.head()


# In[40]:


movies['tags'][0]


# In[41]:


new_df = movies[['movie_id' , 'title' ,'tags']]


# In[42]:


new_df.head()


# In[43]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[44]:


new_df.head()


# In[45]:


new_df['tags'][0]


# In[46]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[47]:


new_df.head()


# In[48]:


new_df['tags'][0]


# In[49]:


new_df['tags'][1]


# In[50]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[51]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[52]:


cv.fit_transform(new_df['tags']).toarray().shape


# In[53]:


vectors


# In[54]:


vectors[0]


# In[55]:


cv.get_feature_names_out()


# In[56]:


get_ipython().system('pip install nltk')


# In[57]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[58]:


def stem(text):
    y=[]
    
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)    


# In[59]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[60]:


new_df.head()


# In[61]:


from sklearn.metrics.pairwise import cosine_similarity


# In[62]:


cosine_similarity(vectors)


# In[63]:


cosine_similarity(vectors).shape


# In[64]:


similarity = cosine_similarity(vectors)


# In[65]:


sorted(list(enumerate(similarity[0])),reverse = True,key = lambda x:x[1])[1:6]


# In[66]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse = True,key = lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
    return


# In[67]:


recommend('Batman Begins')


# In[68]:


import pickle


# In[72]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))


# In[73]:


new_df['title'].values


# In[71]:


new_df.to_dict()


# In[74]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




