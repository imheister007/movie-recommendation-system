#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')


# In[4]:


movies.head(1)


# In[5]:


credits.head(1)


# ## Merging Data
# on the basis on title

# In[6]:


movies=movies.merge(credits,on='title')


# In[7]:


movies.head(1)


# ## Selecting Columns

# In[8]:


movies.info()


# ## selected columns are:  
# genres	
# id   
# keywords  
# title  
# overview   
# cast    
# crew
# 
#   
# 
# 
# 

# In[9]:


movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[10]:


movies.head()


# In[11]:


movies.isnull().sum()


# In[12]:


movies.dropna(inplace=True)


# In[13]:


movies.duplicated().sum()


# ## pre-processing the columns

# In[14]:


#movies.iloc[0].genres


# In[15]:


#lis of dictionary
# i want to be in from--> ['Action','Adventur','Fantasy'...]
#import ast


# def convert(obj):
#     L=[]
#     for i in ast.literal_eval(obj): # traversing in list
#         L.append(i['name'])
#         
#         return L
#         

# In[16]:


#convert('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[17]:


# here we have this format in string
#change to list


# In[18]:


#import ast 
#ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[19]:


movies.head()


# In[20]:


import ast 
def convert(obj):
    L=[]
    for i in ast.literal_eval(obj): # traversing in list
        L.append(i['name'])
    return L


# In[21]:


movies.dropna(inplace=True)


# In[22]:


movies['genres']=movies['genres'].apply(convert)


# In[23]:


movies.head()


# In[24]:


movies.iloc[0].keywords # we will also extract name from this


# In[25]:


movies['keywords']=movies['keywords'].apply(convert)


# In[26]:


movies.head()


# In[27]:


#movies.iloc[0].cast 
#uncomment to see result


# In[28]:


#I will take only first three cast name for each record


# In[29]:


import ast 
def convert3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj): # traversing in list
        if(counter!=3):
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[30]:


movies['cast']=movies['cast'].apply(convert3)


# In[31]:


movies.head()


# In[32]:


#movies.iloc[0].crew
#uncomment to see result


# In[33]:


# i want to extract all name whose job is director


# In[34]:


import ast 
def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj): # traversing in list
        if(i['job']=="Director"):
            L.append(i['name'])
            break
    return L


# In[35]:


movies['crew']=movies['crew'].apply(fetch_director)


# In[36]:


movies.head()


# ## Exploring overview

# In[37]:


movies.iloc[0].overview # will convert in list


# In[38]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[39]:


movies.head()


# I will remove space between name of every person to get as tag  
# for ex-> Sam Wirthington == SamWorthington

# In[40]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[41]:


movies.head()


# ## we will make a column call Tags and merge overvew,genres,keyword,cast,crew into this

# In[42]:


movies['tags']=movies['overview'] +movies['genres']+movies['keywords']+movies['cast']+ movies['crew']


# In[43]:


movies.head()


# In[44]:


new_df=movies[['movie_id','title','tags']]


# In[45]:


new_df


# In[46]:


new_df['tags']=new_df['tags'].apply(lambda x: " ".join(x) )
#converting all into a string


# In[47]:


new_df.head()


# In[48]:


new_df['tags'][0]


# In[49]:


#convert all into lower case


# In[50]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[51]:


new_df.head()


# # Text Vectorization (Bag of words)

#  Will Convert each tags into vector=> each movie will be a vector. now if we choose any vector and say suggest me new movie. then i will pick the nearest movie (i.e nearest vector of my selected vector)

# In Bag of words, we will combine all the word (tags) and find which word which occurs most

# In[52]:


#we will take 5000 most occur words
# we will not consider stop word like--> and,or,I,in,for....etc


# In[53]:


from sklearn.feature_extraction.text import CountVectorizer


# In[54]:


cv= CountVectorizer(max_features=5000,stop_words='english')

#now we will use cv object


# In[55]:


vectors=cv.fit_transform(new_df['tags']).toarray()
#CountVectorizer return matrix so we will convert to np araay


# In[56]:


vectors[0]


# In[57]:


#we can see those 500 words too
cv.get_feature_names()[0:10] # seeing top 10


# here we see many word appearing many time like action,action..  
# to remove this we use staming  
# ex-> say ['love','loving','lover'] is change int  
#     ['love','love','love']

# In[58]:


import nltk 


# In[59]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[60]:


#examle
print(ps.stem('love'))
print(ps.stem('loving'))


# In[61]:


def stem(text):
    y=[]
    
    for i in text.split():
        ps.stem(i)
        y.append(ps.stem(i))
    return " ".join(y)    
        


# In[62]:


new_df['tags'].apply(stem)


# In[63]:


new_df['tags']=new_df['tags'].apply(stem)


# In[64]:


#checking values are replaced or not
vectors=cv.fit_transform(new_df['tags']).toarray()
cv.get_feature_names()[0:7]  # showing only top 7


# We have vector of movies NOW

# We have to calculate the distance of each vector(movie). if far means not related=> not recommending and if near means similar movies

# we will not use eculidian distance, we will calculate cosine distance. it will tell angle between two vector   
# if angle==0 means both are same vector  
# ifangle=1 => points are near  
# if angle=30 => more distance between them
# if anglr=180 => both movies are opposite in nature

# As it is higher dimesion we use Cosine distane (aw we are here 5000 vector of 5000 dimension). so we use cosin distance

# # Cosine Distance

# In[65]:


from sklearn.metrics.pairwise import cosine_similarity


# In[66]:


cosine_similarity(vectors)


# In[67]:


similarity=cosine_similarity(vectors)


# In[68]:


similarity[0] # pehli movie ka sabhi movies se distance


# you can see pehli movie ka pehli movie se similarity 1..hona v chahiye dono movie same hai
# 

# Now i Have to make a function which takes a movie name and return 5 similar movie  
# 
# i will sort and take top 5 movie

# Important--> if someone gives me movie, i have to find the index of that movie. now ye index use krke mai similatiy me jakr us movies ka list nikaluga and uss array ko sort kr duga or near 5 jo movies hoge usko nikal luga

# but if i sort i will break the order that relation 1st movie to 1st or distance of first movie to 2nd...coz we r sorting n values are getting mixed in srting order 

# so we have to hold that order for this we use enumerates (get touple)

# 
# =====================================================================

# Index fetching code

# In[69]:


#for ex
new_df[new_df['title']=='Avatar']


# In[70]:


new_df[new_df['title']=='Avatar'].index


# In[71]:


new_df[new_df['title']=='Avatar'].index[0]


# =======================================================================

# In[72]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]
#we are getting ki avatar se similar movie ka nmbr h 1216...agla 2409..
#we apply this on function


# In[73]:


def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distances=similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
    
    
    


# In[74]:


recommend('Batman Begins')


# ## sending movie nme to website

# In[75]:


import pickle


# In[76]:


pickle.dump(new_df.to_dict(),open('movie_list.pkl','wb'))


# In[77]:


new_df['title'].values


# In[78]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




