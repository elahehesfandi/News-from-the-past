#!/usr/bin/env python
# coding: utf-8

# ![download.png](attachment:download.png)
#                                    "Text mining and sentiment analysis" project report, 2022-2023.  

# # Topic: News from the past (P1)
#  
# Writer: Elaheh Esfandi

# ### step 0:installing the packages we need

# In[6]:


get_ipython().system('pip install mysql-connector')
get_ipython().system('pip install mysql-connector-python')
get_ipython().system('pip install treetaggerwrapper')
get_ipython().system('pip install spacy')
get_ipython().system('python -m spacy download it_core_news_sm')


# In[7]:


get_ipython().system('pip install treetaggerwrapper')


# ### step 1: importing datasets and the libraries

# In[2]:


from tqdm import tqdm
import pandas as pd
import nltk
from nltk.corpus import gutenberg
from nltk import word_tokenize
from nltk import trigrams, ngrams


# In[3]:


import string
myPunctation = string.punctuation +'“'+'”'+'-'+'’'+'‘'+'—'
myPunctation = myPunctation.replace('.', '')
myPunctation


# In[4]:


from collections import defaultdict
import random
sentence_list = []


# In[5]:


import mysql.connector
from mysql.connector import (connection)


# In[6]:


mydb = connection.MySQLConnection(user='root', password='12345',
                              host='127.0.0.1',
                              database='mozart')


# In[7]:


mycursor = mydb.cursor()
mycursor.execute("Show tables;")
mo = mycursor.fetchall()
for x in mo:
    print(x)


# In[87]:


mycursor = mydb.cursor()
mycursor.execute("SELECT * FROM mozart.mxx_lettere_lang")
mo = mycursor.fetchall()
#for row in mo:
    #print(row)
    #print("\n")


# In[9]:


mycursor = mydb.cursor()
mycursor.execute("SELECT testo_lettera_tag FROM mozart.mxx_lettere_lang")
lettera_tag = mycursor.fetchall()
#loop through the rows
for row in lettera_tag:
    print(row)
    print("\n")


# ### 1.2 Normalization_NLP_Basics

# In[10]:


df = pd.DataFrame (lettera_tag, columns = ['lettera'])


# In[11]:


def clean_text(text):
#remove punctuation   
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) 
#remove numbers
    text = re.sub(r'\w*\d\w*', '', text)
# remove 'text; and 'indices'
    text=re.sub(r'text', '', text)
    text=re.sub(r'indices', '', text)
# remove website
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'JW', '', text)
# remove the emoji 
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    text = regrex_pattern.sub(r'', text)
    return text


# In[12]:


def clean_text(text):
#remove punctuation   
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) 
#remove numbers
    text = re.sub(r'\w*\d\w*', '', text)
    return text


# In[13]:


import re
import string
df1 = pd.DataFrame(df.lettera.apply(lambda x: clean_text(x)))
df1


# ### 1.3 Lemmatizing Italian sentences 

# In[14]:


import pandas as pd
import nltk
nltk.download('punkt')
df['tokenized_sents'] = df1.apply(lambda row: nltk.word_tokenize(row['lettera']), axis=1)


# In[15]:


df['tokenized_sents'] 


# In[16]:


# Using filter() method to filter nan values
df =df['tokenized_sents'].dropna()


# In[17]:


df3=df.tolist()
from dataclasses import replace
# Using filter() method to filter None values
data = list(filter(None, df3))
data


# In[19]:


d = ''.join(' '.join(l) for l in data)
d


# ### 1.4 Extracting locations, time and name from text 

# Event dection in text by Named Entity Recognition (NER) 

# Named Entity Recognition is the process of NLP which deals with identifying and classifying named entities. The raw and structured text is taken and named entities are classified into persons, organizations, places, money, time, etc. Basically, named entities are identified and segmented into various predefined classes.
# 
# NER systems are developed with various linguistic approaches, as well as statistical and machine learning methods. NER has many applications for project or business purposes.

# In[22]:


import spacy
from spacy import displacy
get_ipython().system('python -m spacy download it')
import it_core_news_sm
nlp=it_core_news_sm.load()


# In[80]:


ruler = nlp.add_pipe("entity_ruler")
patterns = [{"label": "ORG", "pattern": "MyCorp Inc."}]
ruler.add_patterns(patterns)
doc = nlp(d)
a=[(ent.text, ent.label_) for ent in doc.ents]
a


# In[86]:


displacy.render(text1,style="ent",jupyter=True)


# In[82]:


df2 = pd.DataFrame (a, columns = ["text", "label"])
df2


# In[84]:


# selecting rows based on condition 
options = ['LOC','PER']   
G = df2.loc[df2['label'].isin(options)] 
G


# ### 1.5 Sentiment Analysis

# In[ ]:





# In[ ]:





# In[ ]:




