#!/usr/bin/env python
# coding: utf-8

# In[4]:


import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')


# In[2]:


doc = nlp(u'the sentence is written to test the spacy in NLP from paris!')


# In[21]:


for sen in doc:
    print(sen.text,'\t',sen.pos_,'\t',sen.dep_)


# In[4]:


print(doc[3].text)
print(doc[3].lemma_)


# In[5]:


print(doc[3].lemma_+ '----' +spacy.explain(doc[3].tag_))


# In[6]:


print(doc[9].text +'......'+doc[9].shape_)
print(doc[10].text +'.....'+doc[10].shape_)


# In[7]:


doc1 = nlp(u'Although commmonly attributed to John Lennon from his song "Beautiful Boy", the phrase "Life is what happens to us while we are making other plans" was written by cartoonist Allen Saunders and published in Reader\'s Digest in 1957, when Lennon was 17.')

type(doc1[5:15])


# In[8]:


doc2 = nlp(u'I love football. Manchester united is my favourite. Love the game')

print(doc2)
for text in doc2.sents:
    print(text)
len(doc2)


# In[9]:


doc2 = nlp(u'I love football. Manchester united is my favourite. Love the game 100%')
for text in doc2.ents:
    print(text.text+ '...' + text.label_ + '...'+spacy.explain(text.label_ ))


# In[10]:


doc2 = nlp(u'I love football. Manchester united is my favourite. Love the game 100%')
for text in doc2.noun_chunks:
    print(text)


# In[11]:


#for display (doc,style,jupyter)
doc2 = nlp(u'I love football. Manchester united is my favourite. Love the game 100%')
displacy.render(doc2,style='ent', jupyter = True)

#displacy.serve(doc2,style='dep')
#displacy.serve(doc2,style='ent')
  


# In[12]:


from nltk.stem.snowball import SnowballStemmer


# In[13]:


stemmer = SnowballStemmer('english')


# In[20]:


words = ['swimming','playing','looking']
for verb in words:
    print(verb + '...'+ stemmer.stem(verb))


# In[19]:


words = 'swimming and playing is good for health'
for verb in words.split():
    print(verb + '...'+ stemmer.stem(verb))


# In[35]:


def lemma(text):
    for word in text:
        print(f'{word.text:{15}} {word.pos_:{10}} {word.lemma_}')
    
sentence = nlp(u'The game of football is very good for playing!')
lemma(sentence)


# In[40]:


print(nlp.vocab['mystery'].is_stop)
print(nlp.vocab['call'].is_stop)


# In[45]:


nlp.Defaults.stop_words.add('mystery')
nlp.vocab['mystery'].is_stop = True
print(nlp.vocab['mystery'].is_stop)

nlp.Defaults.stop_words.remove('call')
nlp.vocab['call'].is_stop = False
print(nlp.vocab['call'].is_stop)


# In[122]:


from spacy.matcher import Matcher
match = Matcher(nlp.vocab)


# In[123]:


match1 = [{'LOWER': 'football'}]
match2 = [{'LOWER': 'foot'}, {'LOWER': 'ball'}]
match3 = [{'LOWER': 'foot'}, {'IS_PUNCT': True}, {'LOWER': 'ball'}]



match.add('FootBall', None, match1, match2, match3)


doc = nlp(u'Football is beautiful game playing foot-ball will be nice,manchester united is a football club.')

found_matches = match(doc)
print(found_matches)


# In[16]:


from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(nlp.vocab)


# In[ ]:





# In[3]:


pwd


# In[17]:


with open('C:\\Users\\sanja\\OneDrive\\Documents\\reaganomics.txt') as f:
    doc = nlp(f.read())


# In[20]:


phrase_list = ['voodoo economics', 'supply-side economics', 'trickle-down economics', 'free-market economics']
phrasematch  = [nlp(text) for text in phrase_list]
matcher.add("economics", None, *phrasematch)
match = matcher(doc)
match
len(doc)


# In[21]:


doc[:70]


# In[2]:


import numpy as np
import pandas as pd


# In[3]:


df = pd.read_csv('C:\\Users\\sanja\\OneDrive\\Documents\\moviereviews2.tsv',sep = '\t') 


# In[9]:


print(df.head())
print(len(df))


# In[6]:


df.isnull().sum()


# In[17]:


df.dropna(inplace=True)

print(len(df))


# In[15]:


blanks = []
for i,label,review in df.itertuples():
    if type(review)==str:
        if review.isspace():
             blanks.append(i)
               
print(len(blanks))


# In[18]:


df.dropna(inplace = True)


# In[21]:


df['label'].value_counts()


# In[22]:


from sklearn.model_selection import train_test_split 


# In[25]:


X = df['review']
y = df['label']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.33,random_state =42)


# In[29]:


print(X_train.shape)
print(X_test.shape)


# In[34]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


# In[36]:


movie = Pipeline([('vector', TfidfVectorizer()),('model', LinearSVC())] )
movie.fit(X_train,y_train)


# In[37]:


prediction = movie.predict(X_test)


# In[39]:


from sklearn import metrics
print(metrics.confusion_matrix(y_test,prediction))


# In[41]:


metrics.accuracy_score(y_test,prediction)


# In[42]:


prediction = movie.predict(['This is the movie of the year for sure'])
prediction


# In[47]:


prediction = movie.predict(['The movie is bad'])
print(prediction)


# In[ ]:




