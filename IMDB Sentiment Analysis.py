#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib import style
style.use('ggplot')
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# # Loading and Understanding Dataset

# In[2]:


df= pd.read_csv("C:/Users/Reena Sawant/Desktop/Master Project/ML Projects/IMDB Dataset.csv/IMDB Dataset.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df.describe()


# # Understanding the dataset

# In[8]:


sns.countplot(x='sentiment', data=df)
plt.title("Sentiment distribution")


# In[9]:


for i in range(5):
    print("Review: ", [i])
    print(df['review'].iloc[i], "\n")
    print("Sentiment: ", df['sentiment'].iloc[i], "\n\n")


# In[10]:


def no_of_words(text):
    words= text.split()
    word_count = len(words)
    return word_count


# In[11]:


no_of_words('My name is njhiskd') #to understand how the function is working


# In[12]:


df['word count'] = df['review'].apply(no_of_words)


# In[13]:


df.head()


# # Subplots for sentiments with the word counts in them

# In[14]:


fig, ax = plt.subplots(1,2, figsize=(10,6))
ax[0].hist(df[df['sentiment'] == 'positive']['word count'], label='Positive', color='blue', rwidth=0.9);
ax[0].legend(loc='upper right');
ax[1].hist(df[df['sentiment'] == 'negative']['word count'], label='Negative', color='red', rwidth=0.9);
ax[1].legend(loc='upper right');
fig.suptitle("Number of words in review")
plt.show()


# # Length of the words in the review

# In[15]:


fig, ax = plt.subplots(1,2, figsize=(10,6))
ax[0].hist(df[df['sentiment'] == 'positive']['review'].str.len(), label='Positive', color='blue', rwidth=0.9);
ax[0].legend(loc='upper right');
ax[1].hist(df[df['sentiment'] == 'negative']['review'].str.len(), label='Negative', color='red', rwidth=0.9);
ax[1].legend(loc='upper right');
fig.suptitle("Length of words in review")
plt.show()


# In[16]:


df.sentiment.replace("positive", 1, inplace=True)
df.sentiment.replace("negative", 2, inplace=True)


# In[17]:


df.head()


# # Data Preprocessing

# In[18]:


def data_processing(text):
    text= text.lower()
    '''This line converts all the characters in the text string to lowercase using the lower() function. 
    This is a common preprocessing step because it helps to standardize the text and avoid treating words 
    in different cases as separate words.'''
    
    text = re.sub('<br />', '', text)
    '''This line uses the re.sub() function from Python's regular expression module (re) to replace all 
    occurrences of the '<br />' string with an empty string. This is because '<br />' is an HTML tag that 
    is often used to indicate line breaks in text, but it has no meaning in regular text and can be safely removed.'''
    
    text = re.sub(r"https\S+|www\S+|http\S+", '', text, flags = re.MULTILINE)
    '''This line uses the re.sub() function again to remove all URLs (web addresses) from the text. The regular expression 
    pattern r"https\S+|www\S+|http\S+" matches any URLs that begin with "http", "https", or "www", and the flags parameter 
    is set to re.MULTILINE to allow the pattern to match URLs that span multiple lines.'''
    
    text = re.sub(r'\@w+|\#', '', text)
    
    text = re.sub(r'[^\w\s]', '', text)
    
    text_tokens = word_tokenize(text)
    '''This line uses the word_tokenize() function from the NLTK library to tokenize the processed text into a list of words. 
    Tokenization is the process of breaking up a text into individual words or tokens.'''
    
    filtered_text = [w for w in text_tokens if not w in stop_words]
    '''This line creates a new list called filtered_text that contains only the words in the text_tokens list that are not 
    stopwords. Stopwords are common words that are often removed from text because they are not considered to be meaningful 
    for text analysis, such as "the", "and", and "a". The stop_words variable is assumed to be defined elsewhere in the code, and it should be a set or list of stopwords to be removed from the text.'''
    
    return " ".join(filtered_text)
    '''Finally, this line joins the filtered_text list back into a single string, with each word separated by a space 
    character, and returns the processed text string.'''


# In[19]:


df.review = df['review'].apply(data_processing)


# In[20]:


duplicated_count = df.duplicated().sum()
print("Number of duplicate entries: ", duplicated_count)


# In[21]:


df = df.drop_duplicates('review')


# In[22]:


stemmer = PorterStemmer()
def stemming(data):
    text = [stemmer.stem(word) for word in data]
    return data


# In[23]:


df.review = df['review'].apply(lambda x: stemming(x))


# In[24]:


df['word count'] = df['review'].apply(no_of_words)
df.head()


# In[25]:


pos_reviews =  df[df.sentiment == 1]
pos_reviews.head()


# In[26]:


text = ' '.join([word for word in pos_reviews['review']])
'''Here, the pos_reviews is a dataframe or a collection of positive reviews. This line joins all the reviews in the pos_reviews 
dataframe into a single long string with spaces between each word. This string is stored in the variable text.'''

plt.figure(figsize=(20,15), facecolor='None')
'''This line creates a new figure object with a width of 20 inches and height of 15 inches, and with a 
transparent background.'''

wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
'''This line generates a word cloud object using the WordCloud class from the wordcloud library. The max_words 
parameter specifies the maximum number of words to include in the word cloud (here, 500). The width and height parameters 
set the dimensions of the word cloud visualization. The generate() method creates the word cloud from the text variable.'''

plt.imshow(wordcloud, interpolation='bilinear')
'''This line displays the word cloud image using imshow() method from matplotlib library. The interpolation parameter 
specifies the type of interpolation to use when displaying the image.'''

plt.axis('off')

plt.title('Most frequent words in positive reviews', fontsize = 19)

plt.show()


# In[27]:


from collections import Counter
count = Counter()
for text in pos_reviews['review'].values:
    for word in text.split():
        count[word] +=1
count.most_common(15)


# In[28]:


pos_words = pd.DataFrame(count.most_common(15))
pos_words.columns = ['word', 'count']
pos_words.head()


# In[29]:


px.bar(pos_words, x='count', y='word', title='Common words in positive reviews', color = 'word')


# In[30]:


neg_reviews =  df[df.sentiment == 2]
neg_reviews.head()


# In[31]:


text = ' '.join([word for word in neg_reviews['review']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most frequent words in negative reviews', fontsize = 19)
plt.show()


# In[32]:


count = Counter()
for text in neg_reviews['review'].values:
    for word in text.split():
        count[word] +=1
count.most_common(15)


# In[33]:


neg_words = pd.DataFrame(count.most_common(15))
neg_words.columns = ['word', 'count']
neg_words.head()


# In[34]:


px.bar(neg_words, x='count', y='word', title='Common words in negative reviews', color = 'word')


# In[35]:


X = df['review']
Y = df['sentiment']


# In[36]:


vect = TfidfVectorizer()
X = vect.fit_transform(df['review'])


# In[37]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)


# In[39]:


print("Size of x_train: ", (x_train.shape))
print("Size of y_train: ", (y_train.shape))
print("Size of x_test: ", (x_test.shape))
print("Size of y_test: ", (y_test.shape))


# # Importing Machine Learning Libraries

# In[40]:


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


# In[41]:


logreg = LogisticRegression()
logreg.fit(x_train, y_train)
logreg_pred = logreg.predict(x_test)
logreg_acc = accuracy_score(logreg_pred, y_test)
print("Test accuracy: {:.2f}%".format(logreg_acc*100))


# In[42]:


print(confusion_matrix(y_test, logreg_pred))
print("\n")
print(classification_report(y_test, logreg_pred))


# In[43]:


mnb = MultinomialNB()
mnb.fit(x_train, y_train)
mnb_pred = mnb.predict(x_test)
mnb_acc = accuracy_score(mnb_pred, y_test)


# In[44]:



print(confusion_matrix(y_test, mnb_pred))
print("\n")
print(classification_report(y_test, mnb_pred))


# In[45]:



svc = LinearSVC()
svc.fit(x_train, y_train)
svc_pred = svc.predict(x_test)
svc_acc = accuracy_score(svc_pred, y_test)
print("Test accuracy: {:.2f}%".format(svc_acc*100))


# In[46]:



print(confusion_matrix(y_test, svc_pred))
print("\n")
print(classification_report(y_test, svc_pred))


# In[47]:


from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.1, 1, 10, 100], 'loss':['hinge', 'squared_hinge']}
grid = GridSearchCV(svc, param_grid, refit=True, verbose = 3)
grid.fit(x_train, y_train)


# In[48]:


print("best cross validation score: {:.2f}".format(grid.best_score_))
print("best parameters: ", grid.best_params_)


# In[49]:



svc = LinearSVC(C = 1, loss='hinge')
svc.fit(x_train, y_train)
svc_pred = svc.predict(x_test)
svc_acc = accuracy_score(svc_pred, y_test)
print("Test accuracy: {:.2f}%".format(svc_acc*100))


# In[50]:


print(confusion_matrix(y_test, svc_pred))
print("\n")
print(classification_report(y_test, svc_pred))


# In[ ]:




