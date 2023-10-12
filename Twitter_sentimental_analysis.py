import pandas as pd
import re
data = pd.read_csv("./twitter.csv")

data_1 = data.head(100)

### Data Preprocessing:
import re
import nltk
nltk.download('stopwords')

#Data preprocessing and cleaning:
def preprocess_text(text):
    
    #removing tags:
    removing_tags = re.compile(r'<[^>]+>')
    text = removing_tags.sub("", text)
    
    #removing speacial chatracters:
    removing_sc = re.compile(r'[^A-Za-z0-9\s]+')
    text = removing_sc.sub('', text)
    
    #removing url:
    removing_url = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    text = removing_url.sub('', text)
    
    #lowercase:
    text = text.lower()
    
    return text

#
data_1['preprocessed_tweet'] = data_1['tweet'].apply(preprocess_text)

#Lexicon Textblob:
#Addressing spelling mistakes and grammatical erros involves NLP tools. We shall use TextBlob for spelling correction & POS tag:

#!pip install textblob
#Correcting spelling and grammar:
from textblob import TextBlob
def correct_spelling_and_grammar(text):
    # Create a TextBlob object with the input text
    blob = TextBlob(text)
    
    # Correct spelling mistakes in the text
    corrected_text = blob.correct()
    
    # Perform part-of-speech tagging to identify and correct grammatical errors
    # This step helps in identifying and correcting grammar issues, but it may not catch all errors.
    #tagged_text = corrected_text.correct()
    
    return corrected_text

#
data_1['spelling'] = data_1['preprocessed_tweet'].apply(correct_spelling_and_grammar).apply(str)

#To tokenize sentences into words or subword units, you can use various Python libraries and tools using NLTK
#One popular library for this task is the Natural Language Toolkit (NLTK).
#!pip install nltk

#Tokenizing:
import nltk
#nltk.download('punkt')  # Download necessary resources

from nltk.tokenize import word_tokenize, sent_tokenize

def tokenize_sentences(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    #The sent_tokenize function from NLTK (Natural Language Toolkit) is used to split a given text into individual sentences. 
    #It recognizes sentence boundaries in the text based on punctuation and capitalization patterns.
    
    # Tokenize each sentence into words
    tokenized_sentences = [word_tokenize(i) for i in sentences]
    
    return tokenized_sentences

#
data_1['token_data'] = data_1['spelling'].apply(tokenize_sentences)

#Stopwords removal from nltk:
#To remove stop words from a text in Python, you can use libraries like NLTK (Natural Language Toolkit) or spaCy.

from nltk.corpus import stopwords
#nltk.download('stopwords')

def remove_stopwords(words):
    # Get a set of English stop words
    stop_words = set(stopwords.words("english"))
    
    # Remove stop words from the list of words
    filtered_words = [word for word in words[0] if word not in stop_words]
    
    return filtered_words

#
data_1['data_remove_stopwords'] = [remove_stopwords(i) for i in data_1['token_data']]

#Lemmatization and Stemming:
'''Lemmatization is a text normalization technique in natural language processing (NLP) that reduces words to their base or dictionary form, known as the "lemma." This process helps in reducing the dimensionality of text data and can improve the accuracy of text analysis tasks such as text classification, sentiment analysis, and information retrieval.'''

'''Stemming is a text normalization technique in natural language processing (NLP) that aims to reduce words to their word stems or root forms. The process involves removing prefixes and suffixes from words to obtain a common base form. The resulting stem may not always be a valid word, but it represents the core meaning of related words.'''

'''Using POS information improves the accuracy of lemmatization. It ensures that the lemmatizer selects the most appropriate base form for a word, taking into account its grammatical context. Part-of-speech (POS) tags are used in lemmatization to disambiguate word meanings and determine the correct lemma (base form) of a word based on its grammatical role within a sentence. '''

from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet

# Initialize the NLTK lemmatizer and stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Function to get WordNet POS tags for lemmatization
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# Function to lemmatize a single word
def lemmatize_word(word):
    return lemmatizer.lemmatize(word, get_wordnet_pos(word))

# Function to stem a single word
def stem_word(word):
    return stemmer.stem(word)

# Apply lemmatization and stemming functions to the DataFrame element-wise
data_1['lemmatized_data'] = data_1['data_remove_stopwords'].apply(lambda words: [lemmatize_word(word) for word in words])
data_1['stemmed_data'] = data_1['data_remove_stopwords'].apply(lambda words: [stem_word(word) for word in words])

#Dropping unwanted columns:
data_1.drop(['preprocessed_tweet', 'spelling', 'token_data', 'data_remove_stopwords'], axis = 1, inplace =True)
print(data_1.keys())








