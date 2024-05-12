import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import nltk
import re
import math
import operator
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag                   # ---------> POS tagging
from nltk.corpus import stopwords , state_union
from nltk.tokenize import PunktSentenceTokenizer
import spacy
from spacy import displacy                 # ---------> for Visualization
from string import punctuation
from string import punctuation
from IPython.display import HTML
from scipy.stats import norm
# ---------> for highlighting words
# Downloading Files
#nltk.download('stopwords')
#nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nlp = spacy.load('en_core_web_sm')
wordlemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
nltk.download('stopwords')
nltk.download('punkt')
Stopwords = set(stopwords.words('english'))
import streamlit as st
import uuid

# Define the title text
title_text = "<h1 style='text-align: center; color: white;'>Text Summarization Web App</h1>"

# Define the CSS for the gradient background
gradient_bg_css = """
    background: linear-gradient(to right, #4C0FB5, #198DD0); 
    padding: 20px; 
    border-radius: 10px; 
    border: 4px solid white; /* Adding a 2px solid white border */
"""

# Combine title text with gradient background CSS
styled_title = f"<div style='{gradient_bg_css}'>{title_text}</div>"

# Render the title
st.write("")
st.markdown(styled_title, unsafe_allow_html=True)
st.write("")
st.write("")

# Initialize text1 variable
text1 = ""

# Function to handle actions when "Upload .txt/.pdf file" button is clicked
def upload_file():
    global text1  # Use the global variable text1
    uploaded_file = st.file_uploader("Upload .txt/.pdf file", type=['txt', 'pdf'])
    if uploaded_file is not None:
        # Process the uploaded file
        text = uploaded_file.read().decode('utf-8')  # Read the contents of the file
        st.text_area("Text from file", value=text, height=200)
        #st.button("Submit", key=str(uuid.uuid4()))  # Unique key for each button
        text1 = text  # Update text1 with the uploaded text

# Function to handle actions when "Enter Manually" button is clicked
def enter_manually():
    global text1  # Use the global variable text1
    text = st.text_area("Enter Text Manually", height=200)
    text1 = text  # Update text1 with the manually entered text
    #st.button("Submit", key="submit_button_manual")  # Fixed key for manual submission
    # Further processing code here if needed

# Main part of the app
st.title("Data Input Options")
st.write("")

# Button to select data input method
input_method = st.radio("Select Data Input Method", ("Upload data", "Enter Manually"))

# Show appropriate input widgets based on the selected input method
if input_method == "Upload data":
    st.subheader("Upload Data")
    upload_file()

elif input_method == "Enter Manually":
    st.subheader("Enter Data Manually")
    enter_manually()

# Display the text1 variable after input
#st.write("Text1:", text1)
st.write("")
st.write("")
user_input = st.number_input("Enter Percentage for TF-IDF Summary:")
text  = text1 
# Tokenization using NLTK library
words = word_tokenize(text)
vis = words
#st.write(vis[:10])
# Sentence Tokenization
i = 1
sentences = sent_tokenize(text)
def lowercasing(word):
        word = word.lower()
for word in words:
    lowercasing(word)
vis = words

stopWords = list(stopwords.words("english"))+list(punctuation)+list([0,1,2,3,4,5,6,7,8,9])

without_stop = []
def stopword_removal(word):
    if word not in stopWords:
        without_stop.append(word)
for word in words:
    stopword_removal(word)
words = without_stop
vis = words
vis2 = list(set(vis))
#st.write(vis[:2])
stemmed_words = []
for word in words:
    stemmed = stemmer.stem(word)
    stemmed_words.append(stemmed)
words = stemmed_words
#words[:10]
lemmatized_words = []
for word in words:
    lemmatized_words.append(wordlemmatizer.lemmatize(word))
words = lemmatized_words
#words[:10]
def remove_special_characters(text):
    regex = r'[^a-zA-Z0-9\s]'
    text = re.sub(regex,'',text)
    return text
def freq(words):
    words = [word.lower() for word in words]
    freqTable = {}
    words_unique = []
    for word in words:
        if word not in words_unique:
            words_unique.append(word)
    for word in words_unique:
        freqTable[word] = words.count(word)
    return freqTable


def idf_score(no_of_sentences,word,sentences):
    no_of_sentence_containing_word = 0
    for sentence in sentences:
        sentence = remove_special_characters(str(sentence))
        sentence = re.sub(r'\d+', '', sentence)
        sentence = sentence.split()
        sentence = [word for word in sentence if word.lower() not in Stopwords and len(word)>1]
        sentence = [word.lower() for word in sentence]
        sentence = [wordlemmatizer.lemmatize(word) for word in sentence]
        if word in sentence:
            no_of_sentence_containing_word = no_of_sentence_containing_word + 1
    idf = math.log10(no_of_sentences/no_of_sentence_containing_word)
    return idf

def tf_score(word,sentence):
    freq_sum = 0
    word_frequency_in_sentence = 0
    len_sentence = len(sentence)
    for word_in_sentence in sentence.split():
        if word == word_in_sentence:
            word_frequency_in_sentence = word_frequency_in_sentence + 1
    tf =  word_frequency_in_sentence/ len_sentence
    return tf


def pos_tagging(text):
    pos_tag = nltk.pos_tag(text.split())
    pos_tagged_noun_verb = []
    for word,tag in pos_tag:
        if tag == "NN" or tag == "NNP" or tag == "NNS" or tag == "VB" or tag == "VBD" or tag == "VBG" or tag == "VBN" or tag == "VBP" or tag == "VBZ":
            pos_tagged_noun_verb.append(word)
    return pos_tagged_noun_verb



def tf_idf_score(tf,idf):
    return tf*idf

def word_tfidf(freqTable,word,sentences,sentence):
    word_tfidf = []
    tf = tf_score(word,sentence)
    idf = idf_score(len(sentences),word,sentences)
    tf_idf = tf_idf_score(tf,idf)
    return tf_idf

def sentence_importance(sentence,freqTable,sentences):
    sentence_score = 0
    sentence = remove_special_characters(str(sentence))
    sentence = re.sub(r'\d+', '', sentence)
    pos_tagged_sentence = []
    no_of_sentences = len(sentences)
    pos_tagged_sentence = pos_tagging(sentence)
    for word in pos_tagged_sentence:
        if word.lower() not in Stopwords and word not in Stopwords and len(word)>1:
            word = word.lower()
            word = wordlemmatizer.lemmatize(word)
            sentence_score = sentence_score + word_tfidf(freqTable,word,sentences,sentence)
    return sentence_score
freqTable={}
for word in words:
    word = word.lower()
    if word not in stopWords:
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1
word_freq = freqTable
#st.write(len(sentences))
# Take text input from the user
# Display the input
#st.write("You entered:", user_input)


import opendatasets as od
od.download(r'https://www.kaggle.com/datasets/sawarn69/glove6b100dtxt?select=glove.6B.100d.txt')
from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = r"C:\Users\acer\Downloads\glove6b100dtxt\glove.6B.100d.txt"
#glove_input_file = r"/content/glove6b100dtxt/glove.6B.100d.txt"
# Extract word vectors
word_embeddings = {}
f = open(glove_input_file, encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()
sentence_vectors = []
for i in sentences:
    if len(i) != 0:
        v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
    else:
        v = np.zeros((100,))
    sentence_vectors.append(v)

# similarity matrix
sim_mat = np.zeros([len(sentences), len(sentences)])
#sim_mat.shape
from sklearn.metrics.pairwise import cosine_similarity
i=0
j=1
#cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0][0]




for i in range(len(sentences)):
    for j in range(len(sentences)):
        if i != j:
            sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]

#print(sim_mat.shape)

import networkx as nx
nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)
ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)




input_user =user_input #int(input('Percentage of information to retain(in percent):'))
no_of_sentences = int((input_user * len(sentences))/100)
#st.write(no_of_sentences)
#*******************************
c = 1
sentence_with_importance = {}
for sent in sentences:
    sentenceimp = sentence_importance(sent,word_freq,sentences)
    sentence_with_importance[c] = sentenceimp
    c = c+1
sentence_with_importance = sorted(sentence_with_importance.items(), key=operator.itemgetter(1),reverse=True)
cnt = 0
summary = []
sentence_no = []
for word_prob in sentence_with_importance:
    if cnt < no_of_sentences:
        sentence_no.append(word_prob[0])
        cnt = cnt+1
    else:
        break
sentence_no.sort()
cnt = 1
for sentence in sentences:
    if cnt in sentence_no:
        summary.append(sentence)
        summary.append('\n')
    cnt = cnt+1

summary = " ".join(summary)
#print("\n")
st.write("")
st.write("")
sn = int(st.number_input("Please Enter Number of Sentences for Word Embedding Summary: "))
# Define the subheader text
subheader_text = "TF-IDF Approach Summary:"
subheader_text2 = "Word Embedding Approach Summary:"

# Define the CSS for the gradient background
gradient_bg_css = """
    background: linear-gradient(to right, #4C0FB5, #198DD0); 
    padding: 10px; 
    border-radius: 10px; 
    color: white;
"""

# Combine subheader text with gradient background CSS
styled_subheader = f"<div style='{gradient_bg_css}'>{subheader_text}</div>"
styled_subheader2 = f"<div style='{gradient_bg_css}'>{subheader_text2}</div>"

# Render the subheader
st.markdown(styled_subheader, unsafe_allow_html=True)
st.write("")
st.write("")
st.write(summary)

# Specify number of sentences to form the summary
st.write("")
st.write("")
st.write("")
st.write("")
# Generate summary
st.markdown(styled_subheader2, unsafe_allow_html=True)
result = ""
# Generate summary
for i in range(sn):
    result += (ranked_sentences[i][1] + " ")
    
st.write(result)
