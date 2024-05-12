import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import math
import nltk
import streamlit as st
import networkx as nx
import compress_pickle as pkl
from transformers import pipeline
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define the title text
title_text = "<h1 style='text-align: center; color: white;'>Text Summarization</h1>"

# Define the CSS for the gradient background
gradient_bg_css = """
    background: linear-gradient(to right, #4C0FB5, #198DD0); 
    padding: 20px; 
    border-radius: 10px; 
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
        text1 = text  # Update text1 with the uploaded text

# Function to handle actions when "Enter Manually" button is clicked
def enter_manually():
    global text1  # Use the global variable text1
    text = st.text_area("Enter Text Manually", height=200)
    text1 = text  # Update text1 with the manually entered text
    #st.button("Submit", key="submit_button_manual")  # Fixed key for manual submission
    # Further processing code here if needed

# Main part of the app
st.title("Data Input")
st.write("")

# Button to select data input method
input_method = st.radio("Select a method to input data: " ,("Enter Manually", "Upload Data"))

# Show appropriate input widgets based on the selected input method
if input_method == "Upload data":
    st.subheader("Upload Data")
    upload_file()

elif input_method == "Enter Manually":
    st.subheader("Enter Data Manually")
    enter_manually()

st.write("")
st.write("")

preproc = st.radio("Choose Between Stemming and Lemmatization: ", ("Stemming", "Lemmatization"))

st.write("")
st.write("")

user_input = st.number_input("Enter Threshold for TF-IDF Summary:", value=1.0)
text  = text1 

def create_frequency_table(text_string, preproc) -> dict:
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text_string)

    if preproc == "Stemming":
        ps = PorterStemmer()
        proc = ps.stem
    elif preproc == "Lemmatization":
        wnl = WordNetLemmatizer()
        proc = wnl.lemmatize

    freqTable = dict()
    for word in words:
        word = proc(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    return freqTable


def create_frequency_matrix(sentences, preproc):
    frequency_matrix = {}
    stopWords = set(stopwords.words("english"))

    if preproc == "Stemming":
        ps = PorterStemmer()
        proc = ps.stem
    elif preproc == "Lemmatization":
        wnl = WordNetLemmatizer()
        proc = wnl.lemmatize

    for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = proc(word)
            if word in stopWords:
                continue

            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        frequency_matrix[sent[:15]] = freq_table

    return frequency_matrix


def create_tf_matrix(freq_matrix):
    tf_matrix = {}

    for sent, f_table in freq_matrix.items():
        tf_table = {}

        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sentence

        tf_matrix[sent] = tf_table

    return tf_matrix


def create_documents_per_words(freq_matrix):
    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1

    return word_per_doc_table


def create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix


def create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()):  # here, keys are the same in both the table
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix


def score_sentences(tf_idf_matrix) -> dict:
    sentenceValue = {}

    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence

    return sentenceValue


def find_average_score(sentenceValue) -> int:
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Average value of a sentence from original summary_text
    average = (sumValues / len(sentenceValue))

    return average


def generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary

def create_word_embeddings(file):
    
    '''
    word_embeddings = {}
    with open('glove.6B.50d.txt') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            word_embeddings[word] = coefs
    return word_embeddings
    '''

    word_embeddings = pkl.load(file)
    return word_embeddings

def create_sim_mat(word_embeddings):
    sentence_vectors = []
    for i in sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((50,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((50,))
        sentence_vectors.append(v)
    sim_mat = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,50), sentence_vectors[j].reshape(1,50))[0,0]
    return sim_mat

def rank_sentences(sim_mat):
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    return ranked_sentences

# TF-IDF
sentences = sent_tokenize(text)
total_documents = len(sentences)
freq_matrix = create_frequency_matrix(sentences, preproc)
tf_matrix = create_tf_matrix(freq_matrix)
count_doc_per_words = create_documents_per_words(freq_matrix)
idf_matrix = create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
tf_idf_matrix = create_tf_idf_matrix(tf_matrix, idf_matrix)
sentence_scores = score_sentences(tf_idf_matrix)
threshold = find_average_score(sentence_scores)
summary_tf_idf = generate_summary(sentences, sentence_scores, user_input * threshold)

# Word Embeddings
word_embeddings = create_word_embeddings("glove.6B.50d.gz")
sim_mat = create_sim_mat(word_embeddings)
ranked_sentences = rank_sentences(sim_mat)

st.write("")
st.write("")
sn = int(st.number_input("Enter Number of Sentences for GloVe Word Embedding: ", min_value=0, value=1))

min = int(st.number_input("Enter Minimum Length for Transformer Summary: ", min_value=0, value=min([len(x) for x in sentences])))
max = int(st.number_input("Enter Maxmimum Length for Transformer Summary: ", min_value=0, value=int(max([len(x) for x in sentences])/2)))


# Define the subheader text
subheader_text = "TF-IDF Approach Summary:"
subheader_text2 = "GloVe Word Embedding Approach Summary:"
subheader_text3 = "Transformer Summary:"

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
styled_subheader3 = f"<div style='{gradient_bg_css}'>{subheader_text3}</div>"


# Render the subheader
st.write("")
st.write("")
st.markdown(styled_subheader, unsafe_allow_html=True)
st.write("")
st.write(summary_tf_idf)
st.write("")
st.write("")

# Generate summary
st.markdown(styled_subheader2, unsafe_allow_html=True)
st.write("")
summary_glove = ""
# Generate summary
for i in range(sn):
    summary_glove += (ranked_sentences[i][1] + " ")
    
st.write(summary_glove)
st.write("")
st.write("")

# Generate summary
st.markdown(styled_subheader3, unsafe_allow_html=True)
st.write("")

summarizer = pipeline("summarization", model="google-t5/t5-small")
summary = summarizer(text, max_length=max, min_length=min, do_sample=False)
print(summary)
st.write(summary[0]['summary_text'])