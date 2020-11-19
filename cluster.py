# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 22:21:40 2020

@author: JIt Shil
"""


import collections
import nltk
from   nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text  import  TfidfVectorizer
from pprint import pprint


def tokenizer(text):
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    tokens= [stemmer.stem(t) for t in tokens if t not in stopwords.words('english')]
    return tokens


def  cluster_sentence(sentences, nb_of_clusters=2):
    tfidi_vecterizer= TfidfVectorizer(tokenizer = tokenizer, stop_words = stopwords.words('english'), lowercas =True)
    tfidf_matrix = tfidi_vecterizer.fit_transform(sentences)
    kmeans = KMeans(n_clusters = nb_of_clusters)
    kmeans.fit(tfidf_matrix)
    clusters = collections.defaultdict(list)
    for i , label in enumerate(kmeans.labels):
        clusters[label].append(i)
        return dict(clusters)
    
    
if __name__== "__main__" :
        sentences= ["Quantum physics needs for modern world",
                    "Machine & deep learning will make future world",
                    "90% news paper don't publish original news",
                    "Corona make horrible sitution day by day ",
                    "And now everything going to be unpredictable",
                    "I feel many more times existance of trisha",
                    "love u trisha"]
        nclusters = 3
        clusters = cluster_sentence(sentences, nclusters)
        for cluster in range(nclusters):
            print("cluster",cluster,":")
            for i,sentence in enumerate(clusters[cluster]):
                print("\sentence", i, ":", sentences[sentence])
                
        