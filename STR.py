# -*- coding: utf-8 -*-
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec,KeyedVectors
from nltk import word_tokenize
from pyemd import emd
import numpy as np
import codecs
import jieba
import math
import re
import time
from config import *

def LogInfo(stri):
    print(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+'  '+stri)
    
def preprocess_data_en(stopwords,doc):
    '''
    Function: preprocess data in Chinese including cleaning, tokenzing...
    Input: document string
    Output: list of words
    '''     
    doc = doc.lower()
    doc = word_tokenize(doc)
    doc = [word for word in doc if word not in set(stopwords)]
    doc = [word for word in doc if word.isalpha()]  
    return doc

def preprocess_data_cn(stopwords,doc):
    '''
    Function: preprocess data in Chinese including cleaning, tokenzing...
    Input: 
        stopwords: Chinese stopwords list
        doc: document string
    Output: list of words
    '''       
    # clean data
    doc = re.sub(u"[^\u4E00-\u9FFF]", "", doc) # delete all non-chinese characters
    # tokenize and move stopwords 
    doc = [word for word in jieba.cut(doc) if word not in set(stopwords)]         
    return doc

def filter_words(vocab,doc):
    '''
    Function: filter words which are not contained in the vocab
    Input:
        vocab: list of words that have word2vec representation
        doc: list of words in a document
    Output:
        list of filtered words     
    '''
    res = [word for word in doc if word in vocab]
    return res

def f(x):
    if x<0.0: return 0.0
    else: return x
    
def handle_sim(x):  
    return 1.0-np.vectorize(f)(x)

def regularize_sim(sims):
    '''
    Function: replace illegal similarity value -1 with mean value
    Input: list of similarity of document pairs
    Output: regularized list of similarity 
    '''
    sim_mean = np.mean([sim for sim in sims if sim!=-1])
    r_sims = []
    errors = 0
    for sim in sims:
        if sim==-1:
            r_sims.append(sim_mean)
            errors += 1
        else:
            r_sims.append(sim)
#     LogInfo('Regularize: '+str(errors))
    return r_sims

def doc_vector(model,doc):
    '''
    Function:
        compute the mean of word vectors
    Input:
        model: gensim word2vec model
        doc: list of words
    Output:
        doc vector 
    '''
    # remove out-of-vocab words
    doc = [word for word in doc if word in model.vocab]
    return np.mean(model[doc],axis=0)

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

def get_nSIM_representations(model,doc1,doc2):
    SIM1 = []
    SIM2 = []
    # get vocabulary 
    V = set(doc1+doc2)
    for word in V:
        word_vec = np.array(model[word]).reshape(1,-1)
        doc_vec1 = np.array(doc_vector(model,doc1)).reshape(1,-1)
        doc_vec2 = np.array(doc_vector(model,doc2)).reshape(1,-1)
        if word in doc1:
            cos1 = 1
        else:
            cos1 = cosine_similarity(word_vec,doc_vec1)[0][0]  
        if word in doc2:
            cos2 = 1
        else:
            cos2 = cosine_similarity(word_vec,doc_vec2)[0][0]  
        SIM1.append(cos1)
        SIM2.append(cos2)

    nSIM1 = softmax(np.array(SIM1)).astype(np.double)
    nSIM2 = softmax(np.array(SIM2)).astype(np.double)
    return V,nSIM1,nSIM2

def STR(lang,refs,hyps):
    '''
    Function:
        calculate similarity of document pairs 
    Input: 
        lang: text language - 'cn' for Chinese/'en' for English
        ref:  list of reference documents 
        hyp: list of hypothesis documents
    Output:
        sent_STRs (sentence score): list of STR scores for each translation (ref-hyp pair) 
        corpus_STR (corpus score): average STR score for the whole corpus
    '''
    
    # change setting according to text language 
    if lang=='cn':
        model_path = config['cn_model_path']
        stopwords_path = config['cn_stopwords_path']
        preprocess_data = preprocess_data_cn
    elif lang=='en':
        model_path = config['en_model_path']
        stopwords_path = config['en_stopwords_path']
        preprocess_data = preprocess_data_en
    LogInfo('Load word2vec model...')
    w2v_model = KeyedVectors.load_word2vec_format(model_path,binary=True,unicode_errors='ignore')
#     model = KeyedVectors.load(model_path, mmap='r')
    w2v_vocab = w2v_model.wv.vocab
    stopwords= set(w.strip() for w in codecs.open(stopwords_path, 'r',encoding='utf-8').readlines())
    sent_STRs = []
    LogInfo('Calculate STR score...')
    for i in range(len(refs)):
        ref = refs[i]
        hyp = hyps[i]
        # preprocess data
        p1 = preprocess_data(stopwords,ref)
        p2 = preprocess_data(stopwords,hyp)
        # filter words which do not have word2vec embeddings
        p1 = filter_words(w2v_vocab,p1)
        p2 = filter_words(w2v_vocab,p2)
        if len(p1)==0 or len(p2)==0:
            # if any filtered document is null, return -1 
            STR = -1
        else:
            # obtain nSIM representations
            V,nSIM1,nSIM2 = get_nSIM_representations(w2v_model,p1,p2)
            # obtain word2vec representations 
            W = [w2v_model[word] for word in V]
            # calculate word distance matrix (distance = 1-cosine similarity) [0,1]
            D = handle_sim(cosine_similarity(W)).astype(np.double) 
            # calculate minimal distance using EMD algorithm
            min_distance = emd(nSIM1,nSIM2,D)
            # calculate STR score (STR = 1-min_distance)
            STR = 1-min_distance
            sent_STRs.append(STR)
    # regularize STR: replace -1 with average STR score
    sent_STRs = regularize_sim(sent_STRs) 
    corpus_STR = np.mean(sent_STRs)
    sent_STRs = np.round(sent_STRs,3)
    corpus_STR = np.round(corpus_STR,3)
    return sent_STRs, corpus_STR

if __name__ == '__main__':
    # Example usage
    refs = ['man sitting using tool at a table in his home.',
                 'vegetable is being sliced.',
                'a speaker presents some products']
    hyps = ['The president comes to China',
                'someone is slicing a tomato with a knife on a cutting board.',
                'the speaker is introducing the new products on a fair.']
    lang = 'en'
    sent_STRs, corpus_STR = STR(lang,refs,hyps)
    for i,sent_STR in enumerate(sent_STRs):
        print("Reference: %s" %refs[i])
        print("Hypothesis: %s" %hyps[i])
        print("STR score: %.3f" %sent_STR)   
    print("Corpus STR score: %.3f" %corpus_STR)
  
    
