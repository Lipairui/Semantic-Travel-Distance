# -*- coding: utf-8 -*-
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec,KeyedVectors
from gensim.models.wrappers import FastText
from nltk import word_tokenize,pos_tag
from pyemd import emd
from pyemd import emd_with_flow
import pandas as pd
import numpy as np
import codecs
import jieba
import math
import re
import time
import os
from os import listdir
from os.path import join
from config import *
from scipy import stats

def LogInfo(stri):
    print(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+'  '+stri)
    
def preprocess_data_en(doc):
    '''
    Function: preprocess data in English
    Input: document string
    Output: list of words
    '''     
    doc = doc.lower()
    doc = word_tokenize(doc)    
    return doc

def preprocess_data_cn(doc):
    '''
    Function: preprocess data in Chinese including cleaning, tokenzing...
    Input: 
        stopwords: Chinese stopwords list
        doc: document string
    Output: list of words
    '''       
    # clean data
    doc = re.sub(u"[^\u4E00-\u9FFF]", "", doc) # delete all non-chinese characters
    # tokenize  
    doc = jieba.cut(doc)         
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
    if 'unk' not in vocab:
        res = [word for word in doc if word in vocab] 
    else:
        res = [] 
        for word in doc:
            if word not in vocab:
                res.append('unk')
            else:
                res.append(word)
    return res

def regularize_score(scores):
    '''
    Function: replace illegal STD score -1 with mean value
    Input: list of STD scores
    Output: regularized list of STD scores 
    '''
    scores_mean = np.mean([s for s in scores if s!=-1])
    r_scores = []
    errors = 0
    for score in scores:
        if score==-1:
            r_scores.append(scores_mean)
            errors += 1
        else:
            r_scores.append(score)
#     LogInfo('Regularize: '+str(errors))
    return r_scores

def doc_vector(mapping,doc):
    '''
    Function:
        compute the mean of n-gram vectors
    Input:
        mapping: ngram - vector mapping
        doc: list of n-grams
    Output:
        doc vector 
    '''
    v = np.array([mapping[d] for d in doc])
    return np.mean(v,axis=0)

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

def f(x):
    if x<0.0: return 0.0
    else: return x
    
def handle_sim(x):  
    return 1.0-np.vectorize(f)(x)

def get_ngram(doc,n_gram):
    '''
    Function:
        convert list of words to list of n-grams
    Input:
        doc: list of words
        n_gram: number of grams
    Output:
        list of n-grams
    '''
    ngram_list = []
    if len(doc)<n_gram:
        ngram_list.append(' '.join(doc+[doc[-1]]*(n_gram-len(doc))))

    else:
        for i in range(len(doc)-n_gram+1):
            ngram_list.append(' '.join(doc[i:i+n_gram]))
    return ngram_list

def get_ngram_vector_dictionary(model,V,n_gram):
    '''
    Function:
        get n-gram to vector mapping
    Input:
        model: gensim word2vec model
        V: list of n-grams
        n_gram: number of grams
    Output:
        mapping dictionary
    '''
    mapping = dict()

    for ngram in V:
        words = ngram.split(' ')
        mapping[ngram] = np.concatenate(model[words],axis=0)
        
    return mapping

def get_nSIM_representations(model,doc1,doc2,n_gram):
    '''
    Function:
        get nSIM representations for doc1 and doc2
    Input:
        model: gensim word2vec model
        doc1/doc2: list of words
        n_gram: number of grams
    Output:
        mapping: n-gram vector mapping
        nSIM1/nSIM2: nSIM representations for doc1 and doc2
        p1/p2: list of n-grams
    '''
    SIM1 = []
    SIM2 = []
    # convert doc1/doc2 to n-gram format
    p1 = get_ngram(doc1,n_gram)
    p2 = get_ngram(doc2,n_gram)
    # get vocabulary 
    V = set(p1+p2)
    # get n-gram vector mapping
    mapping = get_ngram_vector_dictionary(model,V,n_gram)
    for ngram in V:
        ngram_vec = np.array(mapping[ngram]).reshape(1,-1)
        doc_vec1 = np.array(doc_vector(mapping,p1)).reshape(1,-1)
        doc_vec2 = np.array(doc_vector(mapping,p2)).reshape(1,-1)
        if ngram in p1:
            cos1 = 1
        else:
            cos1 = cosine_similarity(ngram_vec,doc_vec1)[0][0]  
        if ngram in p2:
            cos2 = 1
        else:
            cos2 = cosine_similarity(ngram_vec,doc_vec2)[0][0]  
        SIM1.append(cos1)
        SIM2.append(cos2)

    nSIM1 = softmax(np.array(SIM1)).astype(np.double)
    nSIM2 = softmax(np.array(SIM2)).astype(np.double)
    return mapping,nSIM1,nSIM2,p1,p2

def ngram_order_distance(V,p1,p2):
    '''
    Function:
        compute ngram order distance
    Input:
        V:
        p1/p2: list of n-grams
        
    Output:
        M: n-gram order distance matrix
    '''
    # build ngram order mapping dictionary
    d1 = dict(zip(p1,list(range(1,len(p1)+1))))
    d2 = dict(zip(p2,list(range(1,len(p2)+1))))
    # build distance matrix
    M = np.zeros((len(V),len(V)))
    for i,word1 in enumerate(V):
        for j,word2 in enumerate(V):
            if (word1 in p1)&(word2 in p2)&(M[i,j]==0):
                M[i,j] = abs(d1[word1]/len(p1)-d2[word2]/len(p2))
            if (word1 in p2)&(word2 in p1)&(M[i,j]==0):
                M[i,j] = abs(d2[word1]/len(p2)-d1[word2]/len(p1))    
    return M

def STD_ngram(w2v_model,refs,hyps,n_gram,args):
    '''
    Function:
        calculate segment-level and system-level STD score of n_gram
    Input: 
        w2v_model: gensim format model
        refs: list of reference documents 
        hyps: list of hypothesis documents
        n_gram: number of grams
        args: input arguments
    Output:
        seg_STDs (segment-level STD score): list of STD scores for each translation (ref-hyp pair) 
        sys_STD (system-level STD score): average STD score for the whole corpus
    '''
    
    # change setting according to text language 
    if args.lang=='cn':     
        preprocess_data = preprocess_data_cn
    elif args.lang=='en':
        preprocess_data = preprocess_data_en
    # get the vocabulary of word2vec model
    w2v_vocab = w2v_model.vocab
    
    seg_STDs = []
    null_num = 0
    for i in range(len(refs)):
        ref = refs[i]
        hyp = hyps[i]
        # preprocess data
        doc1 = preprocess_data(ref)
        doc2 = preprocess_data(hyp)
        # filter words which do not have word2vec embeddings
        doc1 = filter_words(w2v_vocab,doc1)
        doc2 = filter_words(w2v_vocab,doc2)
        if len(doc1)==0 or len(doc2)==0:
            # if any filtered document is null, return -1 
            null_num += 1
            STD = -1
        else:
            # obtain nSIM representations
            mapping,nSIM1,nSIM2,p1,p2 = get_nSIM_representations(w2v_model,doc1,doc2,n_gram)            
            # obtain word2vec embeddings 
            W = list(mapping.values())   
            # obtain vocabulary
            V = list(mapping.keys()) 
            # calculate semantic distance matrix (distance = 1-cosine similarity) [0,1]
            D1 = handle_sim(cosine_similarity(W)).astype(np.double)          
            # calculate n-gram order distance matrix
            D2 = ngram_order_distance(V,p1,p2)
            # calculate ground distance matrix
            D = args.beta1*D1+args.beta2*D2
            # calculate STD score using EMD algorithm
            STD = emd(nSIM1,nSIM2,D)
            # normalization in the future work
            STD = STD
        seg_STDs.append(STD)
    # regularize STR: replace -1 with average STD score
#     print('Null num: %d' %null_num)
    seg_STDs = regularize_score(seg_STDs) 
    # calculate system-level STD
    sys_STD = np.mean(seg_STDs,axis=0)
  
    return seg_STDs, sys_STD

def STD(w2v_model,refs,hyps,args):
    '''
    Function:
        calculate segment-level and system-level STD score by combining the results of unigram and bigram
    Input: 
        w2v_model: gensim format model
        lang: text language - 'cn' for Chinese/'en' for English
        refs: list of reference documents 
        hyps: list of hypothesis documents
        args: input arguments
    Output:
        seg_STDs (segment-level STD score): list of STD scores for each translation (ref-hyp pair) 
        sys_STD (system-level STD score): average STD score for the whole corpus
    '''
    # calculate unigram STD score
    seg_STDs_unigram, sys_STD_unigram = STD_ngram(w2v_model,refs,hyps,1,args)
    # calculate bigram STD score
    seg_STDs_bigram, sys_STD_bigram = STD_ngram(w2v_model,refs,hyps,2,args)
    # combine unigram and bigram STD score
    # segment-level
    seg_STDs = args.alpha1*np.array(seg_STDs_unigram) + args.alpha2*np.array(seg_STDs_bigram)
    # system-level
    sys_STD = args.alpha3*np.array(sys_STD_unigram) + args.alpha4*np.array(sys_STD_bigram)
    
    return seg_STDs, sys_STD

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description='STD: Semantic Travel Distance - An automatic evaluation metric for Machine Translation.'
        )
    parser.add_argument('-r', '--ref', help='Reference file', required=True)
    parser.add_argument('-o', '--hyp', help='Hypothesis file', required=True)
    parser.add_argument('-l', '--lang', help='Language: en for English; cn for Chinese', required=True, default='en')
    parser.add_argument('-a1', '--alpha1', help='Weight of unigram STD score at segment level (Default: 0.5)', required=False, default=0.5)
    parser.add_argument('-a2', '--alpha2', help='Weight of bigram STD score at segment level (Default: 0.5)', required=False, default=0.5)
    parser.add_argument('-a3', '--alpha3', help='Weight of unigram STD score at syetem level (Default: 0.3)', required=False, default=0.3)
    parser.add_argument('-a4', '--alpha4', help='Weight of bigram STD score at system level (Default: 0.7)', required=False, default=0.7)
    parser.add_argument('-b1', '--beta1', help='Weight of semantic distance matrix (Default: 0.6)', required=False, default=0.6)
    parser.add_argument('-b2', '--beta2', help='Weight of n-gram order distance matrix (Default: 0.4)', required=False, default=0.4)
    parser.add_argument('-v', '--verbose', help='Print score of each sentence (Default: False)',
                        action='store_true', default=False)
    
    return parser.parse_args()

def main():
    # Parsing arguments
    args = parse_args()
    hyps = [x for x in codecs.open(args.hyp, 'r', 'utf-8').readlines()]
    refs = [x for x in codecs.open(args.ref, 'r', 'utf-8').readlines()]
    """
    Check whether the hypothesis and reference files have the same number of
    sentences
    """
    if len(hyps) != len(refs):
        print("Error! {0} lines in the hypothesis file, but {1} lines in the"
              " reference file.".format(len(hyp_lines), len(ref_lines)))
        sys.exit(1)
    # Load word2vec model 
    LogInfo('Load word2vec model...')
    w2v_model = KeyedVectors.load_word2vec_format(config['model_path'],binary=True,unicode_errors='ignore')
  
    # Calculate STD score
    LogInfo('Calculate STD score...')
    seg_STDs, sys_STD = STD(w2v_model,refs,hyps,args)
    # Print out scores of every sentence
    if args.verbose:
        for index in range(1,len(seg_STDs)+1):
            print("STD score of sentence {0} is {1:.4f}".format(index, seg_STDs[index-1])) 
    print('STD score at system level: %.4f' %sys_STD)

if __name__ == '__main__':
    main()
