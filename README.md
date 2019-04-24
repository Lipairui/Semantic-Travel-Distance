# Semantic Travel Distance (STD)
A novel automatic evaluation metric for Machine Translation based on word embeddings. 

## Algorithm description
STD incorporates both semantic and lexical features (word embeddings & n-gram & word order) into one metric. It measures the semantic distance between the hypothesis and reference by calculating the minimum cumulative cost that the embedded n-grams of the hypothesis need to “travel” to reach the embedded n-grams of the reference. Experiment results show that STD has a better and more robust performance than a range of state-of-the-art metrics for both the segment-level and system-level evaluation.

## Dependencies
python 3.6.5     
pyemd, numpy, gensim, sklearn, nltk, jieba, codecs, re

## Pretrained word embeddings model used in this algorithm 
Chinese pretrained model:      
http://pan.baidu.com/s/1qX334vE      
English pretrained model:         
https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip

## Usage
1. Set pretrained word embeddings model path in config.py
2. STD.py [-h] -r REF -o HYP -l LANG [-a1 ALPHA1] [-a2 ALPHA2]       
   [-a3 ALPHA3] [-a4 ALPHA4] [-b1 BETA1] [-b2 BETA2] [-v]        
   
   STD: Semantic Travel Distance - An automatic evaluation metric for Machine Translation.        
   
   optional arguments:        
   -h, --help            show this help message and exit      
   -r REF, --ref REF     Reference file        
   -o HYP, --hyp HYP     Hypothesis file        
   -l LANG, --lang LANG  Language: en for English; cn for Chinese        
   -a1 ALPHA1, --alpha1 ALPHA1        
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Weight of unigram STD score at segment level (Default: 0.5)          
   -a2 ALPHA2, --alpha2 ALPHA2          
                        Weight of bigram STD score at segment level (Default: 0.5)           
   -a3 ALPHA3, --alpha3 ALPHA3           
                        Weight of unigram STD score at syetem level (Default: 0.3)          
   -a4 ALPHA4, --alpha4 ALPHA4         
                        Weight of bigram STD score at system level (Default: 0.7)          
   -b1 BETA1, --beta1 BETA1         
                        Weight of semantic distance matrix (Default: 0.6)          
   -b2 BETA2, --beta2 BETA2         
                        Weight of n-gram order distance matrix (Default: 0.4)           
   -v, --verbose         Print score of each sentence (Default: False)
       
