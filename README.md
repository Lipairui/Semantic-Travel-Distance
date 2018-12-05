# Semantic Travel Rate (STR)
A novel automatic evaluation metric for Machine Translation based on word embeddings. 

## Algorithm description
STR measures the semantic similarity between the hypothesis and references by calculating the minimum amount of distance that the embedded words of hypothesis need to “travel” to reach the embedded words of references. We support the evaluation for both to-English and to-Chinese machine translation.

## Dependencies
python 3.6.5
pyemd, numpy, gensim, sklearn, nltk, jieba, codecs, re

## Pretrained word2vec model used in this algorithm 
Chinese word2vec CBOW: utf8 2.18G
http://pan.baidu.com/s/1qX334vE
English word2vec 1.5G
https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download

## Example usage
Code:
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
 
Result:
Reference: man sitting using tool at a table in his home.
Hypothesis: The president comes to China
STR score: 0.674
Reference: vegetable is being sliced.
Hypothesis: someone is slicing a tomato with a knife on a cutting board.
STR score: 0.814
Reference: a speaker presents some products
Hypothesis: the speaker is introducing the new products on a fair.
STR score: 0.788
Corpus STR score: 0.758
 
    
