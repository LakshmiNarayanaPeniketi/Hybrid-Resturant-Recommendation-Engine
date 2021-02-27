import findspark
findspark.init()
findspark.find()
import itertools
import pyspark
import sys
import time
import json
import string
import re
import math
from math import log2
from collections import defaultdict,Counter
from string import punctuation
from pyspark import SparkContext, SparkConf





def merge(text):
    out=""
    for values in text:
        out+=values
    return out

def text_preprocess(x,stopwords):
    words = []
    for text in x[1]:
        text = re.sub(r'[^\w\s]','',text)
        text = text.replace('\n', " ")
        text = text.split(' ')

    for word in text:
        word = word.strip().lower()
        if (word.isnumeric()):
            continue
        if (word not in stopwords):
            words.append(word)

    return (x[0],words)

def removal_digits(text):
    output = ''.join(c for c in text if not c.isdigit())
    return output

def removal_singledigit(messages):
    list_words=[]
    for words in messages:
        if len(words)>1:
            list_words.append(words)
    return list_words

def strip_punctuation(text):
    return ''.join(c for c in text if c not in punctuation)

def string_lower(text):
    L=' '.join([w.lower() for w in text])
    return L

def removal_stopwords(text,words):
    removing_stopwords = [word for word in text if word not in words]
    return removing_stopwords

def TF_IDF(tf_idf,idf_values):
    IDF={}
    for words in tf_idf:
        IDF[words[0]]=words[1]*idf_values[words[0]]
    #IDF=sorted(IDF.items(), key=lambda x: x[1], reverse=True)
    return [(words[0],IDF[words[0]]) for words in tf_idf]

def create_dict(bagOfWordsA):
    numOfWordsA ={}
    tf_dict={}
    for word in bagOfWordsA:
        if word in numOfWordsA:
            numOfWordsA[word] += 1
        else:
            numOfWordsA[word]=1
    for words in bagOfWordsA:
        tf_dict[words]=numOfWordsA[words]/float(len(bagOfWordsA))
    return [(words,tf_dict[words]) for words in bagOfWordsA]

def merge_lists(values):
    value=[]
    for v in values:
        value.extend(v)
    return value

def Sort_Tuple(tup):  
    return(sorted(tup, key = lambda x: x[1],reverse=True))



if __name__ == "__main__":
    if len(sys.argv)!=4:
        print("This function needs 3 input arguments <case number> <support> <input_file_path> <output_file_path>")
        sys.exit(1)
    
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    stopwords=sys.argv[3]
    
    conf = (
    SparkConf()
    .setAppName("your app name")
    .set("spark.driver.memory", "4g")
    .set("spark.executor.memory", "4g"))
    sc = SparkContext(conf=conf)

    time1=time.time()
    reviews = sc.textFile(train_file).persist()
    rdd=reviews.map(lambda x:json.loads(x))

    #lineList = [line.rstrip('\n') for line in open(stopwords)]
    #punc_words=['[',']','(',')','.',':',';','!','?',',']
    #stop_words=lineList+punc_words

    file = open(stopwords,'r')
    stop_words = set()
    for word in file:
        stop_words.add(word.rstrip())
    
    text_merged=rdd.map(lambda x:(x['business_id'],x['text'])).groupByKey()
    splitted_rdd=text_merged.map(lambda x:text_preprocess(x,stop_words))
    
    #text_merged=rdd.map(lambda x:(x['business_id'],x['text'])).groupByKey().mapValues(list).map(lambda x:(x[0],merge(x[1])))
    #splitted_rdd=text_merged.map(lambda x:(x[0],removal_digits(x[1]))).map(lambda x:(x[0],strip_punctuation(x[1]))).map(lambda x:(x[0],x[1].split()))
    rdd3=splitted_rdd.map(lambda x:(x[0],[word.lower() for word in x[1]])).map(lambda x:(x[0],[word.strip() for word in x[1]]))
    rdd4=rdd3.map(lambda x:(x[0],removal_singledigit(x[1]))).map(lambda x:(x[0],removal_stopwords(x[1],stop_words)))


    list_set_words=rdd4.map(lambda x:[words for words in set(x[1])]).collect()
    final_idf={}
    for reviews in list_set_words:
        for words in reviews:
            if words in final_idf:
                final_idf[words]+=1
            else:
                final_idf[words]=1

    N=len(list_set_words)
    idf_values={}
    for keys in final_idf:
        idf_values[keys]=math.log2(N/float(final_idf[keys]))


    CV=rdd4.map(lambda x:(x[0],create_dict(x[1]))).map(lambda x:(x[0],TF_IDF(x[1],idf_values))).map(lambda x:(x[0],Sort_Tuple(x[1])))
    
    
    CV1=CV.map(lambda x:(x[0],[words[0] for words in x[1]])).mapValues(set).mapValues(list).map(lambda x:(x[0],x[1][:200]))
    
    #CV1=CV.map(lambda x:(x[0],[words[0] for words in x[1]])).map(lambda x:(x[0],set(x[1]))).map(lambda x:(x[0],list(x[1]))).map(lambda x:(x[0],x[1][:200]))
    list_distinct_reviews=CV1.map(lambda x:x[1]).collect()
    distinct_words=[]
    for lists in list_distinct_reviews:
        distinct_words.extend(lists)
    set_distinctwords=sorted(set(distinct_words))
    distinct_word_dict={}
    i=0
    for values in set_distinctwords:
        distinct_word_dict[values]=i
        i+=1


    CV2=dict(CV1.map(lambda x:(x[0],[distinct_word_dict[values] for values in x[1]])).collect())
    #user_rdd=rdd.map(lambda x: (x['user_id'], x['business_id'])).map(lambda x:(x[0],CV2[x[1]])).groupByKey().mapValues(list)
    user_rdd=rdd.map(lambda x: (x['user_id'], x['business_id'])).groupByKey().mapValues(list)
    user_profile_dict=dict(user_rdd.collect())
    #user_profile_dict1=dict(user_rdd.map(lambda x:(x[0],merge_lists(x[1]))).map(lambda x:(x[0],list(set(x[1])))).collect())
    
    with open(model_file, 'w') as fp:
        fp.writelines(json.dumps({x[0] : x[1]}) + '\n' for x in CV2.items())
        fp.writelines(json.dumps({x[0] : x[1]}) + '\n' for x in user_profile_dict.items())

    Duration=time.time()-time1
    print('Duration:',Duration)