

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
#import nltk
import math
from math import log2,sqrt
from collections import defaultdict,Counter
from string import punctuation
from pyspark import SparkContext, SparkConf

def cosine_similarity(lists):
    intersection=len(set(lists[0]).intersection(set(lists[1])))
    denominator=sqrt((len(lists[0]) * len(lists[1])))
    return intersection/denominator

def function1(x,model_dict):
    list_bids=model_dict[x]
    L=[]
    for ids in list_bids:
        L.extend(model_dict[ids])
    return set(L)


if __name__ == "__main__":
    if len(sys.argv)!=4:
        print("This function needs 3 input arguments <case number> <support> <input_file_path> <output_file_path>")
        sys.exit(1)
    
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    output_file = sys.argv[3]

    
    
    sc = SparkContext('local[*]','test')

    timer=time.time()
    
    
    test_reviews = sc.textFile(test_file).persist()
    test_rdd=test_reviews.map(lambda x:json.loads(x))
    test_rdd1=test_rdd.map(lambda x:(x['user_id'],x['business_id']))
    model_file = sc.textFile(model_file).persist()
    model_rdd=model_file.map(lambda x:json.loads(x))
    model_dict=dict(model_rdd.flatMap(lambda x:x.items()).collect())
    final_test=test_rdd1.filter(lambda x:x[0] in model_dict.keys() and x[1] in model_dict.keys())
    calc_sim=final_test.map(lambda x:((x[0],x[1]),(function1(x[0],model_dict),model_dict[x[1]]))).mapValues(cosine_similarity)
    
    final_output=calc_sim.filter(lambda x:x[1]>=0.01).map(lambda x: {"user_id" : x[0][0], "business_id" : x[0][1], "sim" : (x[1])})

    with open(output_file, 'w',encoding='utf-8') as write:
        write.writelines(json.dumps(t) + '\n' for t in final_output.collect())
    Duration=time.time()-timer
    print(Duration)