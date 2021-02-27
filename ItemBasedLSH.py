import findspark
findspark.init()
findspark.find()
import itertools
import pyspark
import sys
import time
import json
from collections import defaultdict
from pyspark import SparkContext, SparkConf
from pyspark import SparkContext
from itertools import combinations
import random

def Prime_check(x):
    k=[False for i in range(2,x) if x%i==0]
    if not k:
        return True
    else:
        return False

def hash_value(j,i,Num_hash,signature_matrix,hashfunction,prime_num):
    for k in range(Num_hash):
        signature_matrix[k][j] = min(signature_matrix[k][j],(i*hashfunction[k][0]+hashfunction[k][1]) % prime_num)
    return signature_matrix

def sign1(j,dict_business,Num_hash,signature_matrix,hashfunction,prime_num):
    for i in dict_business[j]:
        sign1=hash_value(j,i,Num_hash,signature_matrix,hashfunction,prime_num)
    return signature_matrix


def jacc_sim(pair,dict_business):
    busi_id1 = dict_business[pair[0]]
    busi_id2 = dict_business[pair[1]]
    intersection = len(list(set(busi_id1) & set(busi_id2)))
    union = len(set(busi_id1+busi_id2))
    sim = (intersection)/(union)
    return [pair[0],pair[1],sim]


def Candidate(data, c, r):
    cand = []
    d_s = []
    dict1 = {}
    data = list(data)
    for i in range(c):
        s_list = []
        for j in range(r):
            s_list.append(data[j][i])
            d_s.append(data[j][i])
        s_list = tuple(s_list)
        if s_list not in dict1:
            dict1[s_list] = []
        dict1[s_list].append(i)
    for values in dict1.items():
        if len(values[1]) > 1:
            cand.extend(list(combinations(values[1], 2)))
    return iter(cand)


if __name__ == "__main__":
    if len(sys.argv)!=3:
        print("This function needs 2 input arguments <case number> <support> <input_file_path> <output_file_path>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    
    sc = SparkContext('local[*]','test')

    time1=time.time()
    reviews = sc.textFile(input_file).persist()
    rdd=reviews.map(lambda x:json.loads(x))
    check1=rdd.map(lambda x:(x['user_id'],x['business_id'],x['stars']))

    userid_unique=check1.map(lambda x:x[0]).distinct().collect()
    userid_dict={}
    i=0
    for ids in userid_unique:
        userid_dict[ids]=i
        i+=1

    userid_count=len(userid_unique) 

    businessid_unique = check1.map(lambda x: (x[1],x[0])).groupByKey().mapValues(set).collect()
    businessid_count=len(businessid_unique) 
    dict_business = {}
    for i in range(businessid_count):
        dict_business[i]=[]
        for u in businessid_unique[i][1]:
            dict_business[i].append(userid_dict[u])


    prime_num=userid_count
    while not Prime_check(prime_num):
            prime_num += 1


    Num_hash = 50
    hashfunction = []
    random.seed(10000)
    for i in range(Num_hash):  # generate hash functions
        hashfunction.append([random.randint(0, 10000), random.randint(0, 10000)])

    signature_matrix = [[userid_count for col in range(businessid_count)] for row in range(Num_hash)]


    for j in range(businessid_count):
        signature_matrix=sign1(j,dict_business,Num_hash,signature_matrix,hashfunction,prime_num)


    row = 1
    band = Num_hash / 1
    sign_rdd = sc.parallelize(signature_matrix, band)
    candidates = sign_rdd.mapPartitions(lambda x: Candidate(x,businessid_count,row)).map(lambda x: (x,1))
    final_candidates=candidates.reduceByKey(lambda x, y: 1).map(lambda x: x[0]).collect()


    similar_pairs=[]
    for pairs in final_candidates:
        [p1,p2,sim]=jacc_sim(pairs,dict_business)
        if sim>0.05:
            similar_pairs.append([businessid_unique[p1][0],businessid_unique[p2][0],sim])       


    signature = sc.parallelize(similar_pairs)
    similarity = signature.map(lambda x: {"b1" : x[0], "b2": x[1], "sim": x[2]}).cache()
    with open(output_file, 'w') as printer:
            printer.writelines(json.dumps(j) + '\n' for j in similarity.collect())
    Ground_truth=59435
    True_positive=len(similar_pairs)   
    Accuracy=True_positive/Ground_truth
    print('Accuracy: ',Accuracy)
    Duration=time.time()-time1
    print('Duration:',Duration)