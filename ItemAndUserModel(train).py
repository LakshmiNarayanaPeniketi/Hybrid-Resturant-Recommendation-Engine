import findspark
findspark.init()
findspark.find()
import itertools
import pyspark
import sys
import time
import json
from collections import defaultdict
from operator import add
from pyspark import SparkContext, SparkConf
from pyspark import SparkContext
from itertools import combinations
import random

def hash_value(j,i,Num_hash,signature_matrix,hashfunction,prime_num):
    for k in range(Num_hash):
        signature_matrix[k][j] = min(signature_matrix[k][j],(i*hashfunction[k][0]+hashfunction[k][1]) % prime_num)
    return signature_matrix

def sign1(j,dict_business,Num_hash,signature_matrix,hashfunction,prime_num):
    for i in dict_business[j]:
        sign1=hash_value(j,i,Num_hash,signature_matrix,hashfunction,prime_num)
    return signature_matrix

def Prime_check(x):
    k=[False for i in range(2,x) if x%i==0]
    if not k:
        return True
    else:
        return False
    
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

def pairCandidatesz(pairs):
    list_1=dict_user[pairs[0]]
    list_2=dict_user[pairs[1]]
    result = list()
    intersected=list(set(list_1) & set(list_2))
    rated_intersection=len(intersected)
    rated_union=len(set(list_1+list_2))    
    if rated_intersection >= 3 and rated_intersection / rated_union >= 0.01:
        user1=reversed_uid[pairs[0]]
        user2=reversed_uid[pairs[1]]
        List_ratings1=[]
        List_ratings2=[]
        for items in intersected:
            items=reversed_bid[items]
            List_ratings1.append(utility_dict[items,user1])
            List_ratings2.append(utility_dict[items,user2])
        mean1=sum(List_ratings1)/len(List_ratings1)
        mean2=sum(List_ratings2)/len(List_ratings2)
        List_ratings1=[values-mean1 for values in List_ratings1]
        List_ratings2=[values-mean2 for values in List_ratings2]
        result.append(((user1, user2), (List_ratings1,List_ratings2)))
    return result

def Pearson(ratings):
    check1=ratings[0]
    check2=ratings[1]
    numerator=sum([x*y for x,y in zip(check1,check2)])
    sum_den1=(sum([x**2 for x in check1]))**0.5
    sum_den2=(sum([x**2 for x in check2]))**0.5
    denominator=(sum_den1)*sum_den2
    
    try:
        if numerator / denominator > 0:
            return numerator / denominator
    except ZeroDivisionError:
        return None
    
    
def valid_pairs(baskets, support, numofwhole):
    sub_s = int(support*(len(baskets))/numofwhole)
    check_dict = {}    
    for basket in baskets:
        list_basket=list(basket)
        list_basket.sort(key=lambda x:x[0])
        for value1, value2 in combinations(list_basket, 2): 
            if tuple([value1[0],value2[0]]) in check_dict:
                check_dict[tuple([value1[0],value2[0]])]+=[(value1[1],value2[1])]
            else:
                check_dict[tuple([value1[0],value2[0]])]=[(value1[1],value2[1])]
    
    L=[]
    for keys in check_dict.keys():
        if len(check_dict[keys])>=sub_s:
            L.append([keys,check_dict[keys]])
    return L

def pearson_correlation(pairs):
    list_pairs=list(pairs)
    if len(list_pairs)==0:
        return 0
    item1_ratings=[pairs[0] for pairs in pairs]
    item2_ratings=[pairs[1] for pairs in pairs]
    mean_item1=sum(item1_ratings)/len(item1_ratings)
    mean_item2=sum(item2_ratings)/len(item2_ratings)
    check1=[item-mean_item1 for item in item1_ratings]
    check2=[item-mean_item2 for item in item2_ratings]
    numerator=sum([x*y for x,y in zip(check1,check2)])
    sum_den1=(sum([x*y for x,y in zip(check1,check1)]))**0.5
    sum_den2=(sum([x*y for x,y in zip(check2,check2)]))**0.5
    denominator=(sum_den1)*sum_den2
    if numerator==0 or denominator==0:
        return 0
    return numerator/denominator
    

    
if __name__ == "__main__":
    if len(sys.argv)!=4:
        print("This function needs 3 input arguments <case number> <support> <input_file_path> <output_file_path>")
        sys.exit(1)
    input_file=sys.argv[1]
    outputfile=sys.argv[2]
    condition = sys.argv[3]
    
    conf = (
    SparkConf()
    .setAppName("your app name")
    .set("spark.driver.memory", "4g")
    .set("spark.executor.memory", "4g"))
    sc = SparkContext(conf=conf)
    #sc = SparkContext('local[*]','test')
    
    time1=time.time()
    reviews = sc.textFile(input_file).persist()
    rdd=reviews.map(lambda x:json.loads(x))
    
    if condition=='item_based':
        
        reviews = sc.textFile(input_file).persist()
        rdd=reviews.map(lambda x:json.loads(x))
        ext_rdd=rdd.map(lambda x:(x['user_id'],(x['business_id'],x['stars'])))
        test_reviews = sc.textFile('test_review.json').persist()
        test_rdd=test_reviews.map(lambda x:json.loads(x))
        test_rdd=test_rdd.map(lambda x:(x['user_id'],x['business_id']))
        trainRDD_user = ext_rdd.groupByKey().mapValues(dict).collect()
        UserDict = dict(trainRDD_user)
        user_avg_rating=ext_rdd.map(lambda x:(x[0],x[1][1])).groupByKey().mapValues(list).map(lambda x:(x[0],sum(x[1])/len(x[1]))).collect()
        avgsDict = dict(user_avg_rating)
        overall_avg = ext_rdd.map(lambda row: row[1][1]).mean()
        count_user=ext_rdd.map(lambda x:x[0]).distinct().count()
        k=ext_rdd.groupByKey().map(lambda x:x[1]).mapPartitions(lambda x:valid_pairs(list(x),7,count_user))
        pearson_corr_values=k.reduceByKey(add).mapValues(pearson_correlation)
        final_rdd=pearson_corr_values.filter(lambda x:x[1]>0 and x[1]is not None)
        final=final_rdd.map(lambda x: {"b1": x[0][0], "b2": x[0][1], "sim": x[1]}) 
        with open(outputfile, 'w') as fp:
            fp.writelines(json.dumps(t) + '\n' for t in final.collect())   
        
    
    
    
    if condition=='user_based':
        check1=rdd.map(lambda x:(x['user_id'],x['business_id'],float(x['stars'])))
        businessid_unique = check1.map(lambda x: (x[1])).distinct().collect()
        businessid_count=len(businessid_unique) 
        business_dict={}
        i=0
        for ids in businessid_unique:
            business_dict[ids]=i
            i+=1

        reversed_bid = {v : k for k, v in business_dict.items()}

        uid_dict = dict(check1.map(lambda x: (x[0])).distinct().zipWithIndex().collect())
        reversed_uid = {v : k for k, v in uid_dict.items()}

        utility_dict = dict(check1.map(lambda x: ((x[1], x[0]), x[2])).groupByKey().mapValues(lambda l: sum(l) / len(l)).collect())
        userid_unique = check1.map(lambda x: (x[0],x[1])).groupByKey().mapValues(set).collect()
        userid_count=len(userid_unique)
        dict_user = {}
        for i in range(userid_count):
            dict_user[i]=[]
            for u in userid_unique[i][1]:
                dict_user[i].append(business_dict[u])

        prime_num=userid_count
        while not Prime_check(prime_num):
            prime_num += 1
        Num_hash = 50
        hashfunction = []
        random.seed(10000)
        for i in range(Num_hash):  
            hashfunction.append([random.randint(0, 10000), random.randint(0, 10000)])

        signature_matrix = [[businessid_count for col in range(userid_count)] for row in range(Num_hash)]

        row = 1
        b = Num_hash / 1
        for j in range(userid_count):
                signature_matrix=sign1(j,dict_user,Num_hash,signature_matrix,hashfunction,prime_num)

        
        sign_rdd = sc.parallelize(signature_matrix, b)
        candidates = sign_rdd.mapPartitions(lambda x: Candidate(x, userid_count, row)).map(lambda x: (x, 1))
        c=candidates.reduceByKey(lambda x, y: 1).map(lambda x: x[0])

        li=c.flatMap(pairCandidatesz).mapValues(Pearson).filter(lambda x:x[1] is not None).map(lambda x: {"u1" : x[0][0], "u2" : x[0][1], "sim" : (x[1])}) \
                    .filter(lambda x: x["sim"])

        with open(outputfile, 'w') as fp:
            fp.writelines(json.dumps(t) + '\n' for t in li.collect())

    Duration=time.time()-time1
    print('Duration:',Duration)