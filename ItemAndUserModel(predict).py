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


def ItemSimilarity(x,model_Dict):
    if x[1] is not None:
        output=[]
        ta= x[0]
        r_b =dict(x[1])
        if len(r_b) < 7:
            output=less_users(ta,r_b)
        else:
            first=more_users(ta,r_b)
            O=arranging(first)
            LL=[(values[1],r_b[values[0]]) for values in O]
            output=[]
            output.extend(LL)
        if not output:
            return None
        return ta, output
    else:
        return x
    

def arranging(list_items):
    sorted_list=sorted(list_items, key=lambda x: x[1], reverse=True)
    return sorted_list[:7]
    
    
def less_users(ta,r_b):
    output=[]
    for items in r_b.items():
        k = tuple(sorted([ta, items[0]]))
        if k in model_Dict:
            output.append((model_Dict[k], items[1]))
    return output


def more_users(ta,r_b):
    list_a=[]
    for items in r_b:
        k = tuple(sorted([ta, items]))
        if k in model_Dict:
            list_a.append((items, model_Dict[k]))
    return list_a

def Rating1(lists):
    num= 0
    den=0
    list_1=[value[0] for value in lists]
    list_2=[value[1] for value in lists]
    den=sum(list_1)
    list_a= [x*y for x,y in zip(list_1,list_2)]
    num=sum(list_a)
    return num/den

def merge(x):
    L=[]
    for values in x:
        L.extend(values)
    return L

def num_den_calc(x,ratings):
    pairs=x[0]
    def num_den(x,business,p):
        p1=0
        q1=0
        for values in p:
            user_id=values[0]
            similarity=values[1]
            if (user_id,x[0][1]) in ratings:
                v=ratings[(user_id,x[0][1])]-user_avg_dict[user_id]
                p1=p1+(v*similarity)
                q1=q1+similarity
        return p1,q1
    numer_,den_=num_den(x,x[0][1],x[1])
    
    if den_!=0:
        similarity1=numer_/den_
        similarity=similarity1+user_avg_dict[pairs[0]]
        return (pairs,similarity)

def func(x):
    return [(x["u1"], (x["u2"], x["sim"])), (x["u2"], (x["u1"], x["sim"]))]

def sorting(values):
    sorted(values,key = lambda y: y[1],reverse=True)
    return values
if __name__ == "__main__":
    if len(sys.argv)!=6:
        print("This function needs 5 input arguments <case number> <support> <input_file_path> <output_file_path>")
        sys.exit(1)
    input_file=sys.argv[1]
    test_file = sys.argv[2]
    model_file=sys.argv[3]
    outputfile=sys.argv[4]
    condition = sys.argv[5]
    
    sc = SparkContext('local[*]','test')
    time1=time.time()
    if condition=='user_based':
        
        
        reviews = sc.textFile(input_file).persist()
        rdd=reviews.map(lambda x:json.loads(x))
        test_reviews = sc.textFile(test_file).persist()
        test_rdd=test_reviews.map(lambda x:json.loads(x))
        test_rdd=test_rdd.map(lambda x:(x['user_id'],x['business_id']))
        ext_rdd=rdd.map(lambda x:(x['user_id'],(x['business_id'],x['stars'])))
        ext_rdd2=ext_rdd.map(lambda x:((x[0],x[1][0]),float(x[1][1])))
        ratings = dict(ext_rdd2.collect())
        user_avg=ext_rdd.map(lambda x:(x[0],x[1][1])).groupByKey().mapValues(list).map(lambda x:(x[0],sum(x[1])/len(x[1]))).collect()
        user_avg_dict=dict(user_avg)
        calculated_wts=sc.textFile(model_file)
        calc_rdd=calculated_wts.map(lambda x:json.loads(x))
        weigthed_rdd=calc_rdd.flatMap(lambda x:func(x))
        intermediate_rdd=test_rdd.join(weigthed_rdd).map(lambda x: ((x[0], x[1][0]), x[1][1])).groupByKey()
        rdd2=intermediate_rdd.map(lambda x:(x[0],sorted(x[1],key = lambda y: y[1],reverse=True))).map(lambda x:(x[0],x[1][:100]))
        final_output=rdd2.map(lambda x:num_den_calc(x,ratings)).filter(lambda x:x is not None).map(lambda x: {"user_id" : x[0][0], "business_id" : x[0][1], "stars" : x[1]})
        with open(outputfile, 'w') as fp:
            fp.writelines(json.dumps(t) + '\n' for t in final_output.collect())

    if condition=='item_based':
        
        reviews = sc.textFile(input_file).persist()
        rdd=reviews.map(lambda x:json.loads(x))
        trainRDD=rdd.map(lambda x: (x["user_id"],x["business_id"],float(x["stars"])))
        test_reviews = sc.textFile(test_file).persist()
        test_rdd=test_reviews.map(lambda x:json.loads(x))
        testRDD=test_rdd.map(lambda x:(x['user_id'],x['business_id']))
        model_file = sc.textFile(model_file).persist()
        model_rdd=model_file.map(lambda x:json.loads(x))
        model_Dict=dict(model_rdd.map(lambda x:((x['b1'],x['b2']),x['sim'])).collect())
        grouped_rdd=trainRDD.map(lambda x:(x[0],(x[1],x[2]))).groupByKey()
        joined_rdd = testRDD.join(grouped_rdd)
        similar_rdd=joined_rdd.map(lambda x:(x[0],ItemSimilarity(x[1],model_Dict)))
        filtered_rdd=similar_rdd.filter(lambda x: x[1] is not None)
        calculated_rdd=filtered_rdd.map(lambda x:(x[0],x[1][0],Rating1(x[1][1])))
        final_rdd=calculated_rdd.map(lambda x: {"user_id" : x[0], "business_id" : x[1], "stars" : x[2]})
        with open(outputfile, 'w') as file:
            file.writelines(json.dumps(values) + '\n' for values in final_rdd.collect())
            
    Duration=time.time()-time1
    print('Duration:',Duration)