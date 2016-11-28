import sys
import os
# from pyspark import *
# os.environ["SPARK_HOME"] = "/usr/local/spark"

# sc = SparkContext("local", "simple app")
# 
# a = [1, 4, 3, 5]
# a = sc.parallelize(a)
# 
# print a
# print a.take(2)
# 
# b = a.map(lambda x: [x,x+5]);
# print b.collect()
# 
# b = a.flatMap(lambda x: [x,x+5]);
# print b.collect()
# 
# b = a.filter(lambda x: x > 4);
# b.cache()
# print b.collect()
# 
# b = a.reduce(lambda a, b: a*b)
# print b
# 
# b= sc.textFile("prection_xgboost_classifier.py", 4);
# b = b.filter(lambda line : "s" in line)
# print b.collect()

depth = [6, 7, 8, 9, 10, 6, 7, 8, 9, 10, 6, 7, 8, 9, 10, 6, 7, 8, 9, 10]
eta = [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4]
scalePosWeight = [5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 15, 15, 15, 15, 15, 8, 8, 8, 8, 8]
subSample = [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4]
min_child_weight = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]
subsample = [0.3, 0.3, 0.3, 0.3, 0.3, 0.6, 0.6, 0.6, 0.6, 0.6, 1, 1, 1, 1, 1, 0.4, 0.4, 0.4, 0.4, 0.4]
for index in range(len(depth)):
    print eta[index]
    print depth[index]
    print min_child_weight[index]
    print subSample[index]
    print subsample[index]
