import sys
import os
from pyspark import *
os.environ["SPARK_HOME"] = "/usr/local/spark"

sc = SparkContext("local", "simple app")

a = [1, 4, 3, 5]
a = sc.parallelize(a)

print a
print a.take(2)

b = a.map(lambda x: [x,x+5]);
print b.collect()

b = a.flatMap(lambda x: [x,x+5]);
print b.collect()

b = a.filter(lambda x: x > 4);
b.cache()
print b.collect()

b = a.reduce(lambda a, b: a*b)
print b

b= sc.textFile("prection_xgboost_classifier.py", 4);
b = b.filter(lambda line : "s" in line)
print b.collect()




