import numpy as np
from scipy.sparse import csr_matrix
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint

M = csr_matrix([[4, 1, 0], [4, 0, 3], [0, 0, 1]])
label = 0.0
point = LabeledPoint(label, SparseVector(3, [0, 2], [1.0, 3.0]))
 
textRDD = sc.textFile("README.md")
print textRDD.count()