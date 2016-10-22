import csv
import pandas as pd
import scipy.cluster.hierarchy as hac
from scipy.cluster.hierarchy import fcluster

results = []
fileName = '/home/raghunandangupta/Downloads/soc_gen_data/train.csv'

#Transformation of input data to a diff matrix
data = []
for i in range(100):
    data.append([])
for i in range(100):
    with open(fileName) as csvfile:
        reader = csv.DictReader(csvfile)
        currentPrice = 100;
        for row in reader:
            header = 'X' + str(i + 1)
            diff = abs(((float(row[header]) - currentPrice) / currentPrice) * 100)
            data[i].append(diff)
            currentPrice = float(row[header])
df = pd.DataFrame(data)
result = open('/home/raghunandangupta/Downloads/soc_gen_data/abc.txt', 'w')

#prepare a correlation coefficient matrix using spearman
correlation_matrix = df.T.corr(method='spearman')

#prepare clusters using hierarchical cluster algo
Z = hac.linkage(correlation_matrix, 'complete')
k = 1.15
clusters = fcluster(Z, k, criterion='distance')
result.write("Asset,Cluster\n")
count = 1
for cluster in clusters:
    result.write("X%s," % str(count))
    result.write("%s\n" % cluster)
    count = count + 1
    result.flush()