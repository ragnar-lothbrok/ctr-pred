import csv
from scipy.cluster.vq import kmeans, vq

results = []
fileName = '/home/raghunandangupta/Downloads/soc_gen_data/train.csv'

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
            data[i].append(diff **(1/2.0))
            currentPrice = float(row[header])

centroids,_ = kmeans(data,12)
# assign each sample to a cluster
idx,_ = vq(data,centroids)

result = open('/home/raghunandangupta/Downloads/soc_gen_data/abc.txt', 'w')
result.write("Asset,Cluster\n")
count = 1
for cluster in idx:
    result.write("X%s," % str(count))
    result.write("%s\n" % cluster)
    count = count + 1
    result.flush()