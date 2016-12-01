import matplotlib
from xgboost import plot_tree
matplotlib.use('Agg')
from matplotlib import pyplot
matplotlib.use('Agg')
import xgboost as xgb
import os

folderName = "/tmp/click_impression_20161115/model/"
imageFolderName = '/tmp/click_impression_20161115/trees/'

file_list_tmp = os.listdir(folderName)
fulldir = True
file_list = []
suffix = ""
if fulldir:
    for item in file_list_tmp:
        if "model" in item:
            try:
                fileName = os.path.join(folderName, item)
                imageFileName = fileName[fileName.rfind('/') + 1: ].replace("model", "tree")
                imageFileName = imageFolderName + imageFileName+".png"
                print fileName
                bst = xgb.Booster(model_file=fileName)
                print imageFileName
                plot_tree(bst, num_trees=5)
                pyplot.savefig(imageFileName, format='png', dpi=2000)
            except:
                print "Exception occured"
