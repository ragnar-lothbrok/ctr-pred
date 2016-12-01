import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
import operator
import os
import pandas
import xgboost as xgb

dictionary  ={'f1':'originalPrice','f10':'CLP','f11':'PDP','f12':'searchKeyword','f13':'activeProductCategory','f14':'activeSellerCategory','f15':'sellerRatingSdPlus','f16':'sellerRatingNonSdPlus','f17':'supcBrand','f19':'supcCreatedTime','f2':'price','f20':'accId','f21':'adSpaceType','f22':'SIMILAR_AD','f23':'SEARCH_AD','f24':'RETARGATED_AD','f25':'amountSpent','f26':'searchCategory','f27':'searchRelevancyScore','f28':'adSpaceId','f29':'supcCat','f3':'reviewCount','f30':'pageCategory','f31':'keyUserDeviceId','f32':'wiRatingCount','f33':'itemPogId','f34':'wpPercentageOff','f35':'eventKey','f36':'pogId','f37':'displayName','f38':'rating','f39':'ratingCount','f4':'position','f40':'sellerCode','f41':'dpDay','f42':'dpHour','f43':'osVersion','f44':'platformType','f45':'browserDetails','f46':'email','f47':'pincode','f48':'guid','f49':'widgetId','f5':'trackerId','f6':'WEB','f7':'WAP','f8':'APP','f9':'SLP'}
folderName = "/home/raghunandangupta/Desktop/books/mlm/"
featureNamesFile='/home/raghunandangupta/Desktop/books/images/xgb.fmap'
imageFolderName = '/home/raghunandangupta/Desktop/books/images/'

file_list_tmp = os.listdir(folderName)
fulldir = True
file_list = []
suffix=""
if fulldir:
    for item in file_list_tmp:
        if item.endswith(suffix):
            fileName = os.path.join(folderName, item)
            imageFileName =  fileName[fileName.rfind('/')+1: ].replace("model", "feature_importance_xgb")
            imageFileName = imageFolderName + imageFileName + ".png"
            bst = xgb.Booster(model_file=os.path.join(folderName, item))
            importance = bst.get_fscore(fmap=featureNamesFile)
            if importance.items().__len__() > 0 :
                file_list.append(imageFileName)
                importance = sorted(importance.items(), key=operator.itemgetter(1))
                featureTuples = []
                for typ in importance:
                    featureTuples.append((dictionary.get(typ[0]),typ[1]))
                print featureTuples
                df = pandas.DataFrame(featureTuples, columns=['feature', 'fscore'])
                df['fscore'] = df['fscore'] / df['fscore'].sum()
                plt.figure()
                df.plot()
                df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(20, 14))
                plt.title('XGBoost Feature Importance')
                plt.xlabel('relative importance')
                print imageFileName
                plt.gcf().savefig(imageFileName)
print "All : "+str(file_list)
