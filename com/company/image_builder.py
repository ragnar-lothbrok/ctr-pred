import xgboost as xgb
import time as time
from xgboost import plot_importance,plot_tree
from matplotlib import pyplot
import Image

bst = xgb.Booster(model_file='/tmp/xgb.model')
#plot_importance(bst)
plot_tree(bst, num_trees=5)
#pyplot.show()
#pyplot.figure(figsize=(100, 100))
pyplot.savefig('/tmp/latest_model.png', format='png', dpi=2000)
#pyplot.savefig('/home/anujjalan/learning/deploy/tree.png')
#Image.open('/home/anujjalan/learning/deploy/tree.png').save('testplot.jpg','JPEG')