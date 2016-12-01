#sudo apt-get install python-pydot
#wget http://effbot.org/downloads/Imaging-1.1.6.tar.gz
#tar xvf Imaging-1.1.6.tar.gz 
#cd Imaging-1.1.6
#python setup.py install
#pip install pillow
#pip install --upgrade pip


import matplotlib
from xgboost import plot_tree
matplotlib.use('Agg')
from matplotlib import pyplot
matplotlib.use('Agg')
import xgboost as xgb

bst = xgb.Booster(model_file='/home/raghunandangupta/Desktop/books/models/xgb.model')
plot_tree(bst, num_trees=5)
pyplot.savefig('latest_model.png', format='png', dpi=2000)