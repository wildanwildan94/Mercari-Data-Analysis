# -*- coding: utf-8 -*-
## Load Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import re
import datetime
from collections import Counter
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import median_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from google.colab import files
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
import matplotlib.patches as mpatches
import re
plt.rcParams['figure.dpi'] = 200
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

from IPython.display import HTML, Math
display(HTML("<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/"
             "latest.js?config=default'></script>"))
Math(r"e^\alpha")



import unicodedata

def strip_accents(text):

    try:
        text = unicode(text, 'utf-8')
    except NameError: # unicode is a default on python 3 
        pass

    text = unicodedata.normalize('NFD', text)\
           .encode('ascii', 'ignore')\
           .decode("utf-8")

    return str(text)
    
    
## Load google drive content

from google.colab import drive
drive.mount('/content/drive')



## Load Mercari full data

full_data_d=pd.read_csv('/content/drive/My Drive/train_mercari.csv')
# Drop train_id attribute
try:
  full_data_d.drop("train_id", axis=1, inplace=True)
except:
  print "Already removed train_id"
    
# Add index to each products

index_product=range(full_data_d.shape[0])

full_data_d["index_product"]=index_product

## Split Full Data into training and test data

split_size=0.2
random_state=13
train_d, test_d=train_test_split(full_data_d, random_state=random_state,
                                test_size=0.2)



train_d.to_csv('train_data_s02_rs13.csv', index=False)
test_d.to_csv('test_data_s02_rs13.csv', index=False)

## Load transformed data
#train_d=pd.read_csv('/content/drive/My Drive/train_mercari.csv')
train_adj_d=pd.read_csv('/content/drive/My Drive/train_adj_s02_rs13.csv')
train_d=pd.read_csv('train_data_s02_rs13.csv')

test_adj_d=pd.read_csv('/content/drive/My Drive/test_adj_s02_rs13.csv')
test_d=pd.read_csv('test_data_s02_rs13.csv')

# Merge on index product to get price

train_adj_d=train_adj_d.merge(train_d[["index_product", "price"]], on="index_product")
test_full_adj_d=test_adj_d.merge(test_d[["index_product", "price"]], on="index_product")

random_state=19
test_adj_d, val_adj_d=train_test_split(test_full_adj_d, random_state=random_state,
                                        test_size=0.3)


print "---"
print "Shape of test"
print test_adj_d.shape
print "---"
print "Shape of val"
print val_adj_d.shape
print "---"
print "nan of test"
print test_adj_d.isna().sum()
print "---"
print val_adj_d.isna().sum()
print "--"



## Consider different variable transformations

cols=["name_weighted_price", "main_category_average_price", "sub_category_average_price",
     "subsub_category_average_price", "brand_average_price", "item_description_weighted_price"]








fig, ax = plt.subplots(len(cols), 3)

train_adj_d_small=train_adj_d
price_values=train_adj_d["price"]
for i in range(len(cols)):
  
  col_str=cols[i].replace("_"," ").title()
  ax[i,0].scatter(train_adj_d[cols[i]], price_values, facecolor="royalblue",
                 edgecolor="black")
 
  
  ax[i,1].scatter(train_adj_d[cols[i]], np.sqrt(price_values),
                 facecolor="royalblue",
                 edgecolor="black")
 
  
  
  ax[i,2].scatter(np.cbrt(train_adj_d[cols[i]]), np.cbrt(price_values),
                 facecolor="royalblue",
                 edgecolor="black")
 
  



col_titles=["No Transform", "Square Root", "Cubic Root"]

for axes, col in zip(ax[0], col_titles):
  axes.set_title(col)
  
for axes, row in zip(ax[:,2], cols):
  axes.set_ylabel(row.replace("_", " ").title(), rotation=0, size="large",
                 fontsize=13)
  axes.yaxis.set_label_coords(1.65, 0.5)
  
fig.subplots_adjust(right=1.3, wspace=0.3, hspace=0.3, bottom=-0.6, left=-0.3, top=1.2)

for axes in ax.flatten():
  axes.set_facecolor("navajowhite")
  
for axes in ax[:,0]:
  axes.set_ylabel("Price", fontsize=13)
for axes in ax[len(cols)-1,:]:
  axes.set_xlabel("Weighted or Average Price", fontsize=13)
  
fig.set_facecolor("floralwhite")

com_res="\n".join((r"$\cdot$ ""The cubic root transformation makes the different attributes a bit more linearly \n "\
                  "related with the price attribute",
                  r"$\cdot$ " "On the other hand, one can see that the large amount of data points (above 1 million) \n" \
                  "makes the relationships really crowded"))

fig.text(-0.1, -1, com_res, bbox=dict(boxstyle="round", edgecolor="black",
                                      facecolor="wheat"),
        fontsize=14)

fig.suptitle("Different transformations applied to data", y=1.3, fontsize=13)
