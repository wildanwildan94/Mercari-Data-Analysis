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


### Visualize Correlation Matrix for Training Data (cubic root)

# (a) Visualize heatmap for correlation
# How does train_split_drop_sqrt_adj_d


train_adj_cbrt_d=train_adj_d.drop(["index_product", "item_condition_id_nan", "shipping_nan"], axis=1).apply(np.cbrt)
train_adj_cbrt_cols=train_adj_cbrt_d.columns

fig, ax = plt.subplots()

sns.heatmap(train_adj_cbrt_d.corr(),
           xticklabels=[x.replace("_"," ").title() for x in train_adj_cbrt_cols],
           yticklabels=[x.replace("_"," ").title() for x in train_adj_cbrt_cols],
           cmap="Blues",
           annot=True,
           linecolor="black",
           linewidths=1.2,
           cbar_kws={'label':'Correlation'},
           ax=ax)

fig.subplots_adjust(right=0.7, left=-0.7, bottom=-0.7, top=0.7)
ax.set_title("Correlation Between Different Attributes")
ax.tick_params(left=False, bottom=False)
fig.set_facecolor("floralwhite")



com_res="\n".join((r"$\cdot$ " "Price is highly correlated with all non-categorical \n" \
                  "attributes, except perhaps the main category average price",
                  r"$\cdot$ " "The nice relation between price and other attributes \n" \
                  "might be a sign that our attributes may be able to explain the  \n" \
                  "variation in price, especially when considered together",
                  r"$\cdot$ " "Another interesting observation is that the name \n" \
                   "weighted price is highly correlated with the average prices from \n"\
                   "the main, sub and subsub categories",
                  r"$\cdot$ " "Hence, there might be a relationship between the name \n" \
                  "of a product and what category it is placed in, i.e. the name conveys \n" \
                   "information about what type of product is considered",
                  r"$\cdot$ " "In contrast, the item description weighted price is not as \n"\
                  "well correlated as the name attribute. Might imply that the words \n" \
                  "in the name of a product gives a better description of what typ of \n"\
                  "product it may be. The name will most likely convey a more 'direct' \n" \
                  "message to the consumer about the product than item description",
                  r"$\cdot$ " "The name is also highly correlated with brand, which might \n"\
                  "indicate that the brand name is frequently used in the name of \n" \
                  "the product, which makes sense - if you are selling a Macbook Pro \n" \
                  "you will most likely put the words 'Macbook' and 'Pro', in addition \n"\
                  "to other words, in the name"))

fig.text(0.67, -0.55, com_res, bbox=dict(boxstyle="round",
                                     edgecolor="black",
                                     facecolor="wheat"),
        fontsize=12, family="serif")
        


