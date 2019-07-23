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
## Cook's Distance of Data

train_adj_cols=train_adj_d.columns.tolist()
print train_adj_cols
train_adj_cols.remove("index_product")
train_adj_cols.remove("item_condition_id_5.0")
train_adj_cols.remove("shipping_nan")
train_adj_cols.remove("shipping_0.0")

X_train_df=train_adj_d[train_adj_cols].drop("price", axis=1)
X_train=X_train_df.as_matrix()
y_train=train_adj_d["price"].as_matrix()

X_train_ones=sm.add_constant(X_train)

model=sm.OLS(y_train, X_train_ones)
res=model.fit()

#print res.summary()


# (b) Define Cook's Distance
model_cooks = np.array(res.get_influence().cooks_distance[0])

# (c) Define Cook's Distance cutoff value: (1) 3*mean of Cook's Distance

threemean_cutoff=3*np.mean(model_cooks)


print "---"
print "Q: What is the value of 3*mean of Cook's Distance"
print threemean_cutoff
print "---"

# (d) Compute all indices where Cook's distance is above threemean_cutoff
cook_cutoff_threem_exc_id=np.argwhere(model_cooks>threemean_cutoff)[:,0]


print "---"
print "Q: How many points exceeds the 3*Mean of Cook's Distance cutoff?"
print str(len(cook_cutoff_threem_exc_id)) + " out of a total of %s points"%len(model_cooks)
print "---"

perc_above_cutoff=np.round(len(cook_cutoff_threem_exc_id)/float(len(model_cooks)), 2)*100
perc_below_cutoff=np.round((len(model_cooks)-len(cook_cutoff_threem_exc_id))/float(len(model_cooks)), 2)*100


fig_cook, ax_cook=plt.subplots()
x_labels=np.array(range(model_cooks.shape[0]))

ax_cook.plot(x_labels, model_cooks, linestyle="None", marker="o",
             markerfacecolor="cornflowerblue", markeredgecolor="black")
ax_cook.plot(x_labels[cook_cutoff_threem_exc_id], model_cooks[cook_cutoff_threem_exc_id],
            linestyle="None", marker="o", markerfacecolor="crimson", markeredgecolor="black")
ax_cook.plot()
ax_cook.plot(x_labels, [threemean_cutoff]*model_cooks.shape[0], color="black", lw=2)
ax_cook.set_facecolor("navajowhite")
ax_cook.set_xlabel("Index")
ax_cook.set_ylabel("Cook's Distance")

cutoff_patch=mpatches.Patch(label="Cutoff Line Cook's Distance",
                           color="black")
above_cutoff_patch=mpatches.Patch(label="Points above cutoff line: %s %%"%perc_above_cutoff, 
                                 color="crimson")
below_cutoff_patch=mpatches.Patch(label="Points below the cutoff line:  %s %%"%perc_below_cutoff,
                                 color="blue")

fig_cook.legend(handles=[cutoff_patch, above_cutoff_patch, below_cutoff_patch],
         bbox_to_anchor=(0.8, 0.9))
fig_cook.set_facecolor("floralwhite")

com_res="\n".join((r"$\cdot$ " "The 1 % of points above \n" \
                  "the cutoff line indicates extreme values \n" \
                  "with great influence on the linear model",
                 r"$\cdot$ " "One possible remedy is to remove \n" \
                 "these points from the linear model",
                 r"$\cdot$ " "The influential points are most \n" \
                 "likely products with a high price, which may distort \n" \
                 "the prediction of price of low-moderate \n" \
                 "price products - which are more frequent in the  \n" \
                 "dataset",
                 r"$\cdot$ " "If we want our model to perform better for \n" \
                 "products with low-moderate prices, then the 1% of \n" \
                 "points above the cutoff line should be discarded \n" \
                 "when building the model",
                  r"$\cdot$ " "Because they are only 1 %, it makes \n" \
                  "sense to remove these extreme values. However, if one \n" \
                  "wishes to retain the ability of prediciting high price \n" \
                  "items, one should probably keep the 1 % of points"))

fig_cook.text(0.92,0.15, com_res,
             bbox=dict(boxstyle="round", edgecolor="black",
                      facecolor="wheat"))
fig_cook.suptitle("Cook's Distance of Fitted Values")
