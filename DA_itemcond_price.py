# -*- coding: utf-8 -*-
## Load Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import re
import datetime
from collections import Counter
from google.colab import files
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


## Load Mercari training data
train_d=pd.read_csv('/content/drive/My Drive/train_mercari.csv')
# Drop train_id attribute
try:
  train_d.drop("train_id", axis=1, inplace=True)
except:
  print "Already removed train_id"
 
## Visualize average price for item conditions

itemcond_price_d=train_d[["item_condition_id", "price"]]


# (a) Compute average price of item_condition

itemcond_avgprice_d=itemcond_price_d.groupby("item_condition_id").agg({'price':'mean'}).rename(columns={'price':'avgprice'}).reset_index()


print "---"
print "Q: How does itemcond_avgprice_d look like?"
print itemcond_avgprice_d.iloc[0]
print "---"


# (b) Merge item_conditon_id with conditon name

itemcond_name_d=pd.DataFrame({'item_condition_id':[1,2,3,4,5],
                             'item_condition_name':["New", "Almost New",
                                                   "Good", "Fair", "Poor"]})

itemcondname_avgprice_d=itemcond_avgprice_d.merge(itemcond_name_d,
                                                  on="item_condition_id", how="left").sort_values(by="item_condition_id")


print "---"
print "Q: How does itemcondname_avgprice_d look like?"
print itemcondname_avgprice_d.iloc[0]
print "--"


# (c) Visualize the average price of item condition

y_labels=itemcondname_avgprice_d["item_condition_id"]

fig, ax = plt.subplots()


ax.barh(y_labels, itemcondname_avgprice_d["avgprice"], facecolor="palevioletred",
       edgecolor="black")
ax.set_yticks(y_labels)
ax.set_yticklabels(itemcondname_avgprice_d["item_condition_name"], fontsize=13)
ax.set_facecolor("navajowhite")
ax.set_xlabel("Average Price ($USD)", fontsize=13)
ax.set_title("Average Price of Products With a Item Condition", fontsize=13)

fig.set_facecolor("floralwhite")

com_res="\n".join((r"$\cdot$ " "No particular pattern in the average price \n" \
                  "for different item conditions, as they are all \n" \
                  "pretty close to each other",
                  r"$\cdot$ " "This suggest that it may be preferable to \n" \
                  "consider the item condition as a category when \n" \
                  "modelling the relationship between price and \n" \
                  "products"))
com_alg="\n".join((r"$\cdot$ " "For all possible Item Condition:",
                  "   Collect all products with that Item Condition",
                  "   Calculate the average price of all those products"))

box=dict(boxstyle="round", edgecolor="black",
        facecolor="wheat")
fig.text(0.92, 0.52, com_res, bbox=box,
         fontsize=13)
fig.text(0.92, 0.3, com_alg, bbox=box, fontsize=13)

