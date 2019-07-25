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
    

## Analyze the distribution of the item condition

item_cond_d=train_d["item_condition_id"]
# (a) Compute the count of item condtion

count_item_condition=[]
item_condition=[]
for a, b in train_d.groupby("item_condition_id"):
  temp_count_condition=b.shape[0]
  temp_condition=a
  
  count_item_condition.append(temp_count_condition)
  item_condition.append(temp_condition)
  
total_items_count=sum(count_item_condition)  
# (b) Sort the count of item condition and item condition

#[x for _,x in sorted(zip(Y,X))]

count_item_condition=[x for _,x in sorted(zip(item_condition, count_item_condition))]
item_condition=[x for x,_ in sorted(zip(item_condition, count_item_condition))]

# (c) Visualize the item condition and counts

fig, ax = plt.subplots()
y_labels=range(len(item_condition))
ax.barh(y_labels, count_item_condition, edgecolor="black",
       facecolor="lightcoral")
ax.set_yticks(y_labels)
ax.set_yticklabels(item_condition)

for x_val, y in zip(count_item_condition, y_labels):
  ax.text(x_val+4e03, y, str(np.round(x_val/float(total_items_count)*100, 1))+" %", color="black", fontweight="bold")
  
count_patch=mpatches.Patch(color="black", label="Percentage of Products \nwith Item Condition")
ax.set_xlabel("Amount of Products")
ax.set_ylabel("Item Condition")
ax.set_title("Amount of Products in each Item Condition")
ax.set_xlim((0, 7.5e5))
ax.set_facecolor("navajowhite")
ax.legend(handles=[count_patch], bbox_to_anchor=(1,0.8),
         facecolor="lavender",
          edgecolor="black")
fig.set_facecolor("floralwhite")

com_itemcond="\n".join((r"$\cdot$ " "1: New",
                       r"$\cdot$ " "2: Almost New",
                       r"$\cdot$ " "3: Good",
                       r"$\cdot$ " "4: Fair",
                       r"$\cdot$ " "5: Poor"))
com_res="\n".join((r"$\cdot$ " "Most products are New, followed by \n" \
                  "products in Good condition and products in \n"\
                   "Almost New condition",
                  r"$\cdot$ " "A low percentage of products are either \n" \
                  "Fair or Poor, an indication that most people \n" \
                  "don't bother to post products in bad conditions",
                  r"$\cdot$ " "A possible explanation is that most people \n" \
                  "tend to sell recently bought item, by e.g. \n" \
                  "regret or some other reason",
                  r"$\cdot$ " "It may also indicate that buyers \n" \
                  "are mostly interested in products that are \n" \
                  "relatively new, and generally don't bother \n" \
                  "buying products with a low condition, because \n" \
                  "of e.g. less of a status symbol having low \n" \
                  "condition products"))
box=dict(boxstyle="round", edgecolor="black", facecolor="wheat")
fig.text(0.92, 0.67, com_itemcond, bbox=box)
fig.text(0.92,0.03, com_res, bbox=box)


  
