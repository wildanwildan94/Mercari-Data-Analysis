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
 
## Visualize the distribution of shipping

shipping_d=train_d[["shipping"]].dropna()

# (a) Count the shipping attribute by value

#brand_count_d=brand_d.groupby("brand_name").agg({'brand_name':'size'}).rename(columns={'brand_name':'brand_count'}).reset_index()
ship_count_d=shipping_d.groupby("shipping").agg({'shipping':'size'}).rename(columns={'shipping':'shipping_count'}).reset_index()

print ship_count_d.head(3)
# (b) Visualize count of shipping

fig, ax = plt.subplots()
x_labels=range(ship_count_d.shape[0])
ax.bar(x_labels, ship_count_d["shipping_count"], facecolor="palevioletred",
       edgecolor="black")
ax.set_xticks(x_labels)
ax.set_xticklabels(["Shipping Fee Paid by \n Seller", "Shipping Fee Paid by \n Buyer"],
                  fontsize=13)

ax.set_facecolor("navajowhite")
fig.set_facecolor("floralwhite")
ax.set_title("Count of who is paying for the shipping fee", fontsize=13)
ax.set_ylabel("Count")

com_res="\n".join((r"$\cdot$ " "The shipping fee is usually paid \n" \
                  "by the seller, but not by a wide margin",
                  r"$\cdot$ " "The higher percentage for the shipping fee \n" \
                  "paid by the seller can indicate that there is \n" \
                  "a higher supply than demand, such that \n" \
                  "the sellers are forced to buy shipping fee \n" \
                  "more often to, perhaps, make their product listings \n" \
                  "more attractive to potential buyers",
                  r"$\cdot$ " "Another possibility is that the seller \n" \
                  "paying for shipping might be more attractive \n"\
                  "to potential buyers - it signals a burden \n" \
                  "(both practical and economical) left to the seller"))
fig.text(0.92, 0.25, com_res, bbox=dict(boxstyle="round", edgecolor="black",
                                      facecolor="wheat"),
        fontsize=13)
for x, y_val in zip(x_labels, ship_count_d["shipping_count"]):
  ax.text(x-0.1, y_val+30000, str(int(y_val/float(shipping_d.shape[0])*100)) +"%", fontweight="bold",
         fontsize=13)
  
  
ax.set_ylim((0, 970000))
