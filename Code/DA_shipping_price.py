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
 
 
## Visualize the relationship between shipping and price

ship_price_d=train_d[["shipping", "price"]]


# (a) Compute the average price of each shipping possibilities

ship_avgprice_d=ship_price_d.groupby("shipping").agg({'price':'mean'}).rename(columns={'price':'avgprice'}).reset_index().sort_values(by="shipping")
ship_avgprice_d["ship_str"]=["Shipping Fee Paid by Buyer", "Shipping Fee Paid by Seller"]


print "---"
print "Q: How doe ship_avgprice_d look like?"
print ship_avgprice_d.iloc[0]
print "---"


# (b) Visualize the average price for each shipping possibility

fig, ax = plt.subplots()

x_labels=range(ship_avgprice_d.shape[0])

ax.bar(x_labels, ship_avgprice_d["avgprice"], facecolor="palevioletred",
      edgecolor="black")

ax.set_xticks(x_labels)
ax.set_xticklabels(ship_avgprice_d["ship_str"], fontsize=13)
ax.set_ylabel("Average Price ($USD)")
ax.set_facecolor("navajowhite")
ax.set_title("Average Price of Products with Shipping Type")
fig.set_facecolor("floralwhite")

fig.subplots_adjust(right=1.2)


for x, y in zip(ship_avgprice_d["shipping"], ship_avgprice_d["avgprice"]):
  ax.text(x-0.1, y+1, str(int(y))+" ($USD)", fontsize=12, fontweight="bold")
  
  
avgprice_patch=mpatches.Patch(color="black", label="Average Price")
ax.legend(handles=[avgprice_patch], fontsize=13)
ax.set_ylim((0, 35))

com_res="\n".join((r"$\cdot$ " "There is a small difference in the average price \n" \
                  "between the two different shipping types",
                  r"$\cdot$ " "The difference is relatively small, but indicates \n" \
                  "that products for which the shipping is paid by \n" \
                  "the buyer is a little bit more expensive",
                  r"$\cdot$ " "It should be noted that the Price attribute \n" \
                  "does not include the shipping fee price paid by, \n" \
                  "when that is the case, a potential buyer"))

fig.text(1.22, 0.41, com_res, bbox=dict(boxstyle="round", edgecolor="black",
                                     facecolor="wheat"),
        fontsize=13)
