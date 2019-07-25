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
 
 
## Visualize average price for different brands

brand_price_d=train_d[["brand_name", "price"]]

# (a) Calculate the average price for each brand name

brand_avgprice_d=brand_price_d.groupby("brand_name").agg({'price':'mean'}).rename(columns={'price':'avgprice'}).reset_index()

print "---"
print "Q: How does brand_avgprice_d look like?"
print brand_avgprice_d.iloc[0]
print "---"

# (b) Consider different price ranges

A_lower=5
A_upper=20
B_lower=20
B_upper=60
C_lower=60
C_upper=200
D_lower=200

random_state=130
brand_avgprice_A_d=brand_avgprice_d.query("avgprice>%s and avgprice<%s"%(A_lower, A_upper)).sample(20, random_state=random_state).sort_values(by="avgprice", ascending=True)
brand_avgprice_B_d=brand_avgprice_d.query("avgprice>%s and avgprice<%s"%(B_lower, B_upper)).sample(20, random_state=random_state).sort_values(by="avgprice", ascending=True)
brand_avgprice_C_d=brand_avgprice_d.query("avgprice>%s and avgprice<%s"%(C_lower, C_upper)).sample(20, random_state=random_state).sort_values(by="avgprice", ascending=True)
brand_avgprice_D_d=brand_avgprice_d.query("avgprice>%s"%(D_lower)).sample(20, random_state=random_state).sort_values(by="avgprice", ascending=True)




fig, ax = plt.subplots(1,4)


# Visualize top ten of categories by count
y_labels=range(brand_avgprice_A_d.shape[0])

ax[0].barh(y_labels, brand_avgprice_A_d["avgprice"], facecolor="palevioletred",
          edgecolor="black")
ax[0].set_yticks(y_labels)
ax[0].set_yticklabels([x.title() for x in brand_avgprice_A_d["brand_name"]], fontsize=12)
ax[0].set_title("Price Range: %s to %s ($USD)"%(A_lower, A_upper))

ax[1].barh(y_labels, brand_avgprice_B_d["avgprice"], facecolor="palevioletred",
          edgecolor="black")
ax[1].set_yticks(y_labels)
ax[1].set_yticklabels([x.title() for x in brand_avgprice_B_d["brand_name"]], fontsize=12)
ax[1].set_title("Price Range: %s to %s ($USD)"%(B_lower, B_upper))

ax[2].barh(y_labels, brand_avgprice_C_d["avgprice"], facecolor="palevioletred",
          edgecolor="black")
ax[2].set_yticks(y_labels)
ax[2].set_yticklabels([x.title() for x in brand_avgprice_C_d["brand_name"]], fontsize=12)
ax[2].set_title("Price Range: %s to %s ($USD)"%(C_lower, C_upper))


ax[3].barh(y_labels, brand_avgprice_D_d["avgprice"], facecolor="palevioletred",
          edgecolor="black")
ax[3].set_yticks(y_labels)
ax[3].set_yticklabels([x.title() for x in brand_avgprice_D_d["brand_name"]], fontsize=12)
ax[3].set_title("Price Range: %s and above ($USD)"%(D_lower))
fig.subplots_adjust(wspace=1.33, right=1.5, top=1,bottom=0.1, left=-0.3)
fig.suptitle("Average of Prices Related to a Brand Name \n 20 Randomly Picked Brand Names for Each Price Range", y=1.15)
for axes in ax.flatten():
  axes.set_facecolor("navajowhite")
  axes.set_xlabel("Average Price ($USD)", fontsize=13)
  
  
  
fig.set_facecolor("floralwhite")


com_A_pricerange="\n".join((r"$\cdot$ " "A lot of known brands in all price ranges, including brands like Asos, \n" \
                           "David Yurman, Louis Vuitton, Saint Laurent, Alexander Wang",
                           r"$\cdot$ " "The average price of products associated with a brand can give a rough \n" \
                           "idea of the price-class associated with the brand - some may be expensive like Louis Vuitton \n" \
                            "while others may be more affordable like Pier One",
                           r"$\cdot$ " "In addition, certain brands will have a similar price-class, and will probably \n" \
                           "sell products for similar prices"))

com_alg="\n".join((r"$\cdot$ " "Construction of the Average Price:", 
                  " For each possible Brand Name:",
                  "    Collect all products associated with that Brand Name",
                  "    Calculate the average price of all those products"))

box=dict(boxstyle="round", edgecolor="black", facecolor="wheat")
fig.text(-0.6, -0.42, com_A_pricerange, bbox=box,fontsize=13)
fig.text(0.84, -0.31, com_alg, bbox=box, fontsize=13)
