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
 
## Visualize the distribution of brand name


brand_d=train_d[["brand_name"]].dropna()


# (a) Count the amount of products in each brand
brand_count_d=brand_d.groupby("brand_name").agg({'brand_name':'size'}).rename(columns={'brand_name':'brand_count'}).reset_index()


print "---"
print "Q: How many brands are there?"
print brand_count_d.shape[0]
print "---"


# (b) Visualize the top 20 brand by count of amount of products in the 
# brand

# Define top 20 brand by count
brand_count_top_d=brand_count_d.sort_values(by="brand_count", ascending=False).head(20).sort_values(by="brand_count", ascending=True)

# Define the amount of categories
amount_brands=brand_count_d.shape[0]

fig, ax = plt.subplots()


# Visualize top ten of categories by count
y_labels=range(brand_count_top_d.shape[0])

ax.barh(y_labels, brand_count_top_d["brand_count"], facecolor="palevioletred",
       edgecolor="black")
ax.set_yticks(y_labels)
ax.set_yticklabels(brand_count_top_d["brand_name"], fontsize=12)
ax.set_title("Top 20 brands by amount of products \n in brand")
ax.set_facecolor("navajowhite")
fig.subplots_adjust(top=1, hspace=0.4)

fig.set_facecolor("floralwhite")
com_res="\n".join((r"$\cdot$ " "A lot of popular brands are present, \n" \
                  "like Nike, Apple, Michael Kors, Nintendo, Disney",
                   r"$\cdot$ " "A total of 4809 brands indicates that \n" \
                   "it might be a good idea to analyze the average \n" \
                   "price of products in brands - to quantitatively compare \n" \
                   "the price-class of different brands, and perhaps relate \n" \
                   "certain brands to each other",
                   r"$\cdot$ " "The ubiquity of popular brands is not a surprise, \n" \
                   "as there should naturally be a high demand for such products",
                   r"$\cdot$ " "Another observation is how Nike is much more popular \n" \
                   "than Adidas, which is a competitor. Indicating that Nike products \n" \
                   "is to go-to sports brand for most consumers"))


for x_val, y in zip(brand_count_top_d["brand_count"], y_labels):
  ax.text(x_val+100, y-0.3, str(x_val), fontweight="bold")
  
box=dict(boxstyle="round", edgecolor="black", facecolor="wheat")
fig.text(0.92,0.2, com_res, fontsize=13, bbox=box)
ax.set_xlim((0, 68000))

ax.text(4e04, 10, "Amount of Brands: \n%s"%str(amount_brands),
       bbox=dict(boxstyle="round", edgecolor="black", facecolor="white"))
count_patch=mpatches.Patch(color="black", label="Amount of Products \nin Brand",
                          facecolor="white")
ax.legend(handles=[count_patch])
