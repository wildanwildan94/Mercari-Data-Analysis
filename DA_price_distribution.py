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
 
## Visualize the distribution of price

price_d=train_d[["price"]]

# (a) Print summary statistics
price_description=price_d.describe()
print price_description

# (b) Summary statistics of price
price_median=int(np.median(price_d["price"]))
price_mean=int(np.mean(price_d["price"]))
price_std=int(np.std(price_d["price"]))

# (c) Histogram of price attribute, with values above 75 removed

quantile_price=np.quantile(price_d["price"],0.95)
price_reduced=[x for x in price_d["price"] if x<quantile_price]



fig, ax = plt.subplots()

ax.hist(price_reduced, bins=20, facecolor="palevioletred",
       edgecolor="black")
ax.set_xlabel("Price ($USD)", fontsize=13)
ax.set_ylabel("Count", fontsize=13)
ax.set_title("Histogram of price attribute \n for price less than the 95%% quantile of price ($%sUSD)"%int(quantile_price))
ax.set_facecolor("navajowhite")

com_stat="\n".join((r"$\cdot$ " "Mean: %s"%price_mean,
                  r"$\cdot$ " "Median: %s"%price_median,
                  r"$\cdot$ " "Standard Deviation: %s"%price_std))

ax.text(30, 150000, com_stat, bbox=dict(boxstyle="round", edgecolor="black",
                                    facecolor="white"),
       fontsize=14)

com_res="\n".join((r"$\cdot$ " "A major part of products are bought at \n" \
                  "around 5-25 $USD - indicates that people \n" \
                  "predominantly buy low-cost products",
                  r"$\cdot$ " "For increasing prices, the amount of products \n" \
                  "bought decreases almost exponentially - while \n" \
                  "high-cost items are bought, they are bought \n" \
                  "significantly less frequent compared to low-cost \n" \
                  "products",
                   "The 5-25$USD region could correspond to \n" \
                  "clothes, accessories, and other low-cost products",
                  r"$\cdot$ " "It is possible that the high frequency \n" \
                  "of low-cost items bought might mean that consumers \n" \
                  "are less confident in buying high-cost products - \n" \
                  "since high-cost products might be more risky to buy \n" \
                  "second-hand rather than first-hand"))
fig.text(0.92, 0.14, com_res, bbox=dict(boxstyle="round", edgecolor="black",
                                   facecolor="wheat"),
        fontsize=13)
fig.set_facecolor("floralwhite")
