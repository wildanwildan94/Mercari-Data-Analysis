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
    

## Analyze initial form of Mercari training data

# (a) Iterate through all attributes and store a generic form of that
# attribute

train_cols=train_d.columns
train_cols_values=[]
com_attr_and_values=r"$\cdot$ " "Generic Values of Each Attribute:"
for col in train_cols:
  temp_col=train_d[col].dropna()
  temp_val=temp_col.iloc[10]
  train_cols_values.append(temp_val)
  temp_com_attr_val="%s: "%str(col).replace("_"," ").title() +"%s"%temp_val
  com_attr_and_values=com_attr_and_values+"\n"+temp_com_attr_val
  
  
# (b) Visualize generic form of each attribute

fig, ax = plt.subplots()
box=dict(boxstyle="round", edgecolor="black", facecolor="wheat")
fig.text(0, 0.3, com_attr_and_values, bbox=box, fontsize=16)
fig.set_facecolor("floralwhite")


com_data="\n".join((r"$\cdot$ " "The Name is a typical, brief description of the the product in question ",
                   r"$\cdot$ " "The Item Condition is a number representing the condition of the product in question ",
                   r"$\cdot$ " "The Category Name represents the category of the product ",
                   r"$\cdot$ " "The Brand Name is simply the brand of the underlying product, e.g. Nike ",
                   r"$\cdot$ " "The Price is the price the product as sold for, in the unit USD ",
                   r"$\cdot$ ""The Shipping is 1 if the shipping fee is paid by the seller, and 0 if it is paid by the buyer"))
fig.text(0, -0.13, com_data, bbox=box, fontsize=14)
ax.axis("off")

