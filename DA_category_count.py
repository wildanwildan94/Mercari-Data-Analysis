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
  
  
  
## Analyze the distribution of category name

cat_d=train_d[["category_name"]].dropna()

print list(set(cat_d["category_name"]))

# (a) Count the amount of products in each category

cat_count_d=cat_d.groupby("category_name").agg({'category_name':'size'}).rename(columns={'category_name':'category_count'}).reset_index()

print "---"
print "Q: How many categories exists?"
print cat_count_d.shape[0]
print "---"

print "---"
print "Q: What are some examples of categories?"
print cat_count_d.head(5)
print "---"

# (b) Visualize the top 20 categories by count of amount of products in the 
# category

# Define top 20 categories by count
cat_count_top_d=cat_count_d.sort_values(by="category_count", ascending=False).head(20).sort_values(by="category_count", ascending=True)

# Define the amount of categories
amount_categories=cat_count_d.shape[0]

fig, ax = plt.subplots()


# Visualize top ten of categories by count
y_labels=range(cat_count_top_d.shape[0])

ax.barh(y_labels, cat_count_top_d["category_count"], facecolor="palevioletred",
       edgecolor="black")
ax.set_yticks(y_labels)
ax.set_yticklabels(cat_count_top_d["category_name"], fontsize=12)
ax.set_title("Top 20 categories by amount of products \n in category")
ax.set_facecolor("navajowhite")
fig.subplots_adjust(top=1, hspace=0.4)

fig.set_facecolor("floralwhite")
com_res="\n".join((r"$\cdot$ " "By the structure of the category name \n" \
                  "we can see that each category is divided into \n" \
                  "several subcategories", 
                  r"$\cdot$ " "For example, the category Women can be \n" \
                  "divided into several subcategories, e.g. \n" \
                  "Underwear, Athletic Apparel, Shoes, Dresses, etc.",
                  r"$\cdot$ " "The subcategories implies that one could \n" \
                  "consider an analysis of the subcategories, instead \n" \
                  "of analyzing the categories",
                 r"$\cdot$ " "An idea is that different subcategories \n" \
                  "from different categories might have similar \n" \
                  "dependence on price, e.g. Women/Tops might \n" \
                  "behave similarly to Men/Tops",
                  r"$\cdot$ " "Lastly, the high amount of categories \n" \
                  "makes it preferable to, perhaps, reduce the \n" \
                  "amount of categories considered, especially if multiple \n"\
                  "categories share similar information"))
com_depthcat="\n".join((r"$\cdot$ " "It follows that the depth, i.e. how many times we can subdivide a main category, \n" \
                       "ranges from 3 to 5, depending on what subcategories are chosen in the subdivision",
                       r"$\cdot$ " "Hence, to describe the underlying product as much as possible, \n" \
                       "one should subdivide the main category as many times as possible, and then consider the \n" \
                       "the name of the subcategory at the maximum depth",
                       r"$\cdot$ " " For example, for 'Electronics/Video Games & Consoles/Games', the name of the last \n" \
                       "subcategory 'Games' should describe the underlying product the best - of course, the preceding \n" \
                       "subcategories are also important in describing the underlying product"))

for x_val, y in zip(cat_count_top_d["category_count"], y_labels):
  ax.text(x_val+100, y-0.3, str(x_val), fontweight="bold")
  
box=dict(boxstyle="round", edgecolor="black", facecolor="wheat")
fig.text(0.92,0.1, com_res, fontsize=13, bbox=box)
fig.text(0.3, -0.38, com_depthcat, fontsize=13, bbox=box)
ax.set_xlim((0, 68000))

ax.text(4e04, 10, "Amount of Categories: \n%s"%str(amount_categories),
       bbox=dict(boxstyle="round", edgecolor="black", facecolor="white"))

count_patch=mpatches.Patch(color="black", label="Amount of Products \nin Category",
                          facecolor="white")
ax.legend(handles=[count_patch])




 
