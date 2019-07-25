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
 
 
## Analyze of the different subcategories of each category

cat_d=train_d[["category_name"]].dropna()

print list(set(cat_d["category_name"]))

# (a) Count the amount of products in each category

cat_count_d=cat_d.groupby("category_name").agg({'category_name':'size'}).rename(columns={'category_name':'category_count'}).reset_index()

# (b) Example category name

ex_cat_name=cat_count_d.iloc[0]["category_name"]
print "---"
print "Q: What is a generic category?"
print ex_cat_name
print "---"

ex_subcat=[]

ex_subcat.extend(ex_cat_name.split("/"))
print "---"
print "Q: What is generic subcategories?"
print ex_subcat
print "--"

# (c) Want to compute the amount of main categories
category_subcategories_count=Counter([])
category_products_count=Counter([])

for index, row in cat_count_d.iterrows():
  # The main category is the first element in the split
  temp_main_cat=row["category_name"].split("/")[0]
  temp_main_cat_count=row["category_count"]
  category_subcategories_count[temp_main_cat]+=1
  category_products_count[temp_main_cat]+=temp_main_cat_count
  
  
  
maincat_subcat_name, maincat_subcat_count=zip(*category_subcategories_count.items())
maincat_products_name, maincat_products_count=zip(*category_products_count.items())

maincat_subcat_d=pd.DataFrame({'maincategory_name':maincat_subcat_name,
                       'maincategory_count':maincat_subcat_count})
maincat_products_d=pd.DataFrame({'maincategory_name':maincat_products_name,
                               'maincategory_products_count':maincat_products_count})
maincat_subcat_products_d=maincat_products_d.merge(maincat_subcat_d, on="maincategory_name", how="left")

maincat_subcat_products_d.sort_values(by="maincategory_count", ascending=True, inplace=True)



amount_maincategories=len(maincat_subcat_name)
amount_products=sum(maincat_products_d["maincategory_products_count"])

# (d) Visualize the main categories and count

fig, ax = plt.subplots()


# Visualize top ten of categories by count
y_labels=range(maincat_subcat_products_d.shape[0])

ax.barh(y_labels, maincat_subcat_products_d["maincategory_count"], facecolor="palevioletred",
       edgecolor="black")
ax.set_yticks(y_labels)
ax.set_yticklabels(maincat_subcat_products_d["maincategory_name"], fontsize=12)
ax.set_title("Main categories")
ax.set_facecolor("navajowhite")
fig.subplots_adjust(top=1, hspace=0.4)

fig.set_facecolor("floralwhite")


com_res="\n".join((r"$\cdot$ " "There exists a total of ten main categories with a\n" \
                  "moderate mount of subcategories in each main category", 
                  r"$\cdot$ " "The diversity and exclusiveness, from the name\n" \
                  "of the main categories, makes it reasonable to take \n" \
                  "into account which main category a product belongs to, \n" \
                  "when modelling the relationship between a product \n" \
                  "and its associated price",
                  r"$\cdot$ " "As can be seen, most of the products are in \n" \
                   "the Women category (with 45% of products), followed by \n" \
                   "Beauty and Kids, which can imply that the buyers are \n" \
                   "predominantly  women and parents",
                  r"$\cdot$ " "On the other hand, while only 8% of products \n"\
                   "are in the electronics main category, it makes sense as \n" \
                   "electronics might be related to more expensive products \n" \
                   "and therefore result in a lower amount of sold products"))


ax.set_xlabel("Amount of subcategories", fontsize=13)
ax.set_ylabel("Main category")

for x_val, y, prod_count in zip(maincat_subcat_products_d["maincategory_count"], y_labels, maincat_subcat_products_d["maincategory_products_count"]):
  ax.text(x_val+2, y+0.1, str(x_val), fontweight="bold")
  ax.text(x_val+2, y-0.3, str(int(prod_count/float(amount_products)*100))+ "%", fontweight="bold", color="green")
  
ax.set_xlim((0, 320))
box=dict(boxstyle="round", edgecolor="black", facecolor="wheat")
fig.text(0.92,0.15, com_res, fontsize=13, bbox=box)


#ax.text(4e04, 10, "Amount of Categories: \n%s"%str(amount_maincategories),
#       bbox=dict(boxstyle="round", edgecolor="black", facecolor="white"))
count_patch=mpatches.Patch(color="black", label="Amount of Subcategories \nin Main Category",
                          facecolor="white")
count_prod_patch=mpatches.Patch(color="green", label="Percentage of Products \nin Main Category")

ax.legend(handles=[count_patch, count_prod_patch])

