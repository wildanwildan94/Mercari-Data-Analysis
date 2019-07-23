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
 
## Analyze of the different subcategories of top main categories

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

# (c) Want to compute the amount of products in subcategories of four 
# main categories

cat_subcat_prodcount={'Women':Counter([]), 'Beauty':Counter([]), 'Kids':Counter([]),
                   'Electronics':Counter([])}
cat_subcat_subsubcount={'Women':Counter([]), 'Beauty':Counter([]), 'Kids':Counter([]),
                 'Electronics':Counter([])}

list_of_main_categories=["Women", "Beauty", "Kids", "Electronics"]
for index, row in cat_count_d.iterrows():
  # The main category is the first element in the split
  temp_main_sub_cat=row["category_name"].split("/")[0:2]
  if temp_main_sub_cat[0] in list_of_main_categories:
    cat_subcat_prodcount[temp_main_sub_cat[0]][temp_main_sub_cat[1]]+=row["category_count"]
    cat_subcat_subsubcount[temp_main_sub_cat[0]][temp_main_sub_cat[1]]+=1
    
    
    
    

# (d) Create dataframes for the count of products of each
# subcategory in the main categories considered, and the amount of
# subsubcategories in each subcategory

maincat_subcat_subsubcount_d={'Women':'A',
                       'Beauty':'B',
                       'Kids':'C',
                       'Electronics':'D'}
maincat_subcat_prodcount_d={'Women':'A',
                           'Beauty':'B',
                           'Kids':'C',
                           'Electronics':'D'}

for key in maincat_subcat_subsubcount_d:
  temp_subcat_name, temp_subcat_subsubcount=zip(*cat_subcat_subsubcount[key].items())
  
  maincat_subcat_subsubcount_d[key]=pd.DataFrame({'subcat_name':temp_subcat_name,
                                          'subcat_subsubcount':temp_subcat_subsubcount})
  
for key in maincat_subcat_prodcount_d:
  temp_subcat_name, temp_subcat_prodcount=zip(*cat_subcat_prodcount[key].items())
  
  maincat_subcat_prodcount_d[key]=pd.DataFrame({'subcat_name':temp_subcat_name,
                                               'subcat_prodcount':temp_subcat_prodcount})
  
  

# (e) Visualize the top subcategories of four main categories

fig, ax=plt.subplots(2,2, sharex=True)

max_subsubcount=0

for maincat, temp_ax in zip(list_of_main_categories, ax.flat):
  print "Currently iterating through %s"%maincat
  temp_maincat_d=maincat_subcat_subsubcount_d[maincat].merge(maincat_subcat_prodcount_d[maincat], on ="subcat_name", how="left")
  temp_maincat_top_d=temp_maincat_d.sort_values(by="subcat_subsubcount", ascending=False).head(10).sort_values(by="subcat_subsubcount",
                                                                                                              ascending=True)
  
  max_subsubcount=np.amax([max_subsubcount, max(temp_maincat_top_d["subcat_subsubcount"])])
 
  y_labels=range(temp_maincat_top_d.shape[0])
  temp_ax.barh(y_labels, temp_maincat_top_d["subcat_subsubcount"],
              facecolor="palevioletred",
              edgecolor="black")
  temp_ax.set_yticks(y_labels)
  temp_ax.set_yticklabels(temp_maincat_top_d["subcat_name"],
                         fontsize=12)
  temp_ax.set_facecolor("navajowhite")
  
  temp_sum_products=np.sum(temp_maincat_d["subcat_prodcount"])

  
  temp_ax.set_title("Popular subcategories in main category: %s"%maincat)
  for x_val, y, prod_count in zip(temp_maincat_top_d["subcat_subsubcount"], y_labels, temp_maincat_top_d["subcat_prodcount"]):
    temp_ax.text(x_val+0.1, y+0.1, str(x_val), fontweight="bold")
    temp_ax.text(x_val+0.1, y-0.3, str(np.round(np.round(prod_count/float(temp_sum_products),3)*100,1))+ "%", fontweight="bold", color="green")

  
  
x_labels=range(22)
ax[0,0].set_xticks(x_labels)
ax[0,0].set_xlim((0, 22))
fig.subplots_adjust(top=1, bottom=-1, right=1, left=-1, wspace=0.48)
fig.set_facecolor("floralwhite")

count_patch=mpatches.Patch(color="black", label="Amount of subcategories \nin a subcategory")
perc_patch=mpatches.Patch(color="green", label="Percentage of Products \nin subcategory")

ax[0,0].legend(handles=[count_patch, perc_patch])


com_wom_beauty="\n".join((r"$\cdot$ " "There are mostly clothing related products \n" \
                         "among the subcategories of the main category 'Women'",
                         r"$\cdot$ " "In particular, product types like Athletic \n" \
                         "Apparel, Shoes, Tops & Blouses are the most popular \n" \
                         "products. Might indicate that women are confident in \n" \
                         "buying clothing on the platform, especially lower-cost \n" \
                         "products like 'Tops & Blouses' and athletic apparel",
                         r"$\cdot$ " "In the 'Beauty' category it is apparent that \n" \
                         "a major part of purchases is of 'Makeup' (with 60%), followed \n" \
                         "by skin care and fragrance",
                         r"$\cdot$ " "'Makeup' and 'Skin Care' is possibly high because \n"\
                         "they are low-cost items.",
                         r"$\cdot$ " "The moderate value of fragrance might indicate \n" \
                         "that consumers are willing to buy second-hand fragrance, \n" \
                         "which has an odour that consumer may or not like, \n" \
                         "making a first-hand purchase more risky"))

com_kids_electronics="\n".join((r"$\cdot$ " "In the 'Kids', most products sold are 'Toys' \n" \
                               "and clothing for kids. 'Toys' are moderate-cost items, \n" \
                               "and parents with limited disposable income might be \n" \
                               "more prone to buy second-hand toys for their kids",
                               r"$\cdot$ " "Buying clothing for ones kids seems reasonable",
                               r"$\cdot$ " "Most products sold in 'Electronics' are \n" \
                               "'Cell Phones & Accessories', by a modest margin",
                               r"$\cdot$ " "A possible reason might be that Phones, and \n" \
                               "accessories, are increasingly popular in everyday life, \n" \
                               "and being a status symbol"))

box=dict(boxstyle="round", edgecolor="black", facecolor="wheat")
fig.text(1.02, 0.13, com_wom_beauty, fontsize=13, bbox=box)
fig.text(1.02, -0.8, com_kids_electronics, fontsize=13, bbox=box)
ax[1,0].set_xlabel("Amount of Subcategories")
ax[1,1].set_xlabel("Amount of Subcategories")

