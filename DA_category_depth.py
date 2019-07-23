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
 
## Want to analyze how deep categories are


cat_d=train_d[["category_name"]].dropna()

print list(set(cat_d["category_name"]))

# (a) Count the amount of products in each category

cat_count_d=cat_d.groupby("category_name").agg({'category_name':'size'}).rename(columns={'category_name':'category_count'}).reset_index()

  


# (b) Want to compute the amount of products in subcategories of four 
# main categories

third_level_electronics_name=[]
depth_electronics=[]

for index, row in cat_count_d.iterrows():
  # The main category is the first element in the split
  temp_main_sub_cat=row["category_name"].split("/")
  if temp_main_sub_cat[0]=="Electronics":
    third_level_electronics_name.append(temp_main_sub_cat[2])
    depth_electronics.append(len(temp_main_sub_cat))
    
print third_level_electronics_name[0:4]
print max(depth_electronics)
print min(depth_electronics)
print Counter(depth_electronics)

# (c) Want to compute the amount of depth of categories

depth_array=Counter([])
depth_count_product=Counter([])

for index, row in cat_count_d.iterrows():
  temp_depth=len(row["category_name"].split("/"))
  temp_count=row["category_count"]
  
  depth_array[temp_depth]+=1
  depth_count_product[temp_depth]+=temp_count
  
print "---"
print "Q: The amount of categories with certain depth"
print depth_array
print "---"
print "---"
print "Q: The amount of products with certain depth"
print depth_count_product
print "---"

# (d) Want to analyze which categories have length 4 and 5

cat_name=[]
cat_depth=[]

for index, row in cat_count_d.iterrows():
  temp_catname=row["category_name"]
  temp_depth=len(temp_catname.split("/"))
  cat_name.append(temp_catname)
  cat_depth.append(temp_depth)
  
  
cat_depth_d=pd.DataFrame({'category_name':cat_name,
                         'category_depth':cat_depth})

print "---"
print "Q: What are categories with depth 5"
cat_depth_five_d=cat_depth_d.query("category_depth==5")
for index, row in cat_depth_five_d.iterrows():
  print row["category_name"]
print "---"
print "---"
print "Q: What are categories with depth 4"
cat_depth_four_d=cat_depth_d.query("category_depth==4")
for index, row in cat_depth_four_d.iterrows():
  print row["category_name"]
print "---"


com_title="Want to quantitatively analyze the depth of categories \n" \
          " - Is all subcategories for a product necessary?"
com_count_depth_title="The amount of categories with a certain depth:"

com_fourdepth_categories_title="The categories with a depth of 4: "
com_fourdepth_categories=""
for index, row in cat_depth_four_d.iterrows():
  com_fourdepth_categories+=r"$\cdot$ " + row["category_name"]+ "\n"
  
com_fivedepth_categories_title="The categories with a depth of 5: "
com_fivedepth_categories=""
for index, row in cat_depth_five_d.iterrows():
  com_fivedepth_categories+=r"$\cdot$ " + row["category_name"]+ "\n"

com_transform_title="From the structure and low quantity of the categories with \n" \
"a depth of 4 and 5, we can reconsider the categories as: "
com_transform=r"$\cdot$ " "Handmade/Housewares/Entertaining Serving \n" \
             r"$\cdot$ " "Men/Coats & Jackets/Flight Bomber \n" \
r"$\cdot$ " "Men/Coats & Jackets /Varsity Baseball \n" \
r"$\cdot$ " "Sports & Outdoors/Exercise/Dance Ballet \n" \
r"$\cdot$ " "Sports & Outdoors/Outdoors/Indoor Outdoor Games \n" \
r"$\cdot$ " "Electronics/Computers & Tablets/iPad Tablet eBook Access \n" \
r"$\cdot$ " "Electronics/Computers & Tablets/iPad Tablet eBook Readers"
depth_count_product_dict=dict(depth_count_product)
depth_array_dict=dict(depth_array)
amount_products=cat_d.shape[0]
table_depth_count=[[3, depth_count_product_dict[3], depth_array_dict[3]],
                  [4, depth_count_product_dict[4], depth_array_dict[4]],
                  [5, depth_count_product_dict[5], depth_array_dict[5]],
                  ["Total", amount_products, cat_count_d.shape[0]]]





fig, ax = plt.subplots()

ax.axis("off")
cat_table=ax.table(cellText=table_depth_count, bbox=[-0.15, 0.2, 1.4, 0.7],
        colLabels=["Depth", "Amount of Products", "Amount of Categories"])

cat_table.auto_set_font_size(False)
cat_table.set_fontsize(13)

fig.text(0,1, com_title, fontweight="bold", fontsize=16)
fig.text(0,0.9, com_count_depth_title, fontweight="bold", fontsize=14)
fig.text(1.13, 1, com_fourdepth_categories_title, fontweight="bold", fontsize=14)
fig.text(1.13, 0.63, com_fourdepth_categories, fontsize=13)
fig.text(1.13, 0.61, com_fivedepth_categories_title, fontsize=14, fontweight="bold")
fig.text(1.13, 0.44, com_fivedepth_categories, fontsize=13)
fig.text(1.13, 0.37, com_transform_title, fontsize=14, fontweight="bold")
fig.text(1.13, -0.05, com_transform, fontsize=13)

fig.set_facecolor("floralwhite")

