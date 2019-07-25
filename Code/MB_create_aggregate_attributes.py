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
from sklearn.model_selection import train_test_split
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

## Load Mercari full data

full_data_d=pd.read_csv('/content/drive/My Drive/train_mercari.csv')
# Drop train_id attribute
try:
  full_data_d.drop("train_id", axis=1, inplace=True)
except:
  print "Already removed train_id"
    
# Add index to each products

index_product=range(full_data_d.shape[0])

full_data_d["index_product"]=index_product


## Split Full Data into training and test data

split_size=0.2
random_state=13
train_d, test_d=train_test_split(full_data_d, random_state=random_state,
                                test_size=0.2)



train_d.to_csv('train_data_s02_rs13.csv', index=False)
test_d.to_csv('test_data_s02_rs13.csv', index=False)



## Construct Model Building for the Name Attribute

# name_create(data): Based on the names and price in data, it creates
# a dict where the keys are all the differnt words and the associated
# values are the average price of all products with that word in its name
# attribute
# name_apply(data, data_name): Based on dict of words and average prices in
# data_name, it assigns the weighted price of all names in data, based on
# what words occur in the name in an average sense

def name_create(data):
  name_price_d=data[["name", "price"]].dropna()
  
  # count_word=keep a count of the occurence of each word
  # price_word=sum the prices associated with each word
  count_word=Counter([])
  price_word=Counter([])
  
  N=name_price_d.shape[0]
  for index, row in data.iterrows():
    if index==int(N/float(4)):
      print "25% Done of name_create"
    elif index==int(N/float(2)):
      print "50% Done of name_create"
    elif index==int(3*N/float(4)):
      print "75% Done of name_create"
    temp_name=str(row["name"])
    temp_price=row["price"]
    
    temp_words=temp_name.split()
    temp_words=[re.sub('[\W\_]', '', x).lower() for x in temp_words]
    
    for temp_word in temp_words:
      count_word[temp_word]+=1
      price_word[temp_word]+=temp_price
      
  count_word_name, count_word=zip(*count_word.items())
  price_word_name, sumprice_word=zip(*price_word.items())
  
  count_word_d=pd.DataFrame({'word_name':count_word_name,
                            'word_count':count_word})
  price_word_d=pd.DataFrame({'word_name':count_word_name,
                            'word_sumprice':sumprice_word})
  
  word_count_price_d=count_word_d.merge(price_word_d, on="word_name")
  word_count_price_d["word_weighted_price"]=word_count_price_d["word_sumprice"]/word_count_price_d["word_count"]
  
  weighted_price=word_count_price_d["word_weighted_price"].tolist()
  word_names=word_count_price_d["word_name"].tolist()
  wordname_weightedprice=dict(zip(word_names, weighted_price))
  
  return wordname_weightedprice

def name_apply(data, data_name):
  
  name_price_pindex_d=data[["name", "index_product"]]
  
  weighted_price=[]
  
  avg_weightedprice=np.array(list(data_name.values())).mean()
  for index, row in name_price_pindex_d.iterrows():
    temp_name=str(row["name"])
    temp_words=temp_name.split()
    
    temp_words=[re.sub('[\W\_]', '', x).lower() for x in temp_words]
    temp_weightedprice=0
    words_in_name=0
    for temp_word in temp_words:
      try:
        temp_word_weightedprice=data_name[temp_word]
        words_in_name+=1
        temp_weightedprice+=temp_word_weightedprice
      except:
        temp_word_weightedprice=avg_weightedprice
        words_in_name+=1
        temp_weightedprice+=temp_word_weightedprice
    weighted_price.append(temp_weightedprice/float(words_in_name))
    
  name_price_pindex_d["name_weighted_price"]=weighted_price
  
  return name_price_pindex_d[["index_product", "name_weighted_price"]]
        
    
  

## Want to add dummy variables for item conditions
## itemcond_apply(data): Assigns dummy variables for
## each element in data, with respect to the 
## item_condition_id attribute


def itemcond_apply(data):
  
  itemcond_d=data[["index_product", "item_condition_id"]]
  
  itemcond_dummies_d=pd.get_dummies(itemcond_d,
                                    dummy_na=True,
                                   prefix="item_condition_id",
                                   columns=["item_condition_id"])
  
  return itemcond_dummies_d
  
  




## Based on creating two functions for the Category Name attribute

## category_create(data): Create three dataframes, based on the price
## values in data, such that
## (1) The average price of all products in main category
## (2) The average price of all products in main category and subcategory
## (3) The average price of all products in mina category, subcategory
## and subsubcategory

def category_create(data):
  
  category_price_d=data[["category_name", "price"]]
  
  maincat_array=[]
  secondcat_array=[]
  thirdcat_array=[]
  price_array=[]
  for index, row in category_price_d.iterrows():
    
    # Possibly add try except here for NaNs
    try:
      temp_subcategories=row["category_name"].split("/")
    
      temp_price=row["price"]
    
      for temp_cat in temp_subcategories:
        maincat_array.append(temp_subcategories[0])
        secondcat_array.append(temp_subcategories[1])
        thirdcat_array.append(temp_subcategories[-1])
        price_array.append(temp_price)
        
    except:
      # In case if NaN, add unknown for later usage
      maincat_array.append("unknown")
      secondcat_array.append("unknown")
      thirdcat_array.append("unknown")
      price_array.append(row["price"])
  category_price_d=pd.DataFrame({'main_category_name':maincat_array,
                                'sub_category_name':secondcat_array,
                                'subsub_category_name':thirdcat_array,
                                'price':price_array})
  
  maincategory_avgprice_d=category_price_d.groupby("main_category_name").agg({'price':'mean'}).rename(columns={'price':'main_category_average_price'}).reset_index()
  
  subcategory_avgprice_d=category_price_d.groupby(["main_category_name", "sub_category_name"]).agg({'price':'mean'}).rename(columns={'price':'sub_category_average_price'}).reset_index()
  
  subsubcategory_avgprice_d=category_price_d.groupby(["main_category_name", "sub_category_name", "subsub_category_name"]).agg({'price':'mean'}).rename(columns={'price':'subsub_category_average_price'}).reset_index()

  
  return maincategory_avgprice_d, subcategory_avgprice_d, subsubcategory_avgprice_d
      
    
    
def category_apply(data, data_maincat, data_subcat, data_subsubcat):
  
  category_price_d=data[["category_name", "index_product"]]
  
  maincat_array=[]
  secondcat_array=[]
  thirdcat_array=[]
  index_product_array=[]
  print category_price_d.shape
  for index, row in category_price_d.iterrows():
    
    # Possibly add try except here for NaNs
    try:
      temp_subcategories=row["category_name"].split("/")
   
    

      maincat_array.append(temp_subcategories[0])
      secondcat_array.append(temp_subcategories[1])
      thirdcat_array.append(temp_subcategories[-1])
      index_product_array.append(row["index_product"])
        
    except Exception as e:
      # In case if NaN, add unknown for later usage
      maincat_array.append("unknown")
      secondcat_array.append("unknown")
      thirdcat_array.append("unknown")
      index_product_array.append(row["index_product"])
      

  category_price_d=pd.DataFrame({'main_category_name':maincat_array,
                                'sub_category_name':secondcat_array,
                                'subsub_category_name':thirdcat_array,
                                'index_product':index_product_array})
  
  # Merge on main category
  
  category_price_d=category_price_d.merge(data_maincat, on="main_category_name",
                                         how="left")
  
  # Merge on subcategory
  
  category_price_d=category_price_d.merge(data_subcat, on =["main_category_name", "sub_category_name"],
                                         how="left")
  

  # Merge on subsubcategory
  
  category_price_d=category_price_d.merge(data_subsubcat, on=["main_category_name", "sub_category_name", "subsub_category_name"],
                                         how="left")

  
  
  # Fill NaN with average in each subcategory
  
  category_price_d["main_category_average_price"]=category_price_d["main_category_average_price"].fillna(np.mean(category_price_d["main_category_average_price"]))
  
  category_price_d["sub_category_average_price"]=category_price_d["sub_category_average_price"].fillna(np.mean(category_price_d["sub_category_average_price"]))
  
  category_price_d["subsub_category_average_price"]=category_price_d["subsub_category_average_price"].fillna(np.mean(category_price_d["subsub_category_average_price"]))
  
  
  return category_price_d[["index_product", "main_category_average_price", "sub_category_average_price",
                          "subsub_category_average_price"]]
  
  
                   
  
    

 ## Construct two functions for the brand_name attribute


def brand_create(data):
  
  

  brand_price_d=data[["brand_name", "price"]]
  brand_price_d["brand_name"]=brand_price_d["brand_name"].fillna("unknown")
  

  
  brand_avgprice_d=brand_price_d.groupby("brand_name").agg({'price':'mean'}).rename(columns={'price':'brand_average_price'}).reset_index()
  
  
  return brand_avgprice_d


def brand_apply(data, data_brand):
  
  brand_indexprod_d=data[["index_product", "brand_name"]]
  
  brand_indexprod_d["brand_name"]=brand_indexprod_d["brand_name"].fillna("unknown")
  
  brand_indexprod_avgprice_d=brand_indexprod_d.merge(data_brand, on="brand_name",
                                                    how="left")
  unknown_brand_avgprice=0
  try:
    unknown_brand_avgprice=data_brand.query("brand_name=='unknown'")["brand_average_price"].iloc[0]
    print unknown_brand_avgprice
  except:
    unknown_brand_avgprice=np.mean(data_brand["brand_average_price"])
  brand_indexprod_avgprice_d["brand_average_price"]=brand_indexprod_avgprice_d["brand_average_price"].fillna(unknown_brand_avgprice)
  
  return brand_indexprod_avgprice_d[["index_product", "brand_average_price"]]



## Construct Model Building for the Item Description attribute

# name_create(data): Based on the names and price in data, it creates
# a dict where the keys are all the differnt words and the associated
# values are the average price of all products with that word in its name
# attribute
# name_apply(data, data_name): Based on dict of words and average prices in
# data_name, it assigns the weighted price of all names in data, based on
# what words occur in the name in an average sense

def itemdesc_create(data):
  itemdesc_price_d=data[["item_description", "price"]].dropna()
  
  # count_word=keep a count of the occurence of each word
  # price_word=sum the prices associated with each word
  count_word=Counter([])
  price_word=Counter([])
  
  N=itemdesc_price_d.shape[0]
  for index, row in itemdesc_price_d.iterrows():
    if index==int(N/float(4)):
      print "25% Done of itemdesc_create"
    elif index==int(N/float(2)):
      print "50% Done of itemdesc_create"
    elif index==int(3*N/float(4)):
      print "75% Done of itemdesc_create"
    temp_name=str(row["item_description"])
    temp_price=row["price"]
    
    temp_words=temp_name.split()
    temp_words=[re.sub('[\W\_]', '', x).lower() for x in temp_words]
    
    for temp_word in temp_words:
      count_word[temp_word]+=1
      price_word[temp_word]+=temp_price
      
  count_word_name, count_word=zip(*count_word.items())
  price_word_name, sumprice_word=zip(*price_word.items())
  
  count_word_d=pd.DataFrame({'word_name':count_word_name,
                            'word_count':count_word})
  price_word_d=pd.DataFrame({'word_name':count_word_name,
                            'word_sumprice':sumprice_word})
  
  word_count_price_d=count_word_d.merge(price_word_d, on="word_name")
  word_count_price_d["word_weighted_price"]=word_count_price_d["word_sumprice"]/word_count_price_d["word_count"]
  
  weighted_price=word_count_price_d["word_weighted_price"].tolist()
  word_names=word_count_price_d["word_name"].tolist()
  wordname_weightedprice=dict(zip(word_names, weighted_price))
 
  return wordname_weightedprice

def itemdesc_apply(data, data_itemdesc):
  
  itemdesc_price_pindex_d=data[["item_description", "index_product"]]
  
  weighted_price=[]
  
  avg_weightedprice=np.array(list(data_itemdesc.values())).mean()
  for index, row in itemdesc_price_pindex_d.iterrows():
    temp_name=str(row["item_description"])
    temp_words=temp_name.split()
    
    temp_words=[re.sub('[\W\_]', '', x).lower() for x in temp_words]
    if len(temp_words)==0:
      weighted_price.append(avg_weightedprice)
    temp_weightedprice=0
    words_in_name=0
    
    for temp_word in temp_words:
      try:
        temp_word_weightedprice=data_itemdesc[temp_word]
        words_in_name+=1
        temp_weightedprice+=temp_word_weightedprice
      except:
        temp_word_weightedprice=avg_weightedprice
        words_in_name+=1
        temp_weightedprice+=temp_word_weightedprice
    weighted_price.append(temp_weightedprice/float(words_in_name))
    
  itemdesc_price_pindex_d["item_description_weighted_price"]=weighted_price
  
  return itemdesc_price_pindex_d[["index_product", "item_description_weighted_price"]]
        
    
  
## Apply dummy variables to shipping fee


def shipping_apply(data):
  
  shipping_d=data[["index_product", "shipping"]]
  
  shipping_dummies_d=pd.get_dummies(shipping_d,
                                    dummy_na=True,
                                   prefix="shipping",
                                   columns=["shipping"])
  

  
  return shipping_dummies_d
  
  
 def create_model(data_create, data_apply):
  
  # Create a dataframe of only the index of product
  data_apply_merged_d=data_apply[["index_product"]]
  
  # Create and apply transformation of Name
  name_created=name_create(data_create)
  name_applied=name_apply(data_apply, name_created)
  
  data_apply_merged_d=data_apply_merged_d.merge(name_applied, on="index_product", how="left")
  
  
  
  """
  print "---"
  print "After Name"
  print data_apply_merged_d.shape
  print "---"
  """
  
  # Apply transformation of item_condition_id
  
  itemcond_applied=itemcond_apply(data_apply)
  
  data_apply_merged_d=data_apply_merged_d.merge(itemcond_applied, on="index_product", how="left")
  
  
  """
  print "---"
  print "After item cond"
  print data_apply_merged_d.shape
  print "---"
  """
  
  # Create and apply transformation of category
  
  maincat_created, subcat_created, subsubcat_created=category_create(data_create)
  
  category_applied=category_apply(data_apply, maincat_created, subcat_created,
                                 subsubcat_created)
  
  data_apply_merged_d=data_apply_merged_d.merge(category_applied, how="left", on="index_product")
  
  
  """
  print "---"
  print "After category"
  print data_apply_merged_d.shape
  print "---"
  """
  
  # Create and apply transformation of brand
  
  brand_created=brand_create(data_create)
  brand_applied=brand_apply(data_apply, brand_created)
  
  data_apply_merged_d=data_apply_merged_d.merge(brand_applied, how="left", on="index_product")
  
  
  """
  print "---"
  print "After brand"
  print data_apply_merged_d.shape
  print "---"
  """
  
  # Create and apply transformation of itemdesc
  
  itemdesc_created=itemdesc_create(data_create)
  itemdesc_applied=itemdesc_apply(data_apply, itemdesc_created)
  
  data_apply_merged_d=data_apply_merged_d.merge(itemdesc_applied, how="left", on="index_product")
  
  
  """
  print "---"
  print "after itemdesc"
  print data_apply_merged_d.shape
  print "---"
  """
  
  # Apply shipping transformation 
  
  shipping_applied=shipping_apply(data_apply)
  
  data_apply_merged_d=data_apply_merged_d.merge(shipping_applied, how="left", on="index_product")
  
  """
  print "---"
  print "After shipping"
  print data_apply_merged_d.columns
  print data_apply_merged_d.shape
  print "---"
  """
  
  return data_apply_merged_d
  
  
## Test Create Model
#train_d=pd.read_csv('train_data_s02_rs13.csv')
test_d=pd.read_csv('test_data_s02_rs13.csv')


# For Training Data

#train_merged=create_model(train_d, train_d)

#train_merged.to_csv('train_adj_s02_rs13.csv', index=False)


# For testing data

test_merged=create_model(train_d, test_d)

test_merged.to_csv('test_adj_s02_rs13.csv', index=False)
