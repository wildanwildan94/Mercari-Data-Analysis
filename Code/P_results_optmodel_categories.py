# -*- coding: utf-8 -*-
## Load Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import re
import datetime
from collections import Counter
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import median_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from google.colab import files
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
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

## Load transformed data
#train_d=pd.read_csv('/content/drive/My Drive/train_mercari.csv')
train_adj_d=pd.read_csv('/content/drive/My Drive/train_adj_s02_rs13.csv')
train_d=pd.read_csv('train_data_s02_rs13.csv')

test_adj_d=pd.read_csv('/content/drive/My Drive/test_adj_s02_rs13.csv')
test_d=pd.read_csv('test_data_s02_rs13.csv')

# Merge on index product to get price

train_adj_d=train_adj_d.merge(train_d[["index_product", "price"]], on="index_product")
test_full_adj_d=test_adj_d.merge(test_d[["index_product", "price"]], on="index_product")

random_state=19
test_adj_d, val_adj_d=train_test_split(test_full_adj_d, random_state=random_state,
                                        test_size=0.3)


print "---"
print "Shape of test"
print test_adj_d.shape
print "---"
print "Shape of val"
print val_adj_d.shape
print "---"
print "nan of test"
print test_adj_d.isna().sum()
print "---"
print val_adj_d.isna().sum()
print "--"



## Analyze results for certain subsubcategories


## Cook's Distance of Data

train_adj_cols=train_adj_d.columns.tolist()
print train_adj_cols
train_adj_cols.remove("index_product")
train_adj_cols.remove("item_condition_id_5.0")
train_adj_cols.remove("shipping_nan")
train_adj_cols.remove("shipping_0.0")

X_train_df=train_adj_d[train_adj_cols].drop("price", axis=1)
X_train=X_train_df.as_matrix()
y_train=train_adj_d["price"].as_matrix()

X_train_ones=sm.add_constant(X_train)

model=sm.OLS(y_train, X_train_ones)
res=model.fit()




# Define Cook's Distance
model_cooks = np.array(res.get_influence().cooks_distance[0])

# Define Cook's Distance cutoff value: (1) 3*mean of Cook's Distance

threemean_cutoff=3*np.mean(model_cooks)


print "---"
print "Q: What is the value of 3*mean of Cook's Distance"
print threemean_cutoff
print "---"

# Compute all indices where Cook's distance is above threemean_cutoff
cook_cutoff_threem_exc_id=np.argwhere(model_cooks>threemean_cutoff)[:,0]

## Test of model on validation data


train_adj_cols=train_adj_d.columns.tolist()
# Remove index product
train_adj_cols.remove("index_product")

# Remove dummy variables not needed
train_adj_cols.remove("item_condition_id_nan")
train_adj_cols.remove("shipping_nan")
train_adj_cols.remove("shipping_1.0")
train_adj_cols.remove("item_condition_id_5.0")


X_train_df=train_adj_d[train_adj_cols].drop("price", axis=1).apply(np.cbrt)
# Drop cook's distance points
X_train_df=X_train_df.drop(cook_cutoff_threem_exc_id, axis=0)
X_train_center_df=X_train_df-X_train_df.mean()

X_train_m=X_train_center_df.as_matrix()
X_train_m=sm.add_constant(X_train_m)

y_train=train_adj_d["price"].drop(cook_cutoff_threem_exc_id, axis=0).apply(np.cbrt).as_matrix()


X_val_df=val_adj_d[train_adj_cols].drop("price", axis=1).apply(np.cbrt)
X_val_center_df=X_val_df-X_train_df.mean()
X_val_m=X_val_center_df.as_matrix()
X_val_m=sm.add_constant(X_val_m)
y_val=val_adj_d["price"].as_matrix()


clf=LinearRegression(fit_intercept=False)
clf.fit(X_train_m, y_train)

y_pred_val=clf.predict(X_val_m)




val_pred_d=pd.DataFrame({'index_product':val_adj_d["index_product"].values,
                        'price_pred':y_pred_val**3})

val_pred_prop_d=val_pred_d.merge(test_d, on="index_product",
                                how="left")

val_pred_cat_d=val_pred_prop_d[["category_name", "price", "price_pred"]].dropna(subset=["category_name"])

maincat_array=[]
subcat_array=[]
subsubcat_array=[]
price_array=[]
pricepred_array=[]

for index, row in val_pred_cat_d.iterrows():
  
  temp_subcategories=row["category_name"].split("/")
  
  temp_price=row["price"]
  temp_pricepred=row["price_pred"]
  
  for temp_cat in temp_subcategories:
    maincat_array.append(temp_subcategories[0])
    subcat_array.append(temp_subcategories[1])
    subsubcat_array.append(temp_subcategories[-1])
    price_array.append(temp_price)
    pricepred_array.append(temp_pricepred)
    
    
    
categ_price_pred_d=pd.DataFrame({'maincat_name':maincat_array,
                                'subcat_name':subcat_array,
                                'subsubcat_name':subsubcat_array,
                                'price':price_array,
                                'price_pred':pricepred_array})

# (a) Analyze predictions for main categories

categ_avgprice_d=categ_price_pred_d.groupby("maincat_name").agg({'price':'mean',
                                                                'price_pred':'mean',
                                                                'maincat_name':'size'}).rename(columns={'price':'avg_price',
                                                                                                       'price_pred':'avg_pricepred',
                                                                                                       'maincat_name':'count'}).reset_index()

print categ_avgprice_d.iloc[0]



# (a) Define a list of subsubcategories to analyze, compute
# average prices and average predicted prices for products in 
# subsubcategories 

subsubcat_examples=["Raincoat", "Watch", "Bracelets", "Bottoms", "Tops & T-Shirts",
                   "Cell Phone Accessories", "Camera", "Video Gaming Merchandise",
                   "Consoles", "Khakis, Chinos"]

subsubcat_price_pred_d=categ_price_pred_d[categ_price_pred_d["subsubcat_name"].isin(subsubcat_examples)][["subsubcat_name",
                                                                                                         "price",
                                                                                                         "price_pred"]]

subsubcat_avgprice_pred_d=subsubcat_price_pred_d.groupby("subsubcat_name").agg({'price':'mean',
                                                                               'price_pred':'mean'}).rename(columns={'price':'avg_price',
                                                                                                                    'price_pred':'avg_pricepred'}).reset_index()



# (b) Visualize price and predicted price averages

fig, ax = plt.subplots()

x_labels=range(subsubcat_avgprice_pred_d.shape[0])

ax.plot(x_labels, subsubcat_avgprice_pred_d["avg_price"], 'bo',
       label="Average Price, True")
ax.plot(x_labels, subsubcat_avgprice_pred_d["avg_pricepred"], 'ro',
       label="Average Price, Predicted")

ax.set_xticks(x_labels)
ax.set_xticklabels(subsubcat_avgprice_pred_d["subsubcat_name"].values, rotation=90,
                  fontsize=14)
ax.set_facecolor("navajowhite")
ax.set_ylabel("Average Price")
ax.set_title("Predicted and True Average Prices of Products in Certain (Subsub) Categories; \n Predictions on a Validation Set (Independent of Training Set)")


fig.subplots_adjust(bottom=-0.1, right=1.3)
fig.legend()


com_res="\n".join((r"$\cdot$ " "The predictions seems to perform, generally, \n" \
                  "well for most subsubcategories considered.",
                  r"$\cdot$ " "There is a problem in predicting prices \n" \
                  "for prices of products in 'Console' - which may be \n" \
                  "because our model is less well in predicting products \n" \
                  "with very high prices, as 'Console'-products generally are",
                  r"$\cdot$ " "Further, because our model has been shown to \n" \
                  "perform well on low- and moderate-priced products, it is \n" \
                  "no surprise that the predictions are well for the other \n" \
                  "subsubcategories considered")) 

fig.text(1.32, 0.2, com_res, bbox=dict(boxstyle="round",
                                     edgecolor="black", facecolor="wheat"),
        fontsize=13)
fig.set_facecolor("floralwhite")
