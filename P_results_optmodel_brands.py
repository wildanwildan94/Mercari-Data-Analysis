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


## Test of model on validation data for different brands

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

#print res.summary()


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





# (a) For Brand

val_pred_brand_d=val_pred_prop_d.groupby("brand_name").agg({'price':'mean',
                                                           'price_pred':'mean',
                                                           'brand_name':'size'}).rename(columns={'price':'avg_price',
                                                                                                'price_pred':'avg_price_pred',
                                                                                                'brand_name':'brand_count'}).reset_index()

# (b) Define four price ranges, to compare true average price and predicted average price of brand

A_lower=2
A_upper=15

B_lower=15
B_upper=40

C_lower=40
C_upper=100

D_lower=100

val_pred_brand_A_d=val_pred_brand_d.query("avg_price>=%s and avg_price<%s"%(A_lower, A_upper)).sort_values(by="brand_count", ascending=False).head(10).sort_values(by="avg_price", ascending=True)
val_pred_brand_B_d=val_pred_brand_d.query("avg_price>=%s and avg_price<%s"%(B_lower,B_upper)).sort_values(by="brand_count", ascending=False).head(10).sort_values(by="avg_price", ascending=True)
val_pred_brand_C_d=val_pred_brand_d.query("avg_price>=%s and avg_price<%s"%(C_lower, C_upper)).sort_values(by="brand_count", ascending=False).head(10).sort_values(by="avg_price", ascending=True)
val_pred_brand_D_d=val_pred_brand_d.query("avg_price>=%s"%(D_lower)).sort_values(by="brand_count", ascending=False).head(10).sort_values(by="avg_price", ascending=True)


# (c) Visualize prediction and true price for diferent brands

fig, ax = plt.subplots(1,4)

x_labels=range(10)
ax[0].plot(x_labels, val_pred_brand_A_d["avg_price"], 'bo')
ax[0].plot(x_labels, val_pred_brand_A_d["avg_price_pred"], 'ro')
ax[0].set_title("(A) Price Range %s to %s ($USD)"%(A_lower, A_upper))

ax[0].set_xticks(x_labels)
ax[0].set_xticklabels(val_pred_brand_A_d["brand_name"], rotation=90)

ax[1].plot(x_labels, val_pred_brand_B_d["avg_price"], 'bo')
ax[1].plot(x_labels, val_pred_brand_B_d["avg_price_pred"], 'ro')
ax[1].set_title("(B) Price Range %s to %s ($USD)"%(B_lower, B_upper))

ax[1].set_xticks(x_labels)
ax[1].set_xticklabels(val_pred_brand_B_d["brand_name"], rotation=90)

ax[2].plot(x_labels, val_pred_brand_C_d["avg_price"], 'bo')
ax[2].plot(x_labels, val_pred_brand_C_d["avg_price_pred"], 'ro')
ax[2].set_title("(C) Price Range %s to %s ($USD)"%(C_lower, C_upper))

ax[2].set_xticks(x_labels)
ax[2].set_xticklabels(val_pred_brand_C_d["brand_name"], rotation=90)


ax[3].plot(x_labels, val_pred_brand_D_d["avg_price"], 'bo')
ax[3].plot(x_labels, val_pred_brand_D_d["avg_price_pred"], 'ro')
ax[3].set_title("(D) Price Range above %s ($USD)"%D_lower)

ax[3].set_xticks(x_labels)
ax[3].set_xticklabels(val_pred_brand_D_d["brand_name"], rotation=90)


fig.subplots_adjust(bottom=-0.1, right=2.3, wspace=0.3)
fig.suptitle("Predicted and True Average Price of Products in Brands; Ten Brands in Various Price Ranges; \n Predictions on a Validation Set (Independent of Training Set)", x=1.2, y=1.03)

true_price_patch=mpatches.Patch(color="blue", label="Average Price,\nTrue")
pred_price_patch=mpatches.Patch(color="red", label="Average Price,\nPredicted")

fig.legend(handles=[true_price_patch, pred_price_patch], fontsize=10,
          bbox_to_anchor=(1.75, 1.06))

fig.set_facecolor("floralwhite")

for axes in ax.flatten():
  axes.set_facecolor("navajowhite")
  axes.set_ylabel("Average Price")
  
  
com_res_A="\n".join((r"$\cdot$ (A) " "The predictions in the low range \n" \
                    "performs remarkably well, where all predictions are \n" \
                    "just a few dollars away from the true values.",
                    r"$\cdot$ (A) " "Indicates that our model is in general good \n" \
                    "for predicting product prices for low-price range brands \n" \
                    "like H&M, Old Navy and Gap",
                    r"$\cdot$ (A) " "A possible reason might be that a lot of \n" \
                    "products exists in the low-priced range, which makes  \n" \
                    "our model really efficient in predicting such products"))

com_res_B="\n".join((r"$\cdot$ (B) " "Similarly as in the low-price range, the \n" \
                    "prediction of price in moderate price range is really well \n" \
                    "too - again with just a few dollars of margin",
                    r"$\cdot$ (B) " "Hence, our model should perform well predicting \n" \
                    "prices of products like Nike, Disney and American Eagle",
                    r"$\cdot$ (B) " "The well performance might come, as in (A), \n" \
                    "from the ubiquity of moderate priced products"))

com_res_C="\n".join((r"$\cdot$ (C) " "The performance in predicting high \n" \
                    "priced products differs among brands, as can be seen.",
                    r"$\cdot$ (C) ""While a few brands can be predicted well, \n" \
                    "like Coach, Samsung, Michael Kors, the model has a \n" \
                    "difficulty predicting e.g. Free People, Lululemon",
                    r"$\cdot$ (C) " "However, in general, the predictions are not \n" \
                    "too bad. One could argue that a margin of 10-15 \n" \
                    " ($USD)among high-priced products isn't that much, \n" \
                    "especially for brands like Apple"))


com_res_D="\n".join((r"$\cdot$ (D) " "Contrary to the other price ranges, the predictions \n" \
                    "for very-high priced products isn't generally well, as can be seen \n" \
                    "from the increased margin as the true price increases",
                    r"$\cdot$ (D) " "Most likely, it is due to two factors. Firstly, \n" \
                    "in our model a portion of training data points corresponding \n" \
                    "to high cook's distance points were removed, which in the process \n" \
                    "have decreased our model's ability to predict prices for very-high priced \n" \
                    "products. Secondly, there is most likely not as many high-priced products \n" \
                    "in our training data, as opposed to low-, moderate-priced products, which \n" \
                    "most likely have made our model biased towards predicting the correct price \n" \
                    "of low- and moderate-priced products"))

box=dict(boxstyle="round", edgecolor="black", facecolor="wheat")
fig.text(0, -0.8, com_res_A, bbox=box)
fig.text(0.68, -0.78, com_res_B, bbox=box)
fig.text(1.36, -0.8, com_res_C, bbox=box)
fig.text(2.02, -0.92, com_res_D, bbox=box)
