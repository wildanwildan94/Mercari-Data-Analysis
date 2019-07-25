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



## Want to consider models based on utilizing different subsets
## of the available attributes

def powerset(seq):
    """
    Returns all the subsets of this set. This is a generator.
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item
            
            
            
            
            

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




#  Define Cook's Distance
model_cooks = np.array(res.get_influence().cooks_distance[0])

# Define Cook's Distance cutoff value: (1) 3*mean of Cook's Distance

threemean_cutoff=3*np.mean(model_cooks)


print "---"
print "Q: What is the value of 3*mean of Cook's Distance"
print threemean_cutoff
print "---"

#  Compute all indices where Cook's distance is above threemean_cutoff
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
X_train_df=X_train_df-X_train_df.mean()

y_train=train_adj_d["price"].drop(cook_cutoff_threem_exc_id, axis=0).apply(np.cbrt).as_matrix()
X_train_cols=X_train_df.columns.tolist()


X_test_df=test_adj_d[train_adj_cols].drop("price", axis=1).apply(np.cbrt)
X_test_df=X_test_df-X_train_df.mean()

y_test=test_adj_d["price"].apply(np.cbrt).as_matrix()


print "---"
print list(X_train_df.columns)
print "---"
print "---"
print list(X_test_df.columns)
print "---"


X_dummy_vars=["item_condition_id_1.0", "item_condition_id_2.0", "item_condition_id_3.0",
             "item_condition_id_4.0", "shipping_0.0"]
X_vars_vary=["name_weighted_price", "main_category_average_price", "sub_category_average_price",
            "subsub_category_average_price", "brand_average_price", "item_description_weighted_price"]



data_size=len(y_train)
subsets=[x for x in powerset(X_vars_vary)]

median_scorer=make_scorer(median_absolute_error)
mse_scorer=make_scorer(mean_squared_error)
cross_validation_size=8

median_absolute_error_score=[]
mean_squared_error_score=[]
subsets_array=[]
subset_size_array=[]
iterations=len(subsets)
for i in range(len(subsets)):
  print "---"
  print "Doing iteration %s out of %s"%(i, iterations)
  subset_cols=subsets[i]
  subset_cols.extend(X_dummy_vars)
  print subset_cols
  size_subset=len(subset_cols)+1
  subset_cols_list=list(subset_cols)
  X_train_subset=X_train_df[subset_cols].as_matrix()
  
  X_train_subset_ones=sm.add_constant(X_train_subset)
      
  clf=LinearRegression(fit_intercept=False)
   
  temp_cv_score_mae=cross_val_score(clf, X_train_subset_ones,
                                   y_train, cv=cross_validation_size,
                                   scoring='neg_median_absolute_error')
  
  clf=LinearRegression(fit_intercept=False)
  temp_cv_score_mse=cross_val_score(clf, X_train_subset_ones,
                                   y_train, cv=cross_validation_size,
                                   scoring='neg_mean_squared_error')
  
  median_absolute_error_score.append(-np.mean(temp_cv_score_mae))
  mean_squared_error_score.append(-np.mean(temp_cv_score_mse))
  
  
  subsets_array.append(subset_cols)
  subset_size_array.append(size_subset)
  
  print "Finished iteration %s out of %s"%(i, iterations)
  print "---"
  
  
results_measure=pd.DataFrame({'subset':subsets_array,
                             'median_absolute_error':median_absolute_error_score,
                              'mean_squared_error':mean_squared_error_score,
                             'subset_size':subset_size_array})

results_measure.to_csv('results_measures_full_mercari_falseintercept.csv', index=False)





  
  
  
  
  
  ## Analyze results from iteration over different subsetmodels

results_measures=pd.read_csv('results_measures_full_mercari_falseintercept.csv')
print results_measures.iloc[0]


results_measures["index"]=range(results_measures.shape[0])


subset_size_array=results_measures["subset_size"].values
median_abs_error_train=results_measures["median_absolute_error"].values
median_abs_error_test=results_measures["mean_squared_error"].values




# (a) Visualize the results for all subset sizes
fig, ax = plt.subplots(1,2)


ax[0].scatter(subset_size_array,
           median_abs_error_train,
           facecolor="royalblue",
           edgecolor="black")

ax[0].set_xlabel("Amount of Attributes", fontsize=13)
ax[0].set_ylabel("Median Absolute Error", fontsize=13)
ax[0].set_title("Average of Median Absolute Error, \n via Cross-Validation with 8 Partitions, \n for Different Models", fontsize=13)
ax[0].set_facecolor("navajowhite")

ax[1].scatter(subset_size_array,
             median_abs_error_test,
             facecolor="royalblue",
             edgecolor="black")
ax[1].set_xlabel("Amount of Attributes", fontsize=13)
ax[1].set_ylabel("Mean Squared Error", fontsize=13)
ax[1].set_title("Average of Mean Squared Error, \n via Cross-Validation with 8 Partitions, \n for Different Models", fontsize=13)
ax[1].set_facecolor("navajowhite")

fig.suptitle("Cross-validation of training data for different measure of precision; Of a linear model",
            y=1.1, x=0.9, fontsize=13)
fig.subplots_adjust(bottom=-0.1, right=1.5)
fig.set_facecolor("floralwhite")

com_res="\n".join((r"$\cdot$ " "A way to avoid overly optimistic predictions, \n" \
                  "the cross-validation technique can be applied to training data.",
                  r"$\cdot$ " "The results of a cross-validation applied to different \n" \
                  "models implies that a model consisting of all attributes is \n" \
                  "preferable - which indicates that most attributes convey \n" \
                  "meaningful information for regression of the underlying \n" \
                  "price of a product",
                  r"$\cdot$ " "This might make sense from the construction of \n" \
                  "the problem - there is really no natural 'numerical' attributes \n" \
                  "of the problem, only categorical. Hence, the algorithm accepts \n" \
                  "the constructed attributes, for brand, item description, name, \n" \
                  "categories",
                  r"$\cdot$ " "Another reason is the large sample sizes - the combined \n" \
                  "dataset for the products exceed one million in the amount of datapoints - \n" \
                  "any attributes that can help in the prediction is accepted"))

fig.text(1.53, 0.1, com_res, bbox=dict(boxstyle="round",
                                     edgecolor="black",
                                     facecolor="wheat"),
        fontsize=13)

