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


## Second linear regression with outlying values removed

# cook_cutoff_threem_exc_id

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

train_adj_cols=train_adj_d.columns.tolist()
print train_adj_cols
train_adj_cols.remove("index_product")
train_adj_cols.remove("item_condition_id_5.0")
train_adj_cols.remove("shipping_nan")
train_adj_cols.remove("shipping_0.0")

X_train_df=train_adj_d[train_adj_cols].drop("price", axis=1)
# Drop cook's distance points
X_train_df=X_train_df.drop(cook_cutoff_threem_exc_id, axis=0)
# Center
X_train_df=X_train_df-X_train_df.mean()
X_train=X_train_df.as_matrix()
y_train=train_adj_d["price"].drop(cook_cutoff_threem_exc_id, axis=0).as_matrix()

X_train_ones=sm.add_constant(X_train)

model=sm.OLS(y_train, X_train_ones)
res=model.fit()

print res.summary()


model_fitted_y=res.fittedvalues
model_residuals=res.resid
model_norm_residuals=res.get_influence().resid_studentized_internal
model_norm_residuals_abs_sqrt=np.sqrt(np.abs(model_norm_residuals))
model_leverage =res.get_influence().hat_matrix_diag



# (d) Visualize Diagnostic Plots

fig, ax = plt.subplots(2,2)



sns.residplot(model_fitted_y, y_train, ax= ax[0,0],
             scatter_kws={'facecolor':'royalblue',
                         'edgecolor':'black'})

ax[0,0].set_xlabel("Fitted Values")
ax[0,0].set_ylabel("Residuals")
ax[0,0].set_facecolor("navajowhite")
ax[0,0].set_title("(A) Fitted Values vs. Residuals of \n Linear Model")



a=stats.probplot(model_residuals, dist="norm", plot=ax[0,1])

ax[0,1].set_facecolor("navajowhite")
ax[0,1].set_title("(B) QQ-Plot of Residuals of Linear Model")



ax[1,0].scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5)

ax[1,0].set_title("(C) Scale-Location of Linear Model")
ax[1,0].set_xlabel("Fitted Values")
ax[1,0].set_ylabel(r"$\sqrt{Standardized \ Residuals}$")
ax[1,0].set_facecolor("navajowhite")



ax[1,1].scatter(model_leverage, model_norm_residuals, alpha=0.5)

ax[1,1].set_xlabel("Leverage")
ax[1,1].set_ylabel("Standardized Residuals")
ax[1,1].set_title("(D) Leverage vs. Standardised Residuals \n of Linear Model")
ax[1,1].set_xlim((0, 0.01))
ax[1,1].set_facecolor("navajowhite")


fig.set_facecolor("floralwhite")
fig.subplots_adjust(top=0.7, bottom=-0.7, left=-0.7, right=0.7, hspace=0.4, wspace=0.3)

com_fr="\n".join((r"$\cdot$ (A) " "Nonlinear residuals are not immediately clear, \n" \
                 "as indicated from the large residuals in the lower fitted values  \n" \
                  "region",
                 r"$\cdot$ (A) " "The model, still, has a weakness in \n" \
                 "predicting high price products in some cases, especially \n" \
                 "for low fitted values"))


com_qq="\n".join((r"$\cdot$ (B) " "The extreme values in the upper right tail is \n" \
                "less extreme than in the initial regression, but still indicate \n" \
                "that the data might contain a bit more extreme values than expected \n" \
                  "are unfortunately also clear",
                 r"$\cdot$ (B) " "The extreme values indicates that the data may \n" \
                  "have too many extreme values for normally distributed \n" \
                 "residuals"))

                

com_scale=r"$\cdot$ (C) " "A fairly equal spread indicates \n" \
                    "that constant variance of residuals is reasonable, \n" \
"while not perfect, as can be seen by the high standardised residuals for \n" \
"low fitted values"
com_lev="\n".join((r"$\cdot$ (D) " "There is one value with a high leverage, \n" \
                  "but as it has a moderate standardised residual, it shouldn't  \n"\
                  "be that influential",
                  r"$\cdot$ " "Otherwise, the leverage values seems to be fine"))

box=dict(boxstyle="round", edgecolor="black", facecolor="wheat")
fig.text(0.73, 0.43, com_fr, bbox=box)
fig.text(0.73, 0, com_qq, bbox=box)
fig.text(0.73, -0.4, com_scale, bbox=box)
fig.text(0.73, -0.65, com_lev, bbox=box)
fig.suptitle("Fitted Model; A square root transformation and centering of \n attributes around their means are applied; Cook's Distance cut-off points removed", y=0.9, x=0.1)

