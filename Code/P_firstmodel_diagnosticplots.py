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


## First Linear Regression

# train_adj_d

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

print res.summary()

### Construct a Diagnostic Plot; Fitted vs. Residuals



# (c) Define help-values 
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



stats.probplot(model_residuals, dist="norm", plot=ax[0,1])
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

com_fr="\n".join((r"$\cdot$ (A) " "The spread around the horizontal line \n" \
                 "makes nonlinear residuals probable, but not perfect \n",
                 r"$\cdot$ (A) " "In particular, the model has a weakness in \n" \
                 "predicting high price products in some cases, especially \n" \
                 "for low fitted values"))


com_qq="\n".join((r"$\cdot$ (B) " "In the QQ-plot, the assumption of \n" \
                "normally distributed residuals is somewhat probable, \n" \
                "in the middle region, but the extreme deviations at the tails \n" \
                  "are unfortunately also clear",
                 r"$\cdot$ (B) " "The extreme values indicates that the data may \n" \
                  "have too many extreme values for normally distributed \n" \
                 "residuals to be true",
                 r"$\cdot$ (B) " "As in (A), the extreme values are most likely \n" \
                 "cases where the fitted value is low when the real product price \n" \
                 "is high",
                 r"$\cdot$ (B) " "This signifies that the model might not always be \n" \
                 "good at predicting the price of high-price products", 
                 r"$\cdot$ (B) ""A possible reason might be that most data values \n" \
                 "are items which are sold for low/moderate prices, and will tilt \n" \
                 "the model to become better at predicting the price of products with \n" \
                 "low to moderate prices"))
                

com_scale=r"$\cdot$ (C) " "A fairly equal spread indicates \n" \
                    "that constant variance of residuals is reasonable"
com_lev="\n".join((r"$\cdot$ (D) " "Most values with \n" \
                  "high standardised residuals are \n"\
                  "not influential",
                  r"$\cdot$ (D) " "There are a few points with high \n" \
                  "leverage, but with low to moderate standardised \n" \
                  "residuals, which makes them less influential \n" \
                  "on the fit of the linear model"))
box=dict(boxstyle="round", edgecolor="black", facecolor="wheat")
fig.text(0.73, 0.43, com_fr, bbox=box)
fig.text(0.73, -0.27, com_qq, bbox=box)
fig.text(0.73, -0.4, com_scale, bbox=box)
fig.text(0.73, -0.75, com_lev, bbox=box)
fig.suptitle("Fitted Model; A square root transformation and centering of \n attributes around their means are applied", y=0.9, x=0.1)

