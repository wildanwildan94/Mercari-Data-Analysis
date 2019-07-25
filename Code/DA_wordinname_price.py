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
 
## Want to analyze the distribution of name, save data

N=train_d.shape[0]
name_price_d=train_d[["name", "price"]]


# (a) Extract all words

#split
word_array=[]
price_array=[]
for index, rows in name_price_d.iterrows():
  if index==int(N/float(10)):
    print "1/10 done"
    
  if index==int(N/float(5)):
    print "1/5 done"
  if index==int(N/float(2)):
    print "1/2 done"
    
  temp_name=rows["name"]
  temp_words=temp_name.split()
  nmbr_words=len(temp_words)
  temp_words=[x.lower() for x in temp_words]
  word_array.extend(temp_words)
  price_array.extend([rows["price"]]*nmbr_words)
  

word_price_d=pd.DataFrame({'word_name':word_array,
                          'word_price':price_array})

word_avgprice_d=word_price_d.groupby("word_name").agg({'word_price':[np.mean, 'size']}).rename(columns={'mean':'avgprice',
                                                                                                        'size':'countword'})
word_avgprice_d.columns=word_avgprice_d.columns.droplevel(0)
word_avgprice_d.reset_index(inplace=True)
print word_avgprice_d.head(4)
word_avgprice_d.to_csv('word_avgprice.csv', index=False)


## Want to analyze the distribution of name and price
## twenty keywords

word_avgprice_d=pd.read_csv('word_avgprice.csv')

# Consider cases where the word count at least exceeds 20
print word_avgprice_d.shape[0]
word_avgprice_d=word_avgprice_d.query("countword>20")
print word_avgprice_d.shape[0]

# (a) Print the form
print word_avgprice_d.iloc[0]



# (b) Compute four price ranges of avgprice
A_lower=5
A_upper=20
B_lower=20
B_upper=60
C_lower=60
C_upper=200
D_lower=200

random_state=18
word_avgprice_A_d=word_avgprice_d.query("avgprice>%s and avgprice<%s"%(A_lower, A_upper)).sample(20, random_state=random_state).sort_values(by="avgprice", ascending=True)
word_avgprice_B_d=word_avgprice_d.query("avgprice>%s and avgprice<%s"%(B_lower, B_upper)).sample(20, random_state=random_state).sort_values(by="avgprice", ascending=True)
word_avgprice_C_d=word_avgprice_d.query("avgprice>%s and avgprice<%s"%(C_lower, C_upper)).sample(20, random_state=random_state).sort_values(by="avgprice", ascending=True)
word_avgprice_D_d=word_avgprice_d.query("avgprice>%s"%(D_lower)).sample(20, random_state=random_state).sort_values(by="avgprice", ascending=True)

#word_avgprice_top_d=word_avgprice_d.sort_values(by="avgprice", ascending=False).head(20).sort_values(by="avgprice", ascending=True)
#word_avgprice_bottom_d=word_avgprice_d.sort_values(by="avgprice", ascending=True).head(20)


# (c) Visualize random products based on average price and different
# price ranges

fig, ax = plt.subplots(1,4)


# Visualize top ten of categories by count
y_labels=range(word_avgprice_A_d.shape[0])

ax[0].barh(y_labels, word_avgprice_A_d["avgprice"], facecolor="palevioletred",
          edgecolor="black")
ax[0].set_yticks(y_labels)
ax[0].set_yticklabels([x.title() for x in word_avgprice_A_d["word_name"]], fontsize=12)
ax[0].set_title("Price Range: %s to %s ($USD)"%(A_lower, A_upper))

ax[1].barh(y_labels, word_avgprice_B_d["avgprice"], facecolor="palevioletred",
          edgecolor="black")
ax[1].set_yticks(y_labels)
ax[1].set_yticklabels([x.title() for x in word_avgprice_B_d["word_name"]], fontsize=12)
ax[1].set_title("Price Range: %s to %s ($USD)"%(B_lower, B_upper))

ax[2].barh(y_labels, word_avgprice_C_d["avgprice"], facecolor="palevioletred",
          edgecolor="black")
ax[2].set_yticks(y_labels)
ax[2].set_yticklabels([x.title() for x in word_avgprice_C_d["word_name"]], fontsize=12)
ax[2].set_title("Price Range: %s to %s ($USD)"%(C_lower, C_upper))


ax[3].barh(y_labels, word_avgprice_D_d["avgprice"], facecolor="palevioletred",
          edgecolor="black")
ax[3].set_yticks(y_labels)
ax[3].set_yticklabels([x.title() for x in word_avgprice_D_d["word_name"]], fontsize=12)
ax[3].set_title("Price Range: %s and above ($USD)"%(D_lower))
fig.subplots_adjust(wspace=0.6, right=1.5, top=1,bottom=0.1, left=-0.3)
fig.suptitle("Average of Prices Related to Word in Name \n 20 Randomly Picked Words for Each Price Range", y=1.15)
for axes in ax.flatten():
  axes.set_facecolor("navajowhite")
  axes.set_xlabel("Average Price ($USD)", fontsize=13)
  
  
  
fig.set_facecolor("floralwhite")

com_A_pricerange="\n".join((r"$\cdot$ " "A lot of descriptive words in all price ranges, including words like Trolls, \n" \
                           "Princesses, Armani, Hoddie, Oxfords, Acoustic, 120Gb, 256Gb, Macbook, Lambskin, Damier",
                           r"$\cdot$ " "Some words give an almost full description of the underlying product, e.g. Macbook, \n" \
                           "while some words give an important aspect, e.g. Lambskin",
                           r"$\cdot$ " "Hence, it is concluded that the words, especially together, in the name attribute \n" \
                           "of a product can give potentially useful information of a product's underlying price"))

com_alg="\n".join((r"$\cdot$ " "Construction of the Average Price:", 
                  " For each possible Word in the name attribute:",
                  "   Collect all products which has that Word \n" \
                  "   in its name attribute", 
                  "   Calculate the average price from \n" \
                  "   all the prices of those products"))

box=dict(boxstyle="round", edgecolor="black", facecolor="wheat")
fig.text(-0.6, -0.38, com_A_pricerange, bbox=box,fontsize=13)
fig.text(0.81, -0.38, com_alg, bbox=box, fontsize=13)
