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
 
 
## Want to analyze the distribution of item_description, save data

N=train_d.shape[0]
item_desc_price_d=train_d[["item_description", "price"]]

# (a) Extract all words in item descriptions and with count



count_word=Counter([])
price_word=Counter([])

for index, rows in item_desc_price_d.iterrows():
  if index==int(N/float(10)):
    print "1/10 done"
    
  if index==int(N/float(5)):
    print "1/5 done"
  if index==int(N/float(2)):
    print "1/2 done"
  if index==int(3*N/float(4)):
    print "3/4 done"
    
  temp_name=str(rows["item_description"])
  temp_price=rows["price"]
  temp_words=temp_name.split()
  temp_words=[re.sub('[\W\_]', '', x).lower() for x in temp_words]
  for temp_word in temp_words:
    count_word[temp_word]+=1
    price_word[temp_word]+=temp_price

  
count_word_name, count_word=zip(*count_word.items())
price_word_name, price_word=zip(*price_word.items())

count_word_d=pd.DataFrame({'word_name':count_word_name,
                          'word_count':count_word})
price_word_d=pd.DataFrame({'word_name':price_word_name,
                          'word_sumprice':price_word})

word_price_count_d=price_word_d.merge(count_word_d, on="word_name", how="left")
print "---"
print "Q: How does word_price_count_d look like?"
print word_price_count_d.iloc[0]
print "---"

word_price_count_d.to_csv('word_price_count.csv', index=False)

## Want to analyze the distribution of name and price
## twenty keywords

word_price_count_d=pd.read_csv('word_price_count.csv')

word_price_count_d["avgprice"]=word_price_count_d["word_sumprice"]/word_price_count_d["word_count"]

# Consider cases where the word count at least exceeds 20
word_price_count_d=word_price_count_d.query("word_count>20")


# (a) Print the form
print word_price_count_d.iloc[0]



# (b) Compute four price ranges of avgprice
A_lower=5
A_upper=20
B_lower=20
B_upper=60
C_lower=60
C_upper=200
D_lower=200

random_state=18
word_price_count_A_d=word_price_count_d.query("avgprice>%s and avgprice<%s"%(A_lower, A_upper)).sample(20, random_state=random_state).sort_values(by="avgprice", ascending=True)
word_price_count_B_d=word_price_count_d.query("avgprice>%s and avgprice<%s"%(B_lower, B_upper)).sample(20, random_state=random_state).sort_values(by="avgprice", ascending=True)
word_price_count_C_d=word_price_count_d.query("avgprice>%s and avgprice<%s"%(C_lower, C_upper)).sample(20, random_state=random_state).sort_values(by="avgprice", ascending=True)
word_price_count_D_d=word_price_count_d.query("avgprice>%s"%(D_lower)).sample(20, random_state=random_state).sort_values(by="avgprice", ascending=True)

#word_avgprice_top_d=word_avgprice_d.sort_values(by="avgprice", ascending=False).head(20).sort_values(by="avgprice", ascending=True)
#word_avgprice_bottom_d=word_avgprice_d.sort_values(by="avgprice", ascending=True).head(20)


# (c) Visualize random products based on average price and different
# price ranges

fig, ax = plt.subplots(1,4)


# Visualize top ten of categories by count
y_labels=range(word_price_count_A_d.shape[0])

ax[0].barh(y_labels, word_price_count_A_d["avgprice"], facecolor="palevioletred",
          edgecolor="black")
ax[0].set_yticks(y_labels)
ax[0].set_yticklabels([x.title() for x in word_price_count_A_d["word_name"]], fontsize=12)
ax[0].set_title("Price Range: %s to %s ($USD)"%(A_lower, A_upper))

ax[1].barh(y_labels, word_price_count_B_d["avgprice"], facecolor="palevioletred",
          edgecolor="black")
ax[1].set_yticks(y_labels)
ax[1].set_yticklabels([x.title() for x in word_price_count_B_d["word_name"]], fontsize=12)
ax[1].set_title("Price Range: %s to %s ($USD)"%(B_lower, B_upper))

ax[2].barh(y_labels, word_price_count_C_d["avgprice"], facecolor="palevioletred",
          edgecolor="black")
ax[2].set_yticks(y_labels)
ax[2].set_yticklabels([x.title() for x in word_price_count_C_d["word_name"]], fontsize=12)
ax[2].set_title("Price Range: %s to %s ($USD)"%(C_lower, C_upper))


ax[3].barh(y_labels, word_price_count_D_d["avgprice"], facecolor="palevioletred",
          edgecolor="black")
ax[3].set_yticks(y_labels)
ax[3].set_yticklabels([x.title() for x in word_price_count_D_d["word_name"]], fontsize=12)
ax[3].set_title("Price Range: %s and above ($USD)"%(D_lower))
fig.subplots_adjust(wspace=0.6, right=1.9, top=1,bottom=0.1, left=-0.3)
fig.suptitle("Average of Prices Related to Word in Item Description \n 20 Randomly Picked Words for Each Price Range", y=1.15)
for axes in ax.flatten():
  axes.set_facecolor("navajowhite")
  axes.set_xlabel("Average Price ($USD)", fontsize=13)
  
  
  
fig.set_facecolor("floralwhite")

com_A_pricerange="\n".join((r"$\cdot$ " "The words in the item descriptions range from being descriptive, \n" \
                           "like Illuminated, Ecofriendly, Grained, Voldemort, to less descriptive, like \n" \
                            "Disappoint, Gen, Recieve. In general, the words seems to indicate some characteristic \n" \
                            "of the underlying product. For example, a combination of Ecofriendly and Burberry might \n" \
                            "indicate that the underlying product is not only of the luxury brand Burberry, but also an \n" \
                            "ecofriendly product"
                           r"$\cdot$ " "For a given product, all the words in its item description should collectively, \n" \
                           "with the average prices defined above, give a rough indication of a reasonable, possible price \n" \
                            "of the underlying product, for example by weighing the average prices of each word",
                           r"$\cdot$ " "A positive thing about the item description is that it may convey more \n" \
                           "indepth information of the underlying product as compared to the name attribute \n" \
                           "of the product"))

com_alg="\n".join((r"$\cdot$ " "Construction of the Average Price:", 
                  " For each possible Word in the Item Description:",
                  "   Collect all products which has that Word \n" \
                  "   in its Item Description", 
                  "   Calculate the average price from \n" \
                  "   all the prices of those products"))

box=dict(boxstyle="round", edgecolor="black", facecolor="wheat")
fig.text(-0.6, -0.66, com_A_pricerange, bbox=box,fontsize=13)
fig.text(0.9, -0.38, com_alg, bbox=box, fontsize=13)
