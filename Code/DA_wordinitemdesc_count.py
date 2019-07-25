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
item_desc_d=train_d[["item_description"]]

# (a) Extract all words in item descriptions and with count


word_array=[]
count_word_array=Counter(word_array)

for index, rows in item_desc_d.iterrows():
  if index==int(N/float(10)):
    print "1/10 done"
    
  if index==int(N/float(5)):
    print "1/5 done"
  if index==int(N/float(2)):
    print "1/2 done"
  if index==int(3*N/float(4)):
    print "3/4 done"
    
  temp_name=str(rows["item_description"])
  
  temp_words=temp_name.split()
  temp_words=[x.lower() for x in temp_words]
  for temp_word in temp_words:
    count_word_array[temp_word]+=1
  
word_name_array, word_count_array=zip(*count_word_array.items())
word_name_array=np.array([x for x in word_name_array])
word_count_array=np.array([x for x in word_count_array])

print word_count_array[0:30]


np.save('item_desc_word_name_array.npy', word_name_array)
np.save('item_desc_word_count_array.npy', word_count_array)

## Want to analyze the distribution of item_descritipn, load data, and save as csv
word_name_array = np.load('item_desc_word_name_array.npy')
print word_name_array[0:30]
word_count_array=np.load('item_desc_word_count_array.npy')

print word_count_array.shape
print type(word_name_array)
print type(word_count_array)
print len(word_name_array)
print len(word_count_array)


word_count_d=pd.DataFrame({'word_name':word_name_array,
                          'word_count':word_count_array})

word_count_d.to_csv('item_desc_word_count.csv', index=False)
## Want to analyze the distribution of item_descritipn, visualize top 20 words


item_desc_word_count_d=pd.read_csv('item_desc_word_count.csv')

# (a) Print the form
print item_desc_word_count_d.head(3)


# (b) Compute top 20 words in name attribute

item_desc_word_count_top_d=item_desc_word_count_d.sort_values(by="word_count", ascending=False).head(20).sort_values(by="word_count", ascending=True)

amount_words=item_desc_word_count_d.shape[0]

# (c) Visualize top 20 words

fig, ax = plt.subplots()


# Visualize top ten of categories by count
y_labels=range(item_desc_word_count_top_d.shape[0])

x_labels=[str(int(x)/float(1000))+ "e03" for x in item_desc_word_count_top_d["word_count"]]
print x_labels[0:5]
ax.barh(y_labels, item_desc_word_count_top_d["word_count"], facecolor="palevioletred",
       edgecolor="black")
ax.set_yticks(y_labels)
ax.set_yticklabels(item_desc_word_count_top_d["word_name"], fontsize=12)
ax.set_title("Top 20 words by amount of products \nwith word in item description")
ax.set_facecolor("navajowhite")
fig.subplots_adjust(top=1, hspace=0.4, wspace=0.4, right=1.2)

fig.set_facecolor("floralwhite")
com_res="\n".join((r"$\cdot$ " "A lot of non-descriptive words in the top 20 \n" \
                  "words in the item description",
                  r"$\cdot$ " "Might imply that utilizing words in \n" \
                  "the item description might not be useful, as it contains \n" \
                  "a lot of sentence-building words, like and, the, size, brand, \n" \
                  "free, on",
                 r"$\cdot$ " "In addition, a lot of words contain special \n" \
                  "characters which might distort the meaning of words",
                  r"$\cdot$ " "However, all the words in the item description \n" \
                  "might collectively convey useful information, for example \n" \
                  "if the words 'size' appears with '8', it might indicate \n" \
                  "that the underlying product is a clothing piece with \n" \
                  "size 8. This type of inference should be able to convey \n" \
                  "some information on the possible price of the underlying item",
                  r"$\cdot$ " "Hence, the words collectively might convey \n" \
                  "a lot of useful information"))


for x_val, y in zip(item_desc_word_count_top_d["word_count"], y_labels):
  ax.text(x_val+5000, y-0.28, str(x_val), fontweight="bold")
  
box=dict(boxstyle="round", edgecolor="black", facecolor="wheat")
fig.text(1.23,0.14, com_res, fontsize=13, bbox=box)
ax.set_xlim((0, 930000))

ax.text(5e05, 8, "Amount of \nUnique Words: %s"%str(amount_words),
       bbox=dict(boxstyle="round", edgecolor="black", facecolor="white"))
count_patch=mpatches.Patch(color="black", label="Amount of Products \nwith Word in Item Description",
                          facecolor="white")
ax.legend(handles=[count_patch])

