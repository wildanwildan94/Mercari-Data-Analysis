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
name_d=train_d[["name"]]

# (a) Extract all words

#split
word_array=[]

for index, rows in name_d.iterrows():
  if index==int(N/float(10)):
    print "1/10 done"
    
  if index==int(N/float(5)):
    print "1/5 done"
  if index==int(N/float(2)):
    print "1/2 done"
    
  temp_name=rows["name"]
  
  temp_words=temp_name.split()
  temp_words=[x.lower() for x in temp_words]
  word_array.extend(temp_words)
  
np.save('word_array.npy', word_array)
## Want to analyze the distribution of name, load data, and save as csv
word_array = np.load('word_array.npy')

word_count=Counter(word_array)

word_name_array, word_count_array=zip(*word_count.items())

word_count_d=pd.DataFrame({'word_name':word_name_array,
                          'word_count':word_count_array})

word_count_d.to_csv('word_count.csv', index=False)

## Want to analyze the distribution of name, visualize the top
## twenty keywords

word_count_d=pd.read_csv('word_count.csv')

# (a) Print the form
print word_count_d.head(3)

word_name_ex=word_count_d["word_name"].iloc[0]
print word_name_ex


# (b) Compute top 20 words in name attribute

word_count_top_d=word_count_d.sort_values(by="word_count", ascending=False).head(20).sort_values(by="word_count", ascending=True)

amount_words=word_count_d.shape[0]
# (c) Visualize top 20 words

fig, ax = plt.subplots()


# Visualize top ten of categories by count
y_labels=range(word_count_top_d.shape[0])

ax.barh(y_labels, word_count_top_d["word_count"], facecolor="palevioletred",
       edgecolor="black")
ax.set_yticks(y_labels)
ax.set_yticklabels(word_count_top_d["word_name"], fontsize=12)
ax.set_title("Top 20 words by amount of products \nwith word in name")
ax.set_facecolor("navajowhite")
fig.subplots_adjust(top=1, hspace=0.4)

fig.set_facecolor("floralwhite")
com_res="\n".join((r"$\cdot$ " "A lot of descriptive words in the top 20 \n" \
                  "words in the name attribute",
                  r"$\cdot$ " "For example pink, bundle, black, dress, \n" \
                  "leggings, top, shirt",
                  r"$\cdot$ " "Indication that the words in the name attribute \n" \
                  "can describe the underlying product (e.g pink shirt)",
                 r"$\cdot$ " "There exists a few non-descriptive words \n" \
                  "like vs, &, free, but they are relatively few",
                  r"$\cdot$ " "A conclusion is that the words in the name \n" \
                  "could potentially be used, in some way, to model \n" \
                  "what type of product is underlying the name",
                  r"$\cdot$ " "Drawbacks include the large amount of unique \n" \
                  "words, which is at 198515, and that the name \n" \
                  "attribute contains a lot of special characters, \n" \
                  "which can distort the meaning of a word"))


for x_val, y in zip(word_count_top_d["word_count"], y_labels):
  ax.text(x_val+400, y-0.3, str(x_val), fontweight="bold")
  
box=dict(boxstyle="round", edgecolor="black", facecolor="wheat")
fig.text(0.92,0.14, com_res, fontsize=13, bbox=box)
ax.set_xlim((0, 93000))

ax.text(5.6e04, 8, "Amount of \nUnique Words: %s"%str(amount_words),
       bbox=dict(boxstyle="round", edgecolor="black", facecolor="white"))
count_patch=mpatches.Patch(color="black", label="Amount of Products \nwith Word in Name",
                          facecolor="white")
ax.legend(handles=[count_patch])
