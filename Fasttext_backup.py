import pandas as pd
import numpy as np
import csv
import fasttext
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk import pos_tag
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer


# Add the Data using pandas
Corpus = pd.read_csv(r"C:\Users\gunjit.bedi\Desktop\Python\NLP Project\Consumer_Complaints_updated.csv",nrows=100,encoding='latin-1',skipinitialspace=True).replace(r'\\n','', regex=True)
# Step - 1a : Remove blank rows if any.
Corpus.dropna(inplace=True)

##Removing new line character
Corpus = Corpus.replace(r'\n','', regex=True)
Corpus = Corpus.replace(r'\\n','', regex=True)

#Step - 1b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
Corpus['text'] = [entry.lower() for entry in Corpus['text']]

Corpus.drop(['ID'],axis=1,inplace=True)

#Corpus['label'] = Corpus['label'].replace(' ', '_')
Corpus['label']=['__label__'+s.replace(' or ', '$').replace(', or ','$').replace(',','$').replace(' ','_').replace(',','_').replace('$$','$').replace('$','').replace('___','__') for s in Corpus['label']]
Corpus['label']
print("Check",Corpus['label'])
Corpus['text']= Corpus['text'].replace('\n',' ', regex=True).replace('\t',' ', regex=True)

# Step - 1c : Tokenization : In this each entry in the corpus will be broken into set of words
Corpus['text']= [word_tokenize(entry) for entry in Corpus['text']]


# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV


for index,entry in enumerate(Corpus['text']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.

    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    Corpus.loc[index,'text_final'] = str(Final_words)

Corpus = Corpus[['label','text_final','text']]

print("Text Pre-processing done.....")

#Split into train and test
train, test = train_test_split(Corpus, test_size=0.2)
#print(type(train))
train.drop(['text'],axis=1,inplace=True)
#print(train)

train.to_csv(r'C:\Users\gunjit.bedi\Desktop\Python\NLP Project\corpus.train.fasttext.txt', index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")
#test.to_csv(r'C:\Users\gunjit.bedi\Desktop\Python\NLP Project\corpus.test.fasttext.txt', index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")

classifier = fasttext.supervised(r'C:\Users\gunjit.bedi\Desktop\Python\NLP Project\corpus.train.fasttext.txt','model', label_prefix='__label__',thread=5,lr=.1,epoch=10,word_ngrams=2,bucket=4000000)

predictions_fasttext = classifier.predict(test.text_final)
test_label=list(test.label.replace({'__label__':''}, regex=True))

predictions_fasttext_train = classifier.predict(train.text_final)
train_label=list(train.label.replace({'__label__':''}, regex=True))

# Multi labels and their probabilities
# predictions_fasttext_list = classifier.predict(test.text_final,k=2)
# predictions_fasttext_list_prob = classifier.predict_proba(test.text_final,k=2)

print("Training data Accuracy -> ",accuracy_score(predictions_fasttext_train, train_label)*100)
print("Test data Accuracy -> ",accuracy_score(predictions_fasttext, test_label)*100)

#print(predictions_fasttext)
#print(test_label)

# test['predicted']=predictions_fasttext
# test['predicte_list']=predictions_fasttext_list
# test['predicted_list_prob']=predictions_fasttext_list_prob
# test.to_csv(r'C:\Users\gunjit.bedi\Desktop\Python\NLP Project\corpus_test_Predicted.csv')
