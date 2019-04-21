import pandas as pd
import numpy as np
import csv
import fasttext

consumercompliants = pd.read_csv(r'C:\Users\gunjit.bedi\Desktop\Python\NLP Project\Consumer_Complaints.csv')

from io import StringIO
col = ['Product', 'Consumer complaint narrative']
consumercompliants = consumercompliants[col]
consumercompliants = consumercompliants[pd.notnull(consumercompliants['Consumer complaint narrative'])]
consumercompliants.columns = ['Product', 'Consumer_complaint_narrative']
consumercompliants.head()
consumercompliants.to_csv(r'C:\Users\gunjit.bedi\Desktop\Python\NLP Project\Consumer_Complaints_updated.csv')

#consumercompliants['Product'] = consumercompliants['Product'].replace(' ', '_', regex=True)
consumercompliants['Product']=['__label__'+s.replace(' or ', '$').replace(', or ','$').replace(',','$').replace(' ','_').replace(',','__label__').replace('$$','$').replace('$',' __label__').replace('___','__') for s in consumercompliants['Product']]
consumercompliants['Product']

consumercompliants['Consumer_complaint_narrative']= consumercompliants['Consumer_complaint_narrative'].replace('\n',' ', regex=True).replace('\t',' ', regex=True)

#consumercompliants['Consumer_complaint_narrative']=consumercompliants['Consumer_complaint_narrative'][1:-1]
consumercompliants.to_csv(r'C:\Users\gunjit.bedi\Desktop\Python\NLP Project\consumer.complaints.txt', index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")

classifier = fasttext.supervised(r'C:\Users\gunjit.bedi\Desktop\Python\NLP Project\consumer.complaints.txt','model', label_prefix='__label__',thread=8)
labels = classifier.predict(['Credit card lost'])
print(labels)
