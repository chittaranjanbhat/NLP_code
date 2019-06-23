import os
import sys
import argparse
import pandas as pd
import re
import string
import _pickle as cPickle
import numpy as np
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import WordNetLemmatizer
from nltk import wordpunct_tokenize
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


stopwords = set(sw.words('english'))
punctuation = set(string.punctuation)
lemmatizer = WordNetLemmatizer()
minlength = 3
maxlength = 25

def parseArguments():
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--training_datafile",required=False, help="path to training data file.")
    ap.add_argument("-m", "--modelpath",required=False, help="path to learned model file for testing.")
    ap.add_argument("-i", "--test_datafile",required=False, help="path to test data file.")
    ap.add_argument("-r", "--resultfile",required=False, help="result file name.")

    args = vars(ap.parse_args())
    for keys, values in args.items():
        print("Parsing arguments: {} : {}".format(keys,values))

    if args["training_datafile"] is None and args["modelpath"] is None:
        ap.error('-- either of one ( training_datafile/modelpath) is required')
    if args["modelpath"] is not None and args["test_datafile"] is None:
        ap.error('-- both modelpath and test_datafile are required for testing')
    return args

def getWornetPOS(tag):
    tagMap = {
            'N' : wn.NOUN,
            'V' : wn.VERB,
            'R' : wn.ADV,
            'J' : wn.ADJ
        }

    if tag[0] in tagMap.keys():
        return tagMap[tag[0]]
    else:
        return ''

def reviewToWords(review):
    review_text = BeautifulSoup(review,"html.parser").get_text()
    clean_review = re.sub("[^a-zA-Z]"," ", review_text)
    words = []

    for word, tag in pos_tag(wordpunct_tokenize(clean_review)):
        word = word.lower()
        word = word.strip()
        word = word.strip('_')
        word = word.strip('*')

        if word in stopwords:
            continue
        if all(char in punctuation for char in word):
            continue
        tag = getWornetPOS(tag)
        if tag=='':
            continue
        else:
            word = lemmatizer.lemmatize(word, tag)
        words.append(word)

    words = [w for w in words if minlength < len(w) < maxlength]
    return ( " ".join( words ))

def getDataFrame(datafile):
    print ("reading data file...")

    #data_df =  pd.read_csv(datafile, header=0, delimiter="\t", quoting=3)
    data_df =  pd.read_csv(datafile)
    print ("data shape: ", data_df.shape)
    print ("data colums: ", data_df.columns.values)
    data_df = data_df.astype(str)
    return data_df

def cleanDataFrame(dataframe):
    print ("pre-processing data...")

    for index, row in dataframe.iterrows():
        row['review'] = reviewToWords(row['review'])
        
    print(dataframe.head())
    return dataframe

def trainModel(training_set, pipeline):
    print ("learning model...")
    model = pipeline.fit(training_set["review"], training_set["classification"])
    return model

def predictModel(test_set, model):
    predicted = model.predict(test_set["review"])
    return predicted

def saveModel(model, filename="model.pkl"):
    with open(filename,'wb') as fid:
        cPickle.dump(model, fid)

def readModel(filename="model.pkl"):
    model = None
    if os.path.isfile(filename):
        with open(filename, 'rb') as fid:
            model = cPickle.load(fid)
    return model

if __name__ == '__main__':
    args = parseArguments()

    if args["training_datafile"] is not None:
        print ("-------Training-------")
        original_training_df = getDataFrame(args["training_datafile"])
        clean_training_df = cleanDataFrame(original_training_df.copy())

        print ("splitting data into training and validation set")
        training_set, validation_set = train_test_split(clean_training_df, test_size=0.3)
        print (training_set.shape)
        print (validation_set.shape)

        pipeline = Pipeline([
                ('count_vectorizer',CountVectorizer(ngram_range=(1, 2))),
                ('tfidf', TfidfTransformer()),
                ('classifier',BernoulliNB(binarize=0.0)) ])
 
        model = trainModel(training_set,pipeline)

        print ("----Validation and Classification Report----")
        predicted = predictModel(validation_set, model)
        target_classifications = list(set(validation_set["classification"]))
        print(classification_report(validation_set["classification"], predicted, target_names=target_classifications))

        print ("----Saving Model----")
        if args["modelpath"] is not None:
            saveModel(model, args["modelpath"])
        else:
            saveModel(model)
    if args["modelpath"] is not None and args["test_datafile"] is not None:
        print ("----Testing----")
        model = readModel(args["modelpath"])
        original_test_df = getDataFrame(args["test_datafile"])
        clean_test_df = cleanDataFrame(original_test_df.copy())

        predicted = predictModel(clean_test_df,model)

        print (original_test_df["review"][0])
        print (clean_test_df["review"][0])
        print ("----prediction and Classification Report----")
        
        target_classifications = list(set(clean_test_df["classification"]))
        print(classification_report(clean_test_df["classification"], predicted, target_names=target_classifications))

        print ("----Saving Results----")
        output = pd.DataFrame( data={"review":original_test_df['review'], "predicted":predicted} )
        if args["resultfile"] is not None:
            resultfile = args["resultfile"]
        else:
            resultfile = "result.tsv"
        #output.to_csv(resultfile, index=False, sep='\t', quoting=3 )
        output.to_csv(resultfile)

# pipeline = Pipeline([('vect', CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None)),
#                            ('tfidf', TfidfTransformer()),
#                            ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))])
#
#
# pipeline = Pipeline([
# ('count_vectorizer',   CountVectorizer(ngram_range=(1, 2))),
# ('tfidf', TfidfTransformer()),
# ('classifier',         BernoulliNB(binarize=0.0)) ])
#
#
# pipeline = Pipeline([
# ('count_vectorizer',   CountVectorizer(ngram_range=(1, 2))),
# ('tfidf', TfidfTransformer()),
# ('classifier', RandomForestClassifier(n_estimators = 100))])