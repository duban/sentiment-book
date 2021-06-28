from flask import Flask
from flask import Flask, request
from flask.json import jsonify
import pickle
import re
import os
from os import sys, path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

app = Flask(__name__)

def processText(text):
    # process the tweets

    # Convert to lower case
    text = text.lower()
    # Convert www.* or https?://* to URL
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text)
    # Convert @username to AT_USER
    text = re.sub('@[^\s]+', 'AT_USER', text)
    # Remove additional white spaces
    text = re.sub('[\s]+', ' ', text)
    # Replace #word with word
    text = re.sub(r'#([^\s]+)', r'\1', text)
    # trim
    text = text.strip('\'"')
    return text

#start extract_features

word_features5k_f = open("word_features_v1.pickle", "rb")
featureList = pickle.load(word_features5k_f)
word_features5k_f.close()
#start getfeatureVector
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)

def getFeatureVector(tweet, stopWords):
    featureVector = []
    words = tweet.split()
    for w in words:
        #replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if it consists of only words
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*[a-zA-Z]+[a-zA-Z0-9]*$", w)
        #ignore if it is a stopWord
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return featureVector
#start getStopWordList
def getStopWordList(stopWordListFileName):
    #read the stopwords
    stopWords = []
    stopWords.append('AT_USER')
    stopWords.append('URL')

    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords

def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features

open_file = open("book_naivebayes_v1.pickle", "rb")
NBClassifier = pickle.load(open_file)
open_file.close()
@app.route("/api/analisis", methods=['POST'])
def get_sentiment():
    stopWords = getStopWordList('data/feature_list/stopwordsID.txt')
    text = request.args.get('text')
    processedTestTweet = processText(text)
    sentiment = NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet, stopWords)))
    # print ("Text = %s, sentiment = %s\n" % (testTweet, sentiment))
    resp = {"text":text,"sentiment":sentiment}
    # print(resp)
    return jsonify(resp)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)