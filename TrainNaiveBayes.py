#import regex
import re
import csv
import pprint
import nltk.classify
import pickle

#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL) 
    return pattern.sub(r"\1\1", s)
#end

#start process_tweet
def processTweet(tweet):
    # process the tweets
    
    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)    
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet
#end 

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
#end

#start getfeatureVector
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
#end

#start extract_features
def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features
#end


#Read the tweets one by one and process it
# with open('data/sampleTweetsID.csv') as csvfile:
#     inpTweets = csv.reader(csvfile, delimiter=',', quotechar='|')
inpTweets = csv.reader(open('data/test.csv', 'r'), delimiter=',')
stopWords = getStopWordList('data/feature_list/stopwordsID.txt')
count = 0;
featureList = []
tweets = []

for row in inpTweets:
    sentiment = row[0]
    tweet = row[1]
    processedTweet = processTweet(tweet)
    featureVector = getFeatureVector(processedTweet, stopWords)
    featureList.extend(featureVector)
    tweets.append((featureVector, sentiment));
#end loop
# print (tweets)

# Remove featureList duplicates
featureList = list(set(featureList))

save_word_features = open("word_features_v1.pickle","wb")
pickle.dump(featureList, save_word_features)
save_word_features.close()

# Generate the training set
training_set = nltk.classify.util.apply_features(extract_features, tweets)



# Train the Naive Bayes classifier
NBClassifier = nltk.NaiveBayesClassifier.train(training_set)

NBClassifier.show_most_informative_features(50)
print("Naive Bayes Algo accuracy percent :", (nltk.classify.accuracy(NBClassifier, training_set))*100)

save_classifier = open("book_naivebayes_v1.pickle","wb")
pickle.dump(NBClassifier, save_classifier)
save_classifier.close()


# Test the classifier
# testTweet = 'Hari yang mengecewakan. Menghadiri pameran mobil untuk mencari pendanaan, harganya malah lebih mahal'
# processedTestTweet = processTweet(testTweet)
# sentiment = NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet, stopWords)))
# print ("testTweet = %s, sentiment = %s\n" % (testTweet, sentiment))


# kal = getFeatureVector(processTweet(testTweet),stopWords)
# kal = " ".join(str(x) for x in kal)
# print kal
# d = {}
# for word in kal.split():
#     word = int(word) if word.isdigit() else word
#     if word in d:
#         d[word] += 1
#     else:
#         d[word] = 1

# print d