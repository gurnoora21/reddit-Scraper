import praw
import scipy
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import math
from psaw import PushshiftAPI
from pprint import pprint
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import datetime as dt
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.feature_extraction


start_epoch=int(dt.datetime(2020, 3, 20).timestamp())
reddit = praw.Reddit(client_id='Cr4dmtynEpDtmw', client_secret='ml7UbAzfDgmSd5fWjNxiuvyQvdg', user_agent='Reddit sentiment analysis')
api = PushshiftAPI(reddit)


with open("wordList.txt") as f:
    mylist = f.read().splitlines()


gen = list(api.search_submissions(after = start_epoch,
                            subreddit='Calgary',
                            limit=None))
headlines = set()

for filter in mylist:
    for head in gen: 
        if filter in head.title:
            headlines.add(head)
            gen.remove(head)


print(headlines)
def ConvertCheck(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

useless = ["http", "ca", "com", "org", "www"]

def UselessCheck(s):
    for i in useless: 
        if i in s: 
            return True 
    return False

def process_text(headlines):
    tokens = []
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = stopwords.words('english')
    toks = tokenizer.tokenize(headlines)
    toks = [t.lower() for t in toks if t.lower() not in stop_words]
    tokens.extend(toks)
    
    return tokens




sia = SIA()
results =[] 
keywordextract = set()
comment_dict = dict()
keywordtoken = list()
tokenizer = RegexpTokenizer(r'\w+')
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
totalscore = {}
propertable ={}
propertable["Headline"] = []
propertable["Compound"] = []
propertable["Sentiment"] = []

for line in headlines:
    submission = line
    #print(submission.title)
    comment_dict[submission.title] = []
    submission.comments.replace_more(limit=None)
    comment_queue = submission.comments[:]  # Seed with top-level
    totalscore[submission.title] = []
    while comment_queue:
        comment = comment_queue.pop(0)
        mid = tokenizer.tokenize(comment.body)
        mid2 = [t.lower() for t in mid if t.lower() not in stop_words]
        mid3 = [lemmatizer.lemmatize(q) for q in mid2]
        mid4 = [q for q in mid3 if not ConvertCheck(q)]
        mid5 = [q for q in mid4 if not UselessCheck(q)]
        comment_dict[submission.title].append(mid5)
        pol_score = sia.polarity_scores(comment.body)
        pol_score['comments'] = comment.body 
        totalscore[submission.title].append(pol_score['compound'])
        comment_queue.extend(comment.replies)
    
    if comment_dict[submission.title] == []:
        comment_dict.pop(submission.title)
    else:
        propertable["Headline"].append(submission.title)





print(propertable["Headline"])
avgScore = {}

tfidf = TfidfVectorizer(
	analyzer='word',
	tokenizer=lambda x: x,
	preprocessor=lambda x: x,
	lowercase=False
	)  
propertable['keywords'] = []  
for n in range (0, (len(propertable['Headline'])-30)):
    flatvector = [[item for items in v for item in items] for v in comment_dict.values()]
    tfidf_matrix = tfidf.fit_transform(flatvector)
    tfidf_array = tfidf_matrix.toarray()

    tfidf_vocab = tfidf.vocabulary_
    tfidf_vocab_inv = {v:k for k,v in tfidf_vocab.items()}
    argsort = np.argsort(tfidf_array[n])[::-1]
    features = np.arange(len(tfidf_array[n]))
    propertable['keywords'].append([(tfidf_vocab_inv[features[argsort][i]]) for i in range(5)])



for key,value in totalscore.items():
    if len(value) > 0 :
        avgScore[key] = sum(value)/float(len(value))
    else: 
        avgScore.pop(key, None)

new_dict = {}


for key in ["Headline"]:
    for value in propertable[key]:
        propertable["Compound"].append(avgScore[value])
        if avgScore[value] > 0.05:
            propertable["Sentiment"].append(1)
        elif avgScore[value] < -0.05:
            propertable["Sentiment"].append(-1)
        else:
            propertable["Sentiment"].append(0)

print(propertable["Compound"])
print(propertable["keywords"])




#for key, value in propertable.items():
    #propertable["Compound"].append(avgScore[value])


for key, value in comment_dict.items():
    new_dict[key] = [item for items in value for item in items]
    

#pprint(results[:], width=100)

df = pd.DataFrame.from_dict(propertable, orient='index')
df.to_csv('Heafasdfas.csv', mode='a', encoding='utf-8', index=False)
result = df.transpose()
result.to_csv('EdmontonResultsFINAL.csv', mode='a', encoding='utf-8', index=False)





#df = pd.DataFrame.from_records(avgScore)
#df.to_csv('score.csv', mode='a', encoding='utf-8')