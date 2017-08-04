from flask import Flask,render_template,request
import tweepy
import re
import pandas as pd
from tweepy import OAuthHandler
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score




#======================================================= ======================================================
df = pd.read_csv("/home/saurabh/Sentiment_Analysis_Dataset.csv")
t = pd.DataFrame()
t['Sentiment'] = df.Sentiment
t['Text'] = df.SentimentText
#======================================================= ======================================================



stop_words = set(stopwords.words("english"))
vectorizer = TfidfVectorizer(use_idf = True, lowercase = True , strip_accents = 'ascii' , stop_words = stop_words )

X = vectorizer.fit_transform(t.Text)
y = t.Sentiment

X_train,X_test,y_train,y_test = train_test_split(X,y)

clf = naive_bayes.MultinomialNB()
clf.fit(X_train,y_train)

#======================================================= ======================================================


 
def classifier(queries):


		#===================================================================
		#
		query = queries
		tknzr=TweetTokenizer(strip_handles=True,reduce_len=True)
		consumer_key="YOUR_KEY"
		consumer_secret="YOUR SECRET_TOKEN"
		access_token="YOUR TOKEN"
		access_token_secret="TOKEN_SECRET"
		try:      
		    auth = OAuthHandler(consumer_key, consumer_secret)
		    auth.set_access_token(access_token, access_token_secret)
		            
		    api = tweepy.API(auth)
		    tweets_caught = api.search(q=query,count=5)

		  
		except:
		    print("Error")
		            
		#====================================================================

		#===========================cleaning tweet===========================
		count = 0
		text = []
		raw_tweet = []  
		for tweet in tweets_caught:
		    
		    clean_text = []
		    
		    


		    words = tknzr.tokenize(tweet.text)
		    
		    for w in words:
		        if w not in stop_words:
		            clean_text.append(w)
		            
		    str = " "
		    for w in clean_text:
		        str = str+w+" "
		       
		    URLless_str = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', str)
		    
		    if tweet.retweet_count > 0:
		        if URLless_str not in text:
		        	text.append(URLless_str)
		        	raw_tweet.append(tweet.text)
		        	count = count+1			
		    else:
		    	text.append(URLless_str)
		    	raw_tweet.append(tweet.text)
		    	count = count + 1

		    

		#    
		#======================================================================



		text_vec = vectorizer.transform(text)
		Resultant_Sentiment = clf.predict(text_vec)

		answer = pd.DataFrame()
		answer["tweet"] = raw_tweet
		answer["Sentiment"] = Resultant_Sentiment


		return answer



#======================================================= ======================================================


app = Flask(__name__)

@app.route('/')
def dir1():
    return render_template("profile.html")





@app.route('/sentiment' , methods = ['POST'])
def sentiment():
	queries = request.form['query']
	answer = classifier(queries)
	return render_template("sentiment.html",sentiments=answer)


if __name__ == '__main__':
    app.run()

#======================================================= ======================================================
