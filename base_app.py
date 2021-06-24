"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import re # for regular expressions
import string
special = string.punctuation 
import nltk # for text manipulation

import warnings 
warnings.filterwarnings("ignore")

#Importing other libraries
from collections import defaultdict
from collections import  Counter

# Style
import matplotlib.style as style 
from wordcloud import WordCloud
from sklearn import datasets
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize 
from gensim.parsing.preprocessing import remove_stopwords

# Standard libraries
import re
import string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
count_vect = open("resources/CountVectorizer.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file
Count_v = joblib.load(count_vect)
# Load your raw data
raw = pd.read_csv("resources/train.csv")


# Apostrophe Dictionary
apostrophe_dict = {
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"i'd": "I would",
"i'd've": "I would have",
"i'll": "I will",
"i'll've": "I will have",
"i'm": "I am",
"i've": "I have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"It's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that had",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

#Dictionary for short words in tweets
short_word_dict = {
"121": "one to one",
"a/s/l": "age, sex, location",
"adn": "any day now",
"afaik": "as far as I know",
"afk": "away from keyboard",
"aight": "alright",
"alol": "actually laughing out loud",
"b4": "before",
"b4n": "bye for now",
"bak": "back at the keyboard",
"bf": "boyfriend",
"bff": "best friends forever",
"bfn": "bye for now",
"bg": "big grin",
"bta": "but then again",
"btw": "by the way",
"cid": "crying in disgrace",
"cnp": "continued in my next post",
"cp": "chat post",
"cu": "see you",
"cul": "see you later",
"cul8r": "see you later",
"cya": "bye",
"cyo": "see you online",
"dbau": "doing business as usual",
"fud": "fear, uncertainty, and doubt",
"fwiw": "for what it's worth",
"fyi": "for your information",
"g": "grin",
"g2g": "got to go",
"ga": "go ahead",
"gal": "get a life",
"gf": "girlfriend",
"gfn": "gone for now",
"gmbo": "giggling my butt off",
"gmta": "great minds think alike",
"h8": "hate",
"hagn": "have a good night",
"hdop": "help delete online predators",
"hhis": "hanging head in shame",
"iac": "in any case",
"ianal": "I am not a lawyer",
"ic": "I see",
"idk": "I don't know",
"imao": "in my arrogant opinion",
"imnsho": "in my not so humble opinion",
"imo": "in my opinion",
"iow": "in other words",
"ipn": "I’m posting naked",
"irl": "in real life",
"jk": "just kidding",
"l8r": "later",
"ld": "later, dude",
"ldr": "long distance relationship",
"llta": "lots and lots of thunderous applause",
"lmao": "laugh my ass off",
"lmirl": "let's meet in real life",
"lol": "laugh out loud",
"ltr": "longterm relationship",
"lulab": "love you like a brother",
"lulas": "love you like a sister",
"luv": "love",
"m/f": "male or female",
"m8": "mate",
"milf": "mother I would like to fuck",
"oll": "online love",
"omg": "oh my god",
"otoh": "on the other hand",
"pir": "parent in room",
"ppl": "people",
"r": "are",
"rofl": "roll on the floor laughing",
"rpg": "role playing games",
"ru": "are you",
"shid": "slaps head in disgust",
"somy": "sick of me yet",
"sot": "short of time",
"thanx": "thanks",
"thx": "thanks",
"ttyl": "talk to you later",
"u": "you",
"ur": "you are",
"uw": "you’re welcome",
"wb": "welcome back",
"wfm": "works for me",
"wibni": "wouldn't it be nice if",
"wtf": "what the fuck",
"wtg": "way to go",
"wtgp": "want to go private",
"ym": "young man",
"gr8": "great"
}
#Dictionary to change emojis to words
emoticon_dict = {
":)": "happy",
":‑)": "happy",
":-]": "happy",
":-3": "happy",
":->": "happy",
"8-)": "happy",
":-}": "happy",
":o)": "happy",
":c)": "happy",
":^)": "happy",
"=]": "happy",
"=)": "happy",
"<3": "happy",
":-(": "sad",
":(": "sad",
":c": "sad",
":<": "sad",
":[": "sad",
">:[": "sad",
":{": "sad",
">:(": "sad",
":-c": "sad",
":-< ": "sad",
":-[": "sad",
":-||": "sad"
}
#Function to change apostrophy to actual words 
def lookup_dict(text, dictionary):
    for word in text.split():
        if word.lower() in dictionary:
            if word.lower() in text.split():
                text = text.replace(word, dictionary[word.lower()])
    return text

#DATA ClEANING
def TweetCleaner(tweet):
    
    #This function uses regular expressions to remove url's, mentions, hashtags, 
    #punctuation, numbers and any extra white space from tweets after converting everything to lowercase letters.
    
    # Convert everything to lowercase
    tweet = tweet.lower()
    #apostrophe
    tweet = lookup_dict(tweet,apostrophe_dict)
    #short_word
    tweet = lookup_dict(tweet,short_word_dict)
    #emotion
    tweet = lookup_dict(tweet,emoticon_dict)
    #Remove stop words
    tweet = remove_stopwords(tweet)
    # Remove mentions   
    tweet = re.sub('@[\w]*','',tweet)  
    # Remove url's
    tweet = re.sub(r'https?:\/\/.*\/\w*', '', tweet)
    # Remove hashtags
    tweet = re.sub(r'#\w*', '', tweet)    
    # Remove numbers
    tweet = re.sub(r'\d+', '', tweet)  
    # Remove punctuation
    tweet = re.sub(r"[,.;':@#?!\&/$]+\ *", ' ', tweet)
    # Remove that funny diamond
    tweet = re.sub(r"U+FFFD ", ' ', tweet)
    # Remove extra whitespace
    tweet = re.sub(r'\s\s+', ' ', tweet)
    # Remove space in front of tweet
    tweet = tweet.lstrip(' ')
    # Remove emojis
    emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u'\U00010000-\U0010ffff'
                u"\u200d"
                u"\u2640-\u2642"
                u"\u2600-\u2B55"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\u3030"
                u"\ufe0f"
    "]+", flags=re.UNICODE)
    tweet = emoji_pattern.sub(r'', tweet)
    return tweet

# Clean the tweets in the message column
raw['clean_message'] = raw['message'].apply(TweetCleaner)
raw['clean_message'] = raw['clean_message'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>1]))

# function to plot the target _ distribution
fig1, axes = plt.subplots(ncols=2, figsize=(17, 4), dpi=100)
plt.tight_layout()
raw.groupby('sentiment').count()['tweetid'].plot(kind='pie', ax=axes[0], labels=['Anti','Neutral','Pro','News'],autopct='%.0f%%')
sns.countplot(x=raw['sentiment'], hue=raw['sentiment'], ax=axes[1])
axes[0].set_ylabel('')
axes[1].set_ylabel('')
axes[1].set_xticklabels(['Anti','Neutral','Pro','News'])
axes[0].tick_params(axis='x', labelsize=15)
axes[0].tick_params(axis='y', labelsize=15)
axes[1].tick_params(axis='x', labelsize=15)
axes[1].tick_params(axis='y', labelsize=15)
axes[0].set_title('Target Distribution in Training Set', fontsize=13)
axes[1].set_title('Target Count in Training Set', fontsize=13)

# function to plot the character _ distribution
fig2,((ax1, ax2), (ax3, ax4)) =plt.subplots(2,2,figsize=(20,10))
tweet_len=raw[raw['sentiment']==2]['message'].str.len() 
ax1.hist(tweet_len,color='red')
ax1.set_title('News')
tweet_len=raw[raw['sentiment']==1]['message'].str.len()
ax2.hist(tweet_len,color='green')
ax2.set_title('Pro')
tweet_len=raw[raw['sentiment']==0]['message'].str.len()
ax3.hist(tweet_len,color='blue')
ax3.set_title('Neutral')
tweet_len=raw[raw['sentiment']==-1]['message'].str.len()
ax4.hist(tweet_len,color='purple')
ax4.set_title('Anti')
fig2.suptitle('Characters In Tweets')

#Word length
fig,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2,figsize=(15,10))
tweet_len=raw[raw['sentiment']==2]['message'].str.split().map(lambda x: len(x))
ax1.hist(tweet_len,color='red')
ax1.set_title('News')
tweet_len=raw[raw['sentiment']==1]['message'].str.split().map(lambda x: len(x))
ax2.hist(tweet_len,color='green')
ax2.set_title('Pro')
tweet_len=raw[raw['sentiment']==0]['message'].str.split().map(lambda x: len(x))
ax3.hist(tweet_len,color='blue')
ax3.set_title('Neutral')
tweet_len=raw[raw['sentiment']==-1]['message'].str.split().map(lambda x: len(x))
ax4.hist(tweet_len,color='purple')
ax4.set_title('Anti')
fig.suptitle('Words In Tweets')


news_tweets = raw[raw['sentiment']==2]['clean_message'] #select only news-tweets
Pro_tweets = raw[raw['sentiment']==1]['clean_message'] #select only pro-tweets
neutral_tweets = raw[raw['sentiment']==0]['clean_message'] #select only neutral-tweets
anti_tweets = raw[raw['sentiment']==-1]['clean_message'] #select only anti-tweets
alll = raw['clean_message'] #select all tweets
    
fig3, ((ax1, ax2,ax3,ax4,ax5)) = plt.subplots(5, 1, figsize=[40, 30]) 
#Code for Neutral Tweets
wordcloud1 = WordCloud( background_color='white',width=1400,height=400).generate(" ".join(neutral_tweets)) 
ax1.imshow(wordcloud1)
ax1.axis('off')
ax1.set_title('Neutral Tweets',fontsize=30)
#Code for Anti Tweets
wordcloud2 = WordCloud( background_color='white',width=1400,height=400).generate(" ".join(anti_tweets)) 
ax2.imshow(wordcloud2)
ax2.axis('off')
ax2.set_title('Anti Tweets',fontsize=30)
#Code for News Tweets
wordcloud3 = WordCloud( background_color='white',width=1400,height=400).generate(" ".join(news_tweets)) 
ax3.imshow(wordcloud3)
ax3.axis('off')
ax3.set_title('News Tweets',fontsize=30)
#Code for Pro Tweets
wordcloud4 = WordCloud( background_color='white',width=1400,height=400).generate(" ".join(Pro_tweets)) 
ax4.imshow(wordcloud4)
ax4.axis('off')
ax4.set_title('Pro Tweets',fontsize=30)
#Code for All Tweets
wordcloud5 = WordCloud( background_color='white',width=1400,height=400).generate(" ".join(alll)) 
ax5.imshow(wordcloud5)
ax5.axis('off')
ax5.set_title('All Tweets',fontsize=30)

# The main function where we will build the actual app
def main():
    
	""" Tweet Classifier App Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Problem Statement", "Information", "EDA", "Model Description", "Prediction"]
	selection = st.sidebar.selectbox("Choose Option", options)
	# Building out the "Introduction" page

	# Building out the "Problem Statement" page
	if selection == "Problem Statement":
		st.info("Problem Statement")
		st.markdown("Our task is to create a machine learning model that is able to clarify whether or not a person believes in climate change, based on their novel tweet data. To increase Thrive Market’s advertising efficiency by using machine learning to create effective marketing tools that can identify whether or not a person believes in climate change and could possibly be converted to a new customer based on their tweets.")
		st.markdown("""![ChessUrl](https://i.gifer.com/Q8k0.gif "Problem Statement")""")
        
	# Building out the "EDA" page
	if selection == "EDA":
		st.info("Exploratory Data Analysis")
		st.markdown("""![ChessUrl](https://images.theconversation.com/files/213371/original/file-20180405-189801-1wbjtyg.jpg?ixlib=rb-1.1.0&q=45&auto=format&w=496&fit=clip "EDA")""")
		st.subheader("Percentage of Sentimentals (2,1,0 & -1) using Pie chart and Bar graph")
		st.pyplot(fig1)
		st.markdown("Target distribution percentages info:\n\n -54 percentage of our tweets shows that they believe in man-made climate change(Pro = 1) \n\n -23 percentage of our tweets shows the news about man-made climate change (News = 2) \n\n -15 percentage of our tweets shows that they neither believe or do not believe in man-made climate change (Neutral = 0) \n\n-8 percentage of our tweets show that they do not believe in man-made climate change(Anti = -1) \n\n -Imbalanced: The sentiment are not balanced. Which means when we predict our sentiment it will be leaning towards the majority sentiment")
		st.subheader("\n\n AFTER RESAMPLING")
		st.image('./Screenshot 2021-06-24 155126.png') 
		st.subheader("\n\n WORD DISTRIBUTION")
		st.pyplot(fig)
		st.markdown("Observations:\n\n -The distribution of words in tweets seems to be almost the similar. \n\n -The distribution of all sentiment are fairly symmentrical. \n\n -There are no words which are more than 40 characters")
		st.subheader("\n\n CHARACTER DISTRIBUTION")
		st.pyplot(fig2)
		st.markdown("Observations: \n\n -The distribution of characters  seems to be almost the similar. \n\n -130 to 140 character length in a tweet are the most common among all sentiments. \n\n -The distribution of all sentiment are skewed to the left.")
		st.subheader("\n\n WORD CLOUD OF ALL OUR TWEETS")
		st.pyplot(fig3)
		st.markdown("Observations:\n\n -Neutral Tweets: The words like climate change, global warming appear to be big which means that they are mosst coomon words in our neutral tweets. \n\n -Anti Tweets: Donald Trump appears a lot here. He is probably one of the people who do not believe in global warming. Also words like blame and scam are probably unique in here. \n\n -News Tweets: Words like fight climate appear a lot. \n\n -Pro Tweets: Words like tackle climate, change are probably unique in here too. \n\n -All Tweets: As expected, words like climate change and global warming are more visible/common.")
		st.subheader("\n\n PUNCTUATION FREQUENCY")
		st.image('./Screenshot 2021-06-24 154831.png')
		st.subheader("\n\n WORD CLOUD OF PEOLPLE IN OUR TWEETS")
		st.image('./Screenshot 2021-06-24 155559.png')
		st.image('./Screenshot 2021-06-24 155541.png')
		st.markdown("Observations:\n\n -ANTI INFO : We get to see that most mentioned people in our anti tweets are George,Gore and Bill.\n\n -PRO INFO : Hillary, Trump and Clinton are most mentioned names in our Pro tweets\n\n-NEUTRAL INFO : Mike, Hillary and Malcom are most mentioned names\n\n-NEWS INFO : Scott, John and Trump are most mentioned names in our News Tweets, it could be because they are popular or might have a bigger influence to people's lives world wide")
		st.subheader("\n\n WORD CLOUD OF GEOPOLITICS IN OUR TWEETS")
		st.image('./Screenshot 2021-06-24 155438.png')
		st.image('./Screenshot 2021-06-24 155416.png')
		st.markdown("Observations:\n\n-ANTI INFO : We get to see that most mentioned places in our anti tweets are Iran,Russia and Paris.\n\n-PRO INFO : California,India and Florida are most mentioned places in our Pro tweets\n\n-NEUTRAL INFO : France, Washington and Germany are most mentioned places\n\n-NEWS INFO : China, Australia and Chicago are most mentioned places in our News Tweets")
		st.subheader("\n\n WORD CLOUD OF ORGANIZATIONS IN OUR TWEETS")
		st.image('./Screenshot 2021-06-24 155304.png')
		st.image('./Screenshot 2021-06-24 155241.png') 
		st.markdown("Observations:\n\n-ANTI INFO : We get to see that most mentioned Organizations in our anti tweets are FOX,Climate and CNN.\n\n-PRO INFO : GOP, BBC and EXXON are most named organizations in our Pro tweets\n\n-NEUTRAL INFO : We get the name of countries and cities which probably mean that most organisations are named after their countries\n\n-NEWS INFO : The words, like news, energy and department appear most ")       
		st.subheader("\n\n Most Common Hashtags")
		st.image('./Screenshot 2021-06-24 155210.png')
		st.markdown("Observations:\n\n-As expected the most popular hashtag is #climate and also #climatechange\n\n-#BeforeTheFlood, this hashtag is probably about a disater that is happening because of global warming\n\n-#ActOnClimate, this hashtag is probably about an action that has to be done to deal or reduce global warming")
		st.subheader("\n\n APlotting Most Frequent N-grams")
		st.image('./Screenshot 2021-06-24 155017.png')
		st.subheader("\n\n MOST COMMON WORDS")
		st.image('./Screenshot 2021-06-24 154931.png')        
        
	# Building out the "Model Description" page
	if selection == "Model Description":
		st.info("Description")
		st.subheader("Logistic Regression")
		st.markdown("Logistic Regression uses the probability of a data point to belonging to a certain class to classify each datapoint to it's best estimated class. The figure below is the sigmoid function logistic regression models use to make predictions: ![1*a5QwiyaSyvRa6n3VKYVEnQ.png](https://cdn-media-1.freecodecamp.org/images/1*a5QwiyaSyvRa6n3VKYVEnQ.png)")
		st.subheader("Stochastic Gradient Descent - SGD Classifier")
		st.markdown("Stochastic Gradient Descent (SGD) is a simple yet efficient optimization algorithm used to find the values of parameters/coefficients of functions that minimize a cost function. In other words, it is used for discriminative learning of linear classifiers under convex loss functions such as SVM and Logistic regression. It has been successfully applied to large-scale datasets because the update to the coefficients is performed for each training instance, rather than at the end of instances. Stochastic Gradient Descent (SGD) classifier basically implements a plain SGD learning routine supporting various loss functions and penalties for classification. Logistic regression has been rated as the best performing model for linearly separable data especially if it's predicting binary data(Yes & NO or 1 & 0), and performs better when there's no class imbalance.")
		st.subheader("Support Vector Machine")
		st.markdown("In the SVM algorithm, we plot each data item as a point in n-dimensional space (where n is number of features you have) with the value of each feature being the value of a particular coordinate. The goal of the SVM algorithm is to create the best line or decision boundary that can seperate n-dimensional space into classes so that we can easily put the new data point in the correct category in the future. This best decision boundary is called a hyperplane.SVM chooses the extreme points/vectors that help in creating the hyperplane. These extreme cases are called as support vectors. Consider the below diagram in which there are two different categories that are classified using a decision boundary or hyperplane. SVM maps training examples to points in space so as to maximise the width of the gap between the two categories. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall. In addition to performing linear classification, SVMs can efficiently perform a non-linear classification using what is called the kernel trick, implicitly mapping their inputs into high-dimensional feature spaces. ![support-vector-machine-algorithm.png](https://static.javatpoint.com/tutorial/machine-learning/images/support-vector-machine-algorithm.png)")

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		option = ["Stochastic Gradient Descent", "Logistic Regression","Support Vector Machines"]
		select = st.selectbox("Choose Model", option)
     
		if select =="Logistic Regression": 
			# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")
			tweet_text = TweetCleaner(tweet_text)
			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = Count_v.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				# Creating sidebar with selection box -
				# you can create multiple pages this way
				predictor = joblib.load(open(os.path.join("resources/Logistics_Regression_AM5_DSFT21.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				if prediction == -1:
					st.success("Text Categorized as:  Anti Climate Change")
				if prediction == 0:
					st.success("Text Categorized as:  Neutral, Neither Believe Nor Disputes Climate Change")
				if prediction == 1:
					st.success("Text Categorized as:  Pro Climate Change, Believe In Climate Change")
				if prediction == 2:
					st.success("Text Categorized as:  Factual News On Climate Change")      

		if select =="Stochastic Gradient Descent":
			# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")
			tweet_text = TweetCleaner(tweet_text)
			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = Count_v.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				# Creating sidebar with selection box -
				# you can create multiple pages this way
				predictor = joblib.load(open(os.path.join("resources/SGD_Classifier_AM5_DSFT21.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				if prediction == -1:
					st.success("Text Categorized as:  Anti Climate Change")
				if prediction == 0:
					st.success("Text Categorized as:  Neutral, Neither Believe Nor Disputes Climate Change")
				if prediction == 1:
					st.success("Text Categorized as:  Pro Climate Change, Believe In Climate Change")
				if prediction == 2:
					st.success("Text Categorized as:  Factual News On Climate Change")
            
		if select == "Support Vector Machines":
			# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")
			tweet_text = TweetCleaner(tweet_text)
			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = Count_v.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				# Creating sidebar with selection box -
				# you can create multiple pages this way
				predictor = joblib.load(open(os.path.join("resources/Support_Vector_Machine_AM5_DSFT21.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				if prediction == -1:
					st.success("Text Categorized as:  Anti Climate Change")
				if prediction == 0:
					st.success("Text Categorized as:  Neutral, Neither Believe Nor Disputes Climate Change")
				if prediction == 1:
					st.success("Text Categorized as:  Pro Climate Change, Believe In Climate Change")
				if prediction == 2:
					st.success("Text Categorized as:  Factual News On Climate Change")
        
# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Given the recent explosion of Big Data, there is a growing demand for analyzing non traditional data sources. Social Media data is a huge source of this data in the form of chats, messages, news feeds and all of it is in an unstructured form. Text analytics is a process that helps analyze this unstructured data and look for patterns or infer popular sentiment which can help organizations in their decision making.Twitter data is a powerful source of information on a wide list of topics. This data can be analyzed to find trends related to specific topics, measure popular sentiment, obtain feedback on past decisions and also help make future decisions. Climate change has received extensive attention from public opinion in the last couple of years, after being considered for decades as an exclusive scientific debate")
		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

		st.subheader("Cleaned Twitter data and label")
		if st.checkbox('Show cleaned data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'clean_message']]) # will write the df to the page            
		st.markdown("""![ChessUrl](https://i.gifer.com/HnFI.gif "Introduction")""")

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
