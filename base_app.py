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
from nltk.stem.porter import *

import warnings 
warnings.filterwarnings("ignore")

#Importing other libraries
from collections import defaultdict
from collections import  Counter

from nltk.corpus import stopwords
additional  = ['retweet']
stop = set().union(stopwords.words('english'),additional)

# Style 
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 

# Standard libraries
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

#DATA ClEANING
def TweetCleaner(tweet):
    
    #This function uses regular expressions to remove url's, mentions, hashtags, 
    #punctuation, numbers and any extra white space from tweets after converting everything to lowercase letters.
    
    # Convert everything to lowercase
    tweet = tweet.lower() 
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
    #Remove stop words
    remove_stopwords = [w for w in tweet.split() if w not in stop]
    tweet = ' '.join(remove_stopwords)
    
    return tweet

# Clean the tweets in the message column
raw['clean_message'] = raw['message'].apply(TweetCleaner)


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
	"""TEAM AM5 Tweet Classifier App Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Introduction", "Problem Statement", "Information", "EDA", "Model Description", "Prediction"]
	selection = st.sidebar.selectbox("Choose Option", options)
	# Building out the "Introduction" page
	if selection == "Introduction":
		st.info("Introduction")
		st.markdown("Given the recent explosion of Big Data, there is a growing demand for analyzing non traditional data sources. Social Media data is a huge source of this data in the form of chats, messages, news feeds and all of it is in an unstructured form. Text analytics is a process that helps analyze this unstructured data and look for patterns or infer popular sentiment which can help organizations in their decision making.Twitter data is a powerful source of information on a wide list of topics. This data can be analyzed to find trends related to specific topics, measure popular sentiment, obtain feedback on past decisions and also help make future decisions. Climate change has received extensive attention from public opinion in the last couple of years, after being considered for decades as an exclusive scientific debate")
		st.markdown("""![ChessUrl](https://i.gifer.com/HnFI.gif "chess")""")

	# Building out the "Problem Statement" page
	if selection == "Problem Statement":
		st.info("Problem Statement")
		st.markdown("Our task is to create a machine learning model that is able to clarify whether or not a person believes in climate change, based on their novel tweet data. To increase Thrive Market’s advertising efficiency by using machine learning to create effective marketing tools that can identify whether or not a person believes in climate change and could possibly be converted to a new customer based on their tweets.")
		st.markdown("""![ChessUrl](https://i.gifer.com/Q8k0.gif "chess")""")
        
	# Building out the "EDA" page
	if selection == "EDA":
		st.info("Exploratory Data Analysis")
		st.subheader("Percentage of Sentimentals (2,1,0 & -1) using Pie chart and Bar graph")
		st.pyplot(fig1)
		st.markdown("Target distribution percentages info:\n\n -54 percentage of our tweets shows that they believe in man-made climate change(Pro = 1) \n\n -23 percentage of our tweets shows the news about man-made climate change (News = 2) \n\n -15 percentage of our tweets shows that they neither believe or do not believe in man-made climate change (Neutral = 0) \n\n-8 percentage of our tweets show that they do not believe in man-made climate change(Anti = -1) \n\n -Imbalanced: The sentiment are not balanced. Which means when we predict our sentiment it will be leaning towards the majority sentiment")
		st.subheader("\n\n WORD DISTRIBUTION")
		st.pyplot(fig)
		st.markdown("Observations:\n\n -The distribution of words in tweets seems to be almost the similar. \n\n -The distribution of all sentiment are fairly symmentrical. \n\n -There are no words which are more than 40 characters")
		st.subheader("\n\n CHARACTER DISTRIBUTION")
		st.pyplot(fig2)
		st.markdown("Observations: \n\n -The distribution of characters  seems to be almost the similar. \n\n -130 to 140 character length in a tweet are the most common among all sentiments. \n\n -The distribution of all sentiment are skewed to the left.")
		st.subheader("\n\n WORD CLOUD OF ALL OUR TWEETS")
		st.pyplot(fig3)
		st.markdown("Observations:\n\n -Neutral Tweets: The words like climate change, global warming appear to be big which means that they are mosst coomon words in our neutral tweets. \n\n -Anti Tweets: Donald Trump appears a lot here. He is probably one of the people who do not believe in global warming. Also words like blame and scam are probably unique in here. \n\n -News Tweets: Words like fight climate appear a lot. \n\n -Pro Tweets: Words like tackle climate, change are probably unique in here too. \n\n -All Tweets: As expected, words like climate change and global warming are more visible/common.")
		
        
	# Building out the "Model Description" page
	if selection == "Model Description":
		st.info("Description")
		st.subheader("Logistic Regression")
		st.markdown("Logistic Regression uses the probability of a data point to belonging to a certain class to classify each datapoint to it's best estimated class. The figure below is the sigmoid function logistic regression models use to make predictions: ![1*a5QwiyaSyvRa6n3VKYVEnQ.png](https://cdn-media-1.freecodecamp.org/images/1*a5QwiyaSyvRa6n3VKYVEnQ.png)")
		st.subheader("Stochastic Gradient Descent - SGD Classifier")
		st.markdown("Stochastic Gradient Descent (SGD) is a simple yet efficient optimization algorithm used to find the values of parameters/coefficients of functions that minimize a cost function. In other words, it is used for discriminative learning of linear classifiers under convex loss functions such as SVM and Logistic regression. It has been successfully applied to large-scale datasets because the update to the coefficients is performed for each training instance, rather than at the end of instances. Stochastic Gradient Descent (SGD) classifier basically implements a plain SGD learning routine supporting various loss functions and penalties for classification. Logistic regression has been rated as the best performing model for linearly separable data especially if it's predicting binary data(Yes & NO or 1 & 0), and performs better when there's no class imbalance.")
		st.subheader("Support Vector Machine")
		st.markdown("In the SVM algorithm, we plot each data item as a point in n-dimensional space (where n is number of features you have) with the value of each feature being the value of a particular coordinate. The goal of the SVM algorithm is to create the best line or decision boundary that can seperate n-dimensional space into classes so that we can easily put the new data point in the correct category in the future. This best decision boundary is called a hyperplane.SVM chooses the extreme points/vectors that help in creating the hyperplane. These extreme cases are called as support vectors. Consider the below diagram in which there are two different categories that are classified using a decision boundary or hyperplane. SVM maps training examples to points in space so as to maximise the width of the gap between the two categories. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall. In addition to performing linear classification, SVMs can efficiently perform a non-linear classification using what is called the kernel trick, implicitly mapping their inputs into high-dimensional feature spaces. ![support-vector-machine-algorithm.png](https://static.javatpoint.com/tutorial/machine-learning/images/support-vector-machine-algorithm.png)")
	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

		st.subheader("Cleaned Twitter data and label")
		if st.checkbox('Show cleaned data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'clean_message']]) # will write the df to the page            

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

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
