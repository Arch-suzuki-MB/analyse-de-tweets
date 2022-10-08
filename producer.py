import tweepy
from kafka import KafkaProducer
import logging

"""API ACCESS KEYS"""

consumerKey ="vvl6tAUpTVYG2p9XrrY3pGdS3"
consumerSecret ="kv3K5vj9DyTiYB7ELZfgxANrZOZIxlZ29KwKDok4rBo2WHkCyI"
accessToken ="1226574939119136769-bd5aBC0QhzE7Qp2GOWULD8CGAyeisd"
accessTokenSecret ="8yt5u3d2JMPd8ZgBTy9MoA34I8GYnumQP3b9EZ7sExQtx"

producer = KafkaProducer(bootstrap_servers='localhost:9092')
search_term = 'musique'
topic_name = 'test'

def twitterAuth():
    # create the authentication object
    authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret)
    # set the access token and the access token secret
    authenticate.set_access_token(accessToken, accessTokenSecret)
    # create the API object
    api = tweepy.API(authenticate, wait_on_rate_limit=True)
    return api

class TweetListener(tweepy.Stream):

    def on_data(self, raw_data):
        logging.info(raw_data)
        producer.send(topic_name, value=raw_data)
        return True

    def on_error(self, status_code):
        if status_code == 420:
            # returning False in on_data disconnects the stream
            return False

    def start_streaming_tweets(self, search_term):
        self.filter(track=search_term, stall_warnings=True, languages=["en"])

if __name__ == '__main__':
    print("the producer produce tweets")
    twitter_stream = TweetListener(consumerKey, consumerSecret, accessToken, accessTokenSecret)
    twitter_stream.start_streaming_tweets(search_term)