

import tweepy
import json

# completar con las propias credenciales
CONSUMER_TOKEN = "..."
CONSUMER_SECRET = "..."
ACCESS_TOKEN = "..."
ACCESS_TOKEN_SECRET = "..."


#override tweepy.StreamListener to add logic to on_status
class MyStreamListener(tweepy.StreamListener):

    def __init__(self, api,fichero):
        super().__init__(api)
        self.fichero = fichero

    def on_status(self, status):
        with open(self.fichero, 'a+') as f:
                f.write(json.dumps(status._json))
                f.write("\n")     
        
    def on_error(self, status_code):
        print(status_code)
        
        
###################################################

folder = 'C:/datos/' # no olvidar / al final!!!
fichero = 'tweets.txt'
terms = ['#FelizDomingo']

auth = tweepy.OAuthHandler(CONSUMER_TOKEN, CONSUMER_SECRET) 
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

myStreamListener = MyStreamListener(api,folder+fichero)
myStream = tweepy.Stream(auth = api.auth,listener=myStreamListener)
myStream.filter(track=terms, stall_warnings=True)
