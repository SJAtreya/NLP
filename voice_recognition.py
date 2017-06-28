#!/usr/bin/env python3

import speech_recognition as sr
import requests
import pyttsx
import sys

print("Loading TTS...")
engine = pyttsx.init()
engine.setProperty('rate', 165)
print("Starting voice recognizer...")
r = sr.Recognizer()
api = "/"
def conversationHandler(text):
    engine.say(text)
    engine.runAndWait()

def listenAndDecode(api):
    callback = "/"
    text = None
    with sr.Microphone() as source:
        audio = r.listen(source)
        print("Captured Audio... Waiting for transcript")
        try:
            text = r.recognize_google(audio)
            print("Did you say: " + text)
        except sr.UnknownValueError:
            text = None
            print("Sorry, I could not understand, say that again please?")
        except sr.RequestError as e:
            text = None
            print("Apparently I'm having trouble contacting my server.")
    if text is not None:
        try:
            print ("Api:", api)		
            response = requests.post('http://10.211.4.210:8888'+api+'?query='+text)
            print (response)
            parsedResponse = response.json()
            print (parsedResponse['status'])
            if parsedResponse['status'] == "Success":
			    print (parsedResponse['callback'])
			    callback = parsedResponse['callback']
            else:
                callback = "/"
            conversationHandler(parsedResponse['message'])
        except:
            print "Unexpected error:", sys.exc_info()[0]
            conversationHandler("Sorry, I'm unable to contact Nuke Trac at this time.")
    return callback

if __name__ == "__main__":
    api = "/"
    while(True):
        print("Listening for command...")
        api = listenAndDecode(api)
