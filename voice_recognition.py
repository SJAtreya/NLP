#!/usr/bin/env python3

import speech_recognition as sr

def listenAndDecode(callback):
    # obtain audio from the microphone
    print("Listening for command...")
    r = sr.Recognizer()
    with sr.Microphone() as source:
        isQuestionToAtom = False
        while isQuestionToAtom == False:
            audio = r.listen(source)
        # recognize speech using Sphinx
        '''try:
            print("Sphinx thinks you said " + r.recognize_sphinx(audio))
        except sr.UnknownValueError:
            print("Sphinx could not understand audio")
        except sr.RequestError as e:
            print("Sphinx error; {0}".format(e))'''
        # recognize speech using Google Speech Recognition
        try:
            # for testing purposes, we're just using the default API key
            # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
            # instead of `r.recognize_google(audio)`
            text = r.recognize_google(audio)
            print("What I understood:",text)			
            isQuestionToAtom = text.startswith("Atom")
            print("Google Speech Recognition thinks you said " + text)
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
    callback(text)