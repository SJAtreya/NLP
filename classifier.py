from __future__ import unicode_literals, print_function

import random
from pathlib import Path
import random
import requests
import spacy
import en_core_web_sm
from spacy.gold import GoldParse
from spacy.tagger import Tagger
import pyttsx

nlp = en_core_web_sm.load()
def train_ner(nlp, train_data, output_dir):
    # Add new words to vocab
    for raw_text, _ in train_data:
        doc = nlp.make_doc(raw_text)
        for word in doc:
            _ = nlp.vocab[word.orth]
    random.seed(42)
    # You may need to change the learning rate. It's generally difficult to
    # guess what rate you should set, especially when you have limited data.
    nlp.entity.model.learn_rate = 0.001
    for itn in range(200):
        random.shuffle(train_data)
        loss = 0.
        for raw_text, entity_offsets in train_data:
            gold = GoldParse(doc, entities=entity_offsets)
            doc = nlp.make_doc(raw_text)
            nlp.tagger(doc)
            loss += nlp.entity.update(doc, gold, 0.99)
        if loss == 0:
            break
    # This step averages the model's weights. This may or may not be good for
    # your situation --- it's empirical.
    if output_dir:
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.save_to_directory(output_dir)
	
def processResponse(parsed, response):
    message = ""
    status = "Failed"
    callback = ""
    if 'Intent' not in parsed:
        return {message: "Sorry, I could not understand. Are you requesting for scheduling a patient? If so, say, schedule procedure for patient", 'status': status, callback:callback}
    if parsed['Intent'] == 'schedule':
        if response.status_code == 201:
            if 'Who' in parsed:
			    message = "Procedure Scheduled successfully for "+ parsed['Who'] +". Would you like to send the order?"
            else:
                message = 'Procedure Scheduled successfully per physician order. Would you like to send it?'
            status = "Success"
            callback = "/order"			
        else:
            message = "There was a problem scheduling the procedure in Nuke Track. Please validate if the configuration is valid."
    elif parsed['Intent'] == 'performing':
        if response.status_code == 201:
            group = "you"
            if 'Who' in parsed:
                group = parsed['Who']
            message = 'Okay. I will record the readings for '+group+"."
            status = "Success"
            callback = "/task"
        else:
            message = "I'm unable to find an configuration for that task."
    else:
        message = "Oops, I don't support "+ parsed['Intent'] + " yet."
    return {'message': message, 'status': status, 'callback':callback}

def classify(text):
    print("Atom is trying to understand what you said: "+text)
    parsed = dict()
    if text is not None:
        test = [text]
    else:
        test = open('test.tsv', 'r')
    for row in test: 
        doc = nlp(unicode(row))
        for ent in doc.ents:
            if ent.label_ not in parsed:
                print(ent.label_,":", ent.text)
                parsed[ent.label_] = ent.text
    headers = {'Content-type': 'application/json'}
    response = requests.post('https://selva.cfapps.io/schedule',json=parsed, headers=headers)
    return processResponse(parsed, response)
		
def initializeSchedule():
    train_data = []
    survey = open('survey.txt', 'r')
    for line in survey:
        start = line.index('performing')
        forStart = line.index('for')
        at = line.index('at')
        train_data.append((unicode(line),[(start, start+10, u'Intent'), (start+11,forStart,u'What') ,(forStart+4, at,u'Who')]))
    f = open('simplified.tsv', 'r')
    for line in f: 
        start = line.index('schedule')
        forStart = line.index('for')
        at = line.index('at')
        train_data.append((unicode(line),[(start, start+8, u'Intent'), (start+9,forStart,u'What') ,(forStart+4, at,u'Who')]))
    nlp.entity.add_label(u'Intent')
    nlp.entity.add_label(u'What')
    nlp.entity.add_label(u'Who')
    train_ner(nlp, train_data, None)
    #print(classify("schedule an ultrasound"))
    #print(classify("I am performing area survey for group1"))
    #print(classify("I am performing function check for group1"))

'''def initializeHealthPhysics():
    train_data = []
    train_ner(nlp, train_data, None)'''
	
initializeSchedule()
print("Hey! I'm Atom!")

nlp.end_training()

if __name__ == '__main__':
    import plac
    plac.call(initializeSchedule)