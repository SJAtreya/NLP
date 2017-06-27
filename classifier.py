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
    random.seed(0)
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
    nlp.end_training()
    if output_dir:
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.save_to_directory(output_dir)
	
def processResponse(parsed, response):
    message = ""
    status = "Failed"
    callback = ""
    if 'Schedule' in parsed:
        if response.status_code == 201:
            if 'procedure' in parsed:
			    message = parsed['procedure'] + ' Scheduled successfully. Would you like to send the order?'
            else:
                message = 'Procedure Scheduled successfully. Would you like to send the order?'
            status = "Success"
            callback = "/order"			
        else:
            message = "There was a problem scheduling the procedure in Nuke Track. Please validate if the configuration is valid."
    elif 'Order' in parsed:
        if response.status_code == 201:
            message = 'Order processed successfully.'
            status = "Success"
            callback = "/status"
        else:
            message = "There was a problem processind the order in Nuke Track."
    else:
        message = "Sorry, I could not understand. Are you requesting for scheduling a patient?"
    return {'message': message, 'status': status, 'callback':callback}

def classify(text):
    # Test that the entity is recognized
    parsed = dict()
    if text is not None:
        test = [text]
    else:
        test = open('test.tsv', 'r')
    for row in test: 
        doc = nlp(unicode(row))
        for ent in doc.ents:
            print(ent.label_,":", ent.text)
            parsed[ent.label_] = ent.text
    headers = {'Content-type': 'application/json'}
    response = requests.post('https://selva.cfapps.io/schedule',json=parsed, headers=headers)
    return processResponse(parsed, response)
		
def initialize():
    train_data = []
    f = open('simplified.tsv', 'r')
    for line in f: 
        start = line.index('schedule')
        forStart = line.index('for')
        at = line.index('at')
        train_data.append((unicode(line),[(start, start+8, u'Schedule'), (start+9,forStart,u'procedure') ,(forStart+4, at,u'patient')]))
    nlp.entity.add_label(u'Schedule')
    nlp.entity.add_label(u'procedure')
    nlp.entity.add_label(u'patient')
    train_ner(nlp, train_data, None)
    print("Hey there!")

initialize()

if __name__ == '__main__':
    import plac
    plac.call(initialize())