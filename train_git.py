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

print("Loading TTS")
engine = pyttsx.init()
print("Loading initial model")
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
        print("Epoch #:", itn)
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
    print("Training Complete")
    if output_dir:
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.save_to_directory(output_dir)

def classify(text):
    # Test that the entity is recognized
    schedule = dict()
    if text is not None:
        test = [text]
    else:
        test = open('test.tsv', 'r')
    for row in test: 
        doc = nlp(unicode(row))
        print("Ents in :", row)
        for ent in doc.ents:
            schedule[ent.label_] = ent.text
    headers = {'Content-type': 'application/json'}
    r = requests.post('https://selva.cfapps.io/schedule',json=schedule, headers=headers)
    if r.status_code == 201:
        engine.say('Procedure Scheduled successfully.')
        engine.say('Would you like to send the order?')
    else:
        engine.say("The procedure does not exist in Nuke Track. Please configure the procedure before scheduling it.")
    engine.runAndWait()
    print(r)

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

if __name__ == '__main__':
    import plac
    plac.call(main)