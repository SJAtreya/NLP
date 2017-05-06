from __future__ import unicode_literals, print_function

import random
from pathlib import Path
import random

import spacy
import en_core_web_sm
from spacy.gold import GoldParse
from spacy.tagger import Tagger


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
    for itn in range(100):
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


def main(model_name="en", output_directory=None):
    print("Loading initial model", model_name)
    nlp = en_core_web_sm.load()
    if output_directory is not None:
        output_directory = Path(output_directory)
    train_data = []
    f = open('simplified.tsv', 'r')
    for line in f: 
        start = line.index('schedule')
        forStart = line.index('for')
        at = line.index('at')
        print("ProcName:",line[start+9:forStart])
        print("PatientName:",line[forStart+4:at])
        train_data.append((unicode(line),[(start, start+8, u'Schedule'), (start+9,forStart,u'ProcName') ,(forStart+4, at,u'PatientName')]))
    nlp.entity.add_label(u'Schedule')
    nlp.entity.add_label(u'ProcName')
    nlp.entity.add_label(u'PatientName')
    train_ner(nlp, train_data, output_directory)

    # Test that the entity is recognized
    # doc = nlp('please schedule a REST Stress for Sudharshan Atreya?')
    test = open('test.tsv', 'r')
    for row in test: 
        doc = nlp(unicode(row))
        print("Ents in :", row)
        for ent in doc.ents:
            print(ent.label_, ent.text)
if __name__ == '__main__':
    import plac
    plac.call(main)