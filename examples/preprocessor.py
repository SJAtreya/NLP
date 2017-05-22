#In preprocessor.py
import spacy
import en_core_web_sm
from os import listdir
from os.path import isfile, join
import numpy as np

nlp = en_core_web_sm.load()
data_path = 'data'
labels = [f for f in listdir(data_path) if isfile(join(data_path, f))]
class Dataset(object):
    def __init__(self):
        X_all_sent = []
        X_all_vec_seq = []
        X_all_doc_vec = []
        Y_all = []
        for label in labels:
            x_file = open('data/'+label)
            x_sents = x_file.read().split('\n')
            for x_sent in x_sents:
                if len(x_sent) > 0:
                    x_doc = nlp(unicode(x_sent.decode('utf-8')))
                    x_doc_vec = x_doc.vector/x_doc.vector_norm
                    x_vec_seq = []
                    for word in x_doc:
                        x_vec_seq.append(word.vector/word.vector_norm)
                    x_vec_seq = np.array(x_vec_seq)
                    X_all_sent.append(x_sent)
                    X_all_doc_vec.append(x_doc_vec)
                    X_all_vec_seq.append(x_vec_seq)
                    Y_all.append(label.split(".")[0])
        self.X_all_sent = X_all_sent
        self.X_all_vec_seq = X_all_vec_seq
        self.X_all_doc_vec = X_all_doc_vec
        self.Y_all = Y_all
        self.num_classes=len(labels)

def pad_vec_sequences(sequences,maxlen=12):
    new_sequences = []
    for sequence in sequences:
        orig_len, vec_len = np.shape(sequence)
        if orig_len < maxlen:
            new = np.zeros((maxlen,vec_len))
            new[maxlen-orig_len:,:] = sequence
        else:
            templen = orig_len
            new = sequence[orig_len-maxlen:,:]
        new_sequences.append(new)
    return np.array(new_sequences)