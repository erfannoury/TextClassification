import numpy as np
import scipy as sc
from prettyprint import pp
import os
import re

#index label in the dictionary
idx_lbl = 'idx'
dfreq_lbl = "docfreq"


pattern = re.compile(r'([a-zA-Z]+|[0-9]+(\.[0-9]+)?)')

def tokenizeDoc(doc_address, min_len = 0, remove_numerics=True):
    """
    to tokenize a document file to alphabetic tokens use this function.

    Parameters
    ----------
    doc_address: str
                 path to the file that is going to be tokenized
    min_len: int
             minimum length of a token. Default value is zero, it should always be non-negative.
    remove_numerics: boolean
                     whether to remove the numeric tokens or not

    Returns
    -------
    tokens: list
            list of tokens from the input document according to the filtering criteria specified
    """
    from string import punctuation, digits
    tokens = []
    try:
        f = open(doc_address)
        raw = f.read().lower()
        text = pattern.sub(r' \1 ', raw.replace('\n', ' '))
        text_translated = ''
        if remove_numerics:
            text_translated = text.translate(None, punctuation + digits)
        else:
            text_translated = text.translate(None, punctuation)
        tokens = [word for word in text_translated.split(' ') if (word and len(word) > min_len)]
        f.close()
    except:
        print "Error: %s couldn't be opened!", doc_address
    finally:
        return tokens



def createDictionary(classes, tokens_pool):
    """
    this method will create a dictionary out of the tokens_pool it has been provided.

    Parameters
    ----------
    classes: list
             list of the names of the classes of documents
    tokens_pool: dictionary
                 dictionary of tokens. Each value of the dictionary is an list of lists,
                 each list belonging to a document in the corresponding class that has a list of tokens


    Returns
    -------
    token_dict: dictionary
                *Note that the tokens in the dictionary are not sorted, since in the vector space model
                that we are going to use, all words are treated equal.
                We practically believe in justice. Words in dictionary are tired of
                all this injustice they have been forced to take for such a long time.
                Now is the time to rise and earn the justice that belongs to them.
    """

    token_dict = {}
    idx = 0 #a unique index for words in dictionary
    for cl in classes:
        for tokens_list in tokens_pool[cl]:
            for token in tokens_list:
                if token in token_dict:             #if token has been added to the dictionary before
                    if cl in token_dict[token]:
                        token_dict[token][cl] += 1
                    else:
                        token_dict[token][cl] = 1
                else:
                    token_dict[token] = {}
                    token_dict[token][idx_lbl] = idx
                    idx += 1
                    token_dict[token][cl] = 1
    return token_dict



def createTokenPool(classes, paths):
    """
    this method will create a pool of tokens out of the list of paths to documents it will be provided

    Parameters
    ----------
    classes: list
             list of the names of the classes documents belong to
    paths: dictionary
           dictionary of lists of paths to documents

    Returns
    -------
    token_pool: dictionary
                dictionary of lists of lists of tokens. each value bin of dictionary is a has a list of lists,
                for which each list is of a document and it contains a list of tokens in that document
    """
    token_pool = {}
    for cl in classes:
        token_pool[cl] = []
        for path in paths[cl]:
            token_pool[cl].append(tokenizeDoc(path))

    return token_pool



def saveDictToFile(tdict, filename):
    """
    this method will save the key/value pair of the dictionary to a csv file

    Parameters
    ----------
    tdict: dictionary
           dictionary object containing many pairs of key and value
    filename: str
              name of the dictionary file


    Returns
    -------
    """
    import csv
    w = csv.writer(open(filename, "w"))
    for key, val in tdict.items():
        row = []
        row.append(key)
        row.append(val[idx_lbl])
        for cl in class_titles:
            if cl in val:
                row.append(cl + ':' + str(val[cl]))
        w.writerow(row)



def readFileToDict(filename):
    """
    this method will create a dictionary from a file
    *dictionary file is a csv file, each row contains a token and it's index
    Parameters
    ----------
    filename: str
              name of the dictionary file

    Returns
    -------
    tdict: dictionary
           dictionary object created from input file
    """
    import csv, codecs
    tdict = {}
    for row in csv.reader(codecs.open(filename, 'r')):
        try:
            tdict[row[0]] = {}
            tdict[row[0]][idx_lbl] = int(row[1])
            for i in range(2, len(row)):
                lbl, cnt = row[i].split(':')
                tdict[row[0]][lbl] = int(cnt)
        except:
            continue
    return tdict



def train_test_split(ratio, classes, files):
    """
    this method will split the input list of files to train and test sets.
    *Note: currently this method uses the simplest way an array can be split in two parts.

    Parameters
    ----------
    ratio: float
           ratio of total documents in each class assigned to the training set
    classes: list
             list of label classes
    files: dictionary
           a dictionary with list of files for each class

    Returns
    -------
    train_dic: dictionary
               a dictionary with lists of documents in the training set for each class
    test_dict: dictionary
               a dictionary with lists of documents in the testing set for each class
    """
    train_dict = {}
    test_dict = {}
    for cl in classes:
        train_cnt = int(ratio * len(files[cl]))
        train_dict[cl] = files[cl][:train_cnt]
        test_dict[cl] = files[cl][train_cnt:]
    return train_dict, test_dict
