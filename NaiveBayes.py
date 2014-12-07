from __future__ import division
import numpy as np
import scipy as sc
from prettyprint import pp
import os
import re
from datetime import datetime as dt


#index label in the dictionary
idx_lbl = 'idx'
dfreq_lbl = "docfreq"


pattern = re.compile(r'([a-zA-Z]+|[0-9]+(\.[0-9]+)?)')

def tokenizeDoc(doc_address, min_len = 0, remove_numerics=True):
    """
    to tokenize a document file to alphabetic tokens use this function.
    doc_address: path to the file that is going to be tokenized
    min_len: minimum length of a token. Default value is zero, it should always be non-negative.
    remove_numerics: whether to remove the numeric tokens or not
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
    classes: this is a list of the names of the classes documents belong to
    tokens_pool: a pool (in fact implemented as a dictionary) of tokens. Each value of the dictionary is an list of lists,
                 each list belonging to a document in the corresponding class that has a list of tokens

    output:
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
    classes: a list of the names of the classes documents belong to
    paths: a dictionary of lists of paths to documents

    output: a dictionary of lists of lists of tokens. each value bin of dictionary is a has a list of lists,
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
    tdict: a dictionary object containing many pairs of key and value
    filename: name of the dictionary file

    output: a csv file in which dictionary is dumped
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
    filename: name of the dictionary file
    *dictionary file is a csv file, each row contains a token and it's index

    output: a dictionary object created from input file
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
    this method will split the input list of files to train and test sets

    output: a tuple of train and test files after splitting

    *Note: currently this method uses the simplest way an array can be split in two parts
    """
    train_dict = {}
    test_dict = {}
    for cl in classes:
        train_cnt = int(ratio * len(files[cl]))
        train_dict[cl] = files[cl][:train_cnt]
        test_dict[cl] = files[cl][train_cnt+1:]
    return train_dict, test_dict

class NaiveBayes:
    """
    This is an implementation of Naive Bayes classifier.
    In the training phase, this classifier will learn the prior probability for each class
    and also the conditional probability of each feature for each class.
    After the training phase, it can easily predict the class label for a given input vector.
    By calculating the posterior probability of each class for the given vector, input vector's
    class label will be the label of the class having maximum posterior probability.

    lbl = argmax_{k} p(C_k | x) === argmax_{k} p(x | C_k) * p(C_k)
    """
    def __init__(self, class_labels, tdict):
        """
        constructor will get a list of the class labels, a dictionary of terms (as created before).
        Then, by calling the train function, probabilities will be learned from the training set.
        class_labels: a list of class labels
        tdict: a dictionary of terms, termIDs, and number of occurences of term in each class
        """
        self.k = len(class_labels)
        self.priors = np.zeros((self.k, 1))             #prior probabilities for each class
        self.cctermp = np.zeros((len(tdict), self.k))   #class conditional term probabilities
        self.ctermcnt = np.zeros((self.k, 1))           # total number of terms in a class
        self.lbl_dict = dict(zip(class_labels, range(self.k)))
        self.class_labels = class_labels
        self.tdict = tdict
    def train(self, class_counts, tfidf_but_smoothing = True):
        """
        this will learn the prior and class conditional probabilities for the set of documents
        for learning class prior probabilities, it will use the class_counts list
        for learning class conditional probabilities for each class, it will use the
        term dictionary provided
        class_counts: number of documents in each class
        tfidf_but_smoothing: if True, tfidf weighting will be used (ntn.ntn)
                             if False, smoothing will be used
        """
        # First learn the prior probabilities

        if len(class_counts) != len(self.priors):
            print "error! number of classes don't match"
            return
        for i in range(len(class_counts)):
            self.priors[i, 0] = class_counts[0] * 1.0 / sum(class_counts)

        # now learn the class conditional probabilities for each term
        for term, data in self.tdict.items():
            idx = data[idx_lbl]
            for cl in self.lbl_dict.viewkeys():
                if cl in data:
                    self.cctermp[idx, self.lbl_dict[cl]] += data[cl]
                    self.ctermcnt[self.lbl_dict[cl], 0] += data[cl]

        print self.cctermp

        if not tfidf_but_smoothing:
            for i in range(len(self.tdict)):
                for j in range(self.k):
                    self.cctermp[i,j] = (self.cctermp[i,j] + 1) * 1.0 / (self.ctermcnt[j] + len(self.tdict))
        else:
            for i in range(len(self.tdict)):
                for j in range(self.k):
                    if self.cctermp[i,j] > 0:
                        self.cctermp[i,j] = (self.cctermp[i,j] + 1) * np.log(self.ctermcnt[j] * 1.0 / self.cctermp[i,j])

    def predict(self, doc):
        """
        this method will predict the label for the input document using the Naive Bayes classification method

        doc: input document for which its label is going to be predicted, this argument should be provided as a list of tokens

        output: label of the document
        """

        doc_vec = self.__createVectorRepresentation(doc)

        class_score = [0] * self.k
        for i in range(self.k):
            log_class_conditional = np.log(self.cctermp[:,i] + 1e-14)
            class_score[i] = log_class_conditional.transpose().dot(doc_vec)[0] + np.log(self.priors[i,0])

        pp (class_score)
        return self.class_labels[class_score.index(max(class_score))]


    def __createVectorRepresentation(self, tokens_list):
        """
        this method will create a vector space representation of the list of tokens provided
        tdict: dictionary against which the vector space representation will be produced
        tokens_list: a list of tokens all of whom which may or may not belong to the dictionary provided

        output: a vector as a numpy array of size (len(tdict), 1) for which every row shows the number of
                times a token has appeared in a given document
        """
        vec = np.zeros((len(self.tdict), 1), dtype=np.int8)
        for token in tokens_list:
            if token in self.tdict:
                vec[self.tdict[token][idx_lbl], 0] += 1
        return vec


def main():

    root_path = 'E:/University Central/Modern Information Retrieval/Project/Project Phase 2/20_newsgroup/'
    #top_view folders
    folders = [root_path + folder + '/' for folder in os.listdir(root_path)]

    #there are only 4 classes
    class_titles = os.listdir(root_path)


    #list of all the files belonging to each class
    files = {}
    for folder, title in zip(folders, class_titles):
        files[title] = [folder + f for f in os.listdir(folder)]

    train_test_ratio = 0.75

    train, test = train_test_split(train_test_ratio, class_titles, files)

    pool = createTokenPool(class_titles, train)
    print len(pool[class_titles[0]])
    tdict = createDictionary(class_titles, pool)
    print len(tdict)

    dumbBayes = NaiveBayes(class_titles, tdict)
    class_count = [len(train[cl]) for cl in class_titles]

    start = dt.now()
    dumbBayes.train(class_count, False)
    end = dt.now()

    print 'elapsed time for training the Naive Bayes'
    print end - start

    id = 1
    start = dt.now()
    lbl = dumbBayes.predict(tokenizeDoc(test[class_titles[id]][3]))
    end = dt.now()

    print 'elapsed time for testing Naive Bayes'
    print end - start
    print lbl == class_titles[id]



    # saveDictToFile(tdict, 'dictionary.csv')
    #
    # redict = readFileToDict('dictionary.csv')
    # print len(redict)
    #
    # print redict == tdict



if __name__ == "__main__":
    main()
