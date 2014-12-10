from __future__ import division
import numpy as np
import scipy as sc
from prettyprint import pp
import os
import re
from datetime import datetime as dt
from util import *

#index label in the dictionary
idx_lbl = 'idx'
dfreq_lbl = "docfreq"

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

        Parameters
        ----------
        class_labels: list
                      list of class labels
        tdict: dictionary
               a dictionary of terms and number of occurences of term in each class
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

        Parameters
        ----------
        class_counts: list
                      number of documents in each class
        tfidf_but_smoothing: boolean
                             if True, tfidf weighting will be used (ntn.ntn)
                             if False, smoothing will be used

        Returns
        -------

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

        # print self.cctermp

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

        Parameters
        ----------
        doc: list
             input document for which its label is going to be predicted, this argument should be provided as a list of tokens

        Returns
        -------
        output: object
                an element of the class_labels list
        """

        doc_vec = self.__createVectorRepresentation(doc)

        class_score = [0] * self.k
        for i in range(self.k):
            log_class_conditional = np.log(self.cctermp[:,i] + 1e-14)
            class_score[i] = log_class_conditional.transpose().dot(doc_vec)[0] + np.log(self.priors[i,0])


        return self.class_labels[class_score.index(max(class_score))]


    def predictPool(self, doc_collection):
        """
        this method will get a dictionary of collection of documents and predict their label.

        Parameters
        ----------
        doc_collection: dictionary
                        dictionary of collection of documents for which we want to predict their label

        Returns
        -------
        lbl_pool: dictionary
                  dictionary of collection of labels for each corresponding document will be returned
        """
        lbl_pool = {}
        for cl in self.class_labels:
            lbl_pool[cl] = []
            for doc in doc_collection[cl]:
                lbl_pool[cl].append(self.predict(doc))

        return lbl_pool

    def __createVectorRepresentation(self, tokens_list):
        """
        this method will create a vector space representation of the list of tokens provided

        Parameters
        ----------
        tokens_list: list
                     list of tokens all of whom which may or may not belong to the dictionary provided

        Returns
        -------
        vec: np.ndarray
             vector as a numpy array of size (len(tdict), 1) for which every row shows the number of
             times a token has appeared in a given document
        """
        vec = np.zeros((len(self.tdict), 1), dtype=np.int8)
        for token in tokens_list:
            if token in self.tdict:
                vec[self.tdict[token][idx_lbl], 0] += 1
        return vec


def calculateMetrics(class_labels, lbl_pool):
    """
    this method will calculate the tp, tn, fp, fn metrics for each class
    of documents from the pool labels provided
        tp: number of documents in the class that are correctly labeled as belonging to class
        tn: number of documents not in the class that are correctly labeled as not belonging to class
        fp: number of documents not in the class that are incorrectly labeled as belonging to class
        fn: number of documents in the class that are incorrectly labeled as not belonging to class

    Parameters
    ----------
    class_labels: dictionary
                  labels of the classes
    lbl_pool: dictionary
              dictionary of lists of labels

    Returns
    -------
    metrics: dictionary
             dictionary of dictionaries of metrics for each class
    """
    metrics = {}
    for cl in class_labels:
        metrics[cl] = {}
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for lbl in lbl_pool[cl]:
            if lbl == cl:
                tp += 1
            else:
                fp += 1
        for ncl in class_labels:
            if ncl != cl:
                for lbl in lbl_pool[ncl]:
                    if lbl == cl:
                        fn += 1
                    else:
                        tn += 1

        metrics[cl]["tp"] = tp
        metrics[cl]["tn"] = tn
        metrics[cl]["fp"] = fp
        metrics[cl]["fn"] = fn

    return metrics

def main():

    root_path = '20_newsgroup/'
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

    test_pool = createTokenPool(class_titles, test)
    start = dt.now()
    test_lbl_pool = dumbBayes.predictPool(test_pool)
    end = dt.now()

    print 'elapsed time for testing a pool of documents'
    print end - start

    metrics = calculateMetrics(class_titles, test_lbl_pool)
    total_F = 0
    for cl in class_titles:
        print cl
        P = (metrics[cl]["tp"] * 1.0 / (metrics[cl]["tp"] + metrics[cl]["fp"]))
        R = (metrics[cl]["tp"] * 1.0 / (metrics[cl]["tp"] + metrics[cl]["fn"]))
        Acc = ((metrics[cl]["tp"] + metrics[cl]["tn"])* 1.0 / (metrics[cl]["tp"] + metrics[cl]["fp"] + metrics[cl]["fn"] + metrics[cl]["tn"]))
        F_1 = 2 * R * P / (R + P)
        total_F += F_1
        print 'P = ', P
        print 'R = ', R
        print ' '

    print 'macro-averaged F measure', (total_F / len(class_titles))


    # saveDictToFile(tdict, 'dictionary.csv')
    #
    # redict = readFileToDict('dictionary.csv')
    # print len(redict)
    #
    # print redict == tdict



if __name__ == "__main__":
    main()
