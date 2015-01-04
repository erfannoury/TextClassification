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

class Rocchio:
    """
    This is an implementation of the Rocchio classifier.
    In the training phase, this classifier will learn centroids for each class.
    After the training phase, it can easily predict the class label for a given input vector.
    By calculating the input vector's distance to each centroid, input vector's
    class label will be the label of the class having minimum distance.

    *Note: each taining set vector should be normlized to unit length, however even normalizing
           input vectors doesn't indicate that centroid vectors will have unit length.
           Nonetheless, until input vector and each training set vector are normlized, we shouldn't
           have any problems.

    lbl = argmax_{k} |\mu_{k} - v(d)|
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
               dictionary of terms and number of occurences of term in each class
        """
        self.k = len(class_labels)
        # self.centroids = [np.zeros((len(tdict), 1))]*self.k # centroid vector for each class
        self.centroids = []
        self.lbl_dict = dict(zip(class_labels, range(self.k)))
        self.class_labels = class_labels
        self.tdict = tdict
        self.ctermcnt = np.zeros((self.k, 1))           # total number of terms in a class


    def train(self, token_pool, tfidf_but_smoothing = True):
        """
        this method will find the centroids for each class

        Parameters
        ----------
        token_pool: dictionary
                    pool of tokens for each document in each class. We could find the centroid for
                    each class using only the dictionary provided; but the normalization is the problem.
                    This way, each training set vector can be normalized to unit length.
        tfidf_but_smoothing: boolean
                             if True, tfidf weighting will be used (ntn.ntn)
                             if False, smoothing will be used
        """

        if len(token_pool) != len(self.class_labels):
            print "error! number of classes don't match"
            return

        # now find the term frequency for each class
        for term, data in self.tdict.items():
            for cl in self.lbl_dict:
                if cl in data:
                    self.ctermcnt[self.lbl_dict[cl], 0] += data[cl]

        # now normalize each input vector and add it to its corresponding centroid vector
        for cl in self.class_labels:
            self.centroids.append(np.zeros((len(self.tdict), 1)))
            for doc in token_pool[cl]:
                vec = self.__createNormalizedVectorRepresentation(doc, cl)
                self.centroids[self.lbl_dict[cl]] += vec

            self.centroids[self.lbl_dict[cl]] /= len(token_pool[cl])


    def predict(self, doc):
        """
        this method will predict the label for the input document using the Rochhio's classification method

        Parameters
        ----------
        doc: list
             input document for which its label is going to be predicted, this argument should be provided as an array of tokens

        Returns
        -------
        output: an element of the class_labels list
        """

        doc_vec = self.__createNormalizedVectorRepresentation(doc, None)

        distances = []
        for i in range(self.k):
            distances.append(np.linalg.norm(doc_vec - self.centroids[i]))


        # pp (distances)

        return self.class_labels[distances.index(min(distances))]


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


    def __createNormalizedVectorRepresentation(self, tokens_list, cl = None, tfidf = True):
        """
        this method will create a vector space representation of the list of tokens provided with unit length

        Parameters
        ----------
        tokens_list: list
                     list of tokens all of whom which may or may not belong to the dictionary provided
        cl: object or `None`
            the input vector's class, in case it is `None`, term frequency will be calculated from the document vector itself

        Returns
        -------
        vec: np.ndarray
             tfidf vector of size len(tdict)*1 that is normalized to have a unit length
        """
        vec = np.zeros((len(self.tdict), 1))
        for token in tokens_list:
            if token in self.tdict:
                vec[self.tdict[token][idx_lbl], 0] += 1

        token_set = set(tokens_list)
        if tfidf:
            if cl != None:
                for term in token_set:
                    if cl in self.tdict[term]:
                        vec[self.tdict[term][idx_lbl], 0] *= np.log(self.ctermcnt[self.lbl_dict[cl], 0] * 1.0 / self.tdict[term][cl])


        norm_vec = np.linalg.norm(vec)
        vec = (vec / (norm_vec + 1e-14))
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

    print "Rocchio's turn"

    rocchio = Rocchio(class_titles, tdict)

    start = dt.now()
    rocchio.train(pool)
    end = dt.now()

    print 'elapsed time for training rocchio'
    print end - start

    for c in rocchio.centroids:
        print np.linalg.norm(c - rocchio.centroids[0])
    id = 3

    start = dt.now()
    lbl = rocchio.predict(tokenizeDoc(test[class_titles[id]][3]))
    end = dt.now()

    print 'elapsed time for testing rocchio'
    print end - start

    print lbl == class_titles[id]


    test_pool = createTokenPool(class_titles, test)
    start = dt.now()
    test_lbl_pool = rocchio.predictPool(test_pool)
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
        print 'precision = ', P
        print 'recall = ', R
        print 'accuracy = ', Acc
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
