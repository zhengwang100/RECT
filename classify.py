from __future__ import print_function
import os
from statistics import mean
import numpy
from sklearn.preprocessing import normalize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from time import time
from label_utils_functions import cover_all_class_split_train

class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
        return numpy.asarray(all_labels)

class Classifier(object):

    def __init__(self, vectors, clf):

        self.embeddings = vectors
        self.clf = TopKRanker(clf)
        self.binarizer = MultiLabelBinarizer(sparse_output=True)

    def normalize_vector(self, vectors):
        vectors_dict ={ }
        vectors_array=[]
        number =int(len(vectors.keys()))
        for i in range(number):
            vectors_array.append(vectors[str(i)])
        vectors_array = normalize(numpy.array(vectors_array), norm='l2', axis=1)
        for i, embedding in enumerate(vectors_array):
            vectors_dict[str(i)] = embedding
        return  vectors_dict

    def train(self, X, Y, Y_all):
        mydict = {}
        for y in Y:
            for label in y:
                mydict[int(label)] = 1 
        #print( sorted(mydict.keys()) )
        self.binarizer.fit(Y_all)
        X_train = [self.embeddings[x] for x in X]
        Y = self.binarizer.transform(Y)
        self.clf.fit(X_train, Y)

    def evaluate(self, X, Y):
        top_k_list = [len(l) for l in Y]
        Y_ = self.predict(X, top_k_list)
        Y = self.binarizer.transform(Y)
        averages = ["micro", "macro", "samples", "weighted"]
        results = {}
        for average in averages:
            results[average] = f1_score(Y, Y_, average=average)
        #print(results)
        return results
        
    def predict(self, X, top_k_list):
        X_ = numpy.asarray([self.embeddings[x] for x in X])
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        return Y

    def split_train_evaluate(self, X, Y, train_precent, seed=0):
        # not use any more
        state = numpy.random.get_state()

        training_size = int(train_precent * len(X))
        numpy.random.seed(seed)
        shuffle_indices = numpy.random.permutation(numpy.arange(len(X)))
        X_train = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
        X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]

        self.train(X_train, Y_train, Y)
        numpy.random.set_state(state)
        return self.evaluate(X_test, Y_test)

    def cover_all_class_split_train_evaluate(self, X, Y, train_precent):
        # We ensure every class has train/test instances 
        X_train, X_test, Y_train, Y_test = cover_all_class_split_train(X, Y, train_precent)
        self.train(X_train, Y_train, Y)
        return self.evaluate(X_test, Y_test)

    def evaluate_with_fixed_split(self, X_train, Y_train, X_test, Y_test, Y_all):
        '''Given the pre-defined train/test split, evaluate the embedding.'''
        self.train(X_train, Y_train, Y_all)
        return self.evaluate(X_test, Y_test)

def read_node_label(filename):
    #print(os.getcwd())
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split()
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y

def evaluate_embeddings_with_split(vectors, X_train, Y_train, X_test, Y_test, Y_all, testnum=10):
    print("Training an SVM classifier under the pre-defined split setting...")
    #clf = Classifier(vectors=vectors, clf=LogisticRegression())
    clf = Classifier(vectors=vectors, clf=SVC(probability=True))
    micro_list = []; macro_list = []
    for i in range(testnum):
        res = clf.evaluate_with_fixed_split(X_train, Y_train, X_test, Y_test, Y_all)

        micro_list.append(res['micro'])
        macro_list.append(res['macro'])
    return mean(micro_list), mean(macro_list)
