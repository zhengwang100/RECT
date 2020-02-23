import numpy as np
from sklearn.preprocessing import LabelBinarizer
import sys
sys.path.append("..")
from collections import defaultdict
# support multi class

def get_class_set(labels):
    # labels [l, [c1, c2, ..]]
    # return：the labeled class set
    mydict = {}
    for y in labels:
        for label in y:
            mydict[int(label)] = 1
    return mydict.keys()

def check_cover_all_class(Y_train, Y_all):
    # check if the train data cover all classes
    train_classes = get_class_set(Y_train)
    all_classes = get_class_set(Y_all)

    if(len(train_classes) == len(all_classes)): return True
    else: return False

def cover_all_class_split_train(X, Y, train_precent):
    """ We ensure every class has train/test instances
        Input: 
        Output: X_train [l, 1], X_test [n-l, 1]: the index node id list
                Y_train [ l, [c1, c2, ..]] , Y_test [n-l, [c1, c2, ..]]: the label of the cropossending node
    """
    training_size = int(train_precent * len(X))
    flag = False
    while flag is False:
        print('---- split once ------')
        shuffle_indices = np.random.permutation(np.arange(len(X)))
        X_train_idx = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
        X_test_idx = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]
        flag = check_cover_all_class(Y_train, Y)
    #print(type(X_train), type(Y_train))
    return X_train_idx, X_test_idx, Y_train, Y_test

def completely_imbalanced_split_train_single_label(X, Y, train_precent, removed_class):
    ''' 1) first split train/test part, 2) remove the labeled nodes in the removed_class
        Return: 6 list: X_train_idx [l], X_test_idx [n-l], Y_train [l, [c1,c2,...]], Y_test [n-l, [c1,c2,...]] are the common train\test split
                        X_train_cid_idx [l'], Y_train_cid [l', [c1,c2,...]] are the completely-imbalaced training data
    '''
    X_train_idx, X_test_idx, Y_train, Y_test = cover_all_class_split_train(X, Y, train_precent)
    idxlist = []
    for i in range(len(X_train_idx)):
        #print(set(Y_train[i]), set(removed_class), set(Y_train[i]) & set(removed_class))
        if( len(set(Y_train[i])&set(removed_class)) == 0 ):
            idxlist.append(i)
            
    X_train_cid_idx = [X_train_idx[i] for i in idxlist]
    Y_train_cid = [Y_train[i] for i in idxlist]
    return X_train_idx, X_test_idx, Y_train, Y_test, X_train_cid_idx, Y_train_cid

def completely_imbalanced_split_train_multi_label(X, Y, train_precent, removed_class):
    ''' 1) first split train/test part, 2) remove the labeled nodes in the removed_class
        Return: 6 list: X_train_idx [l], X_test_idx [n-l], Y_train [l, [c1,c2,...]], Y_test [n-l, [c1,c2,...]] are the common train\test split
                        X_train_cid_idx [l'], Y_train_cid [l', [c1,c2,...]] are the completely-imbalaced training data
    '''
    X_train_idx, X_test_idx, Y_train, Y_test = cover_all_class_split_train(X, Y, train_precent)
    idxlist = []
    for i in range(len(X_train_idx)):
        all_cover_flag = len(set(Y_train[i])&set(removed_class)) == len(set(Y_train[i])) # if the labels in a sample are all removed, cannot use
        if( all_cover_flag is False ):
            idxlist.append(i)
            
    X_train_cid_idx = [X_train_idx[i] for i in idxlist]
    Y_train_cid_temp = [Y_train[i] for i in idxlist]
    Y_train_cid = []
    for labels in Y_train_cid_temp:
        temp = [x for x in labels if x not in set(removed_class)]
        Y_train_cid.append(temp)

    return X_train_idx, X_test_idx, Y_train, Y_test, X_train_cid_idx, Y_train_cid

def completely_imbalanced_split_train(X, Y, train_precent, removed_class, single_label=True):
    if( single_label ): return completely_imbalanced_split_train_single_label(X, Y, train_precent, removed_class)
    else: return completely_imbalanced_split_train_multi_label(X, Y, train_precent, removed_class)

def get_label_attributes(nodeids, labellist, features):
    '''Suppose a node i is labeled as c, then attribute[c] += node_i_attribute, finally mean(attribute[c])
       Input: nodeids, labellist, features
       Output: label_attribute{}: label -> average_labeled_node_features
    '''
    _, feat_num = features.shape 
    labels = get_class_set(labellist)
    label_attribute_nodes = defaultdict(list)
    for nodeid, labels in zip(nodeids, labellist):
        for label in labels:
            label_attribute_nodes[int(label)].append(int(nodeid))
    label_attribute = {}
    for label in label_attribute_nodes.keys():
        nodes = label_attribute_nodes[int(label)]
        selected_features = features[nodes, :]
        label_attribute[int(label)] = np.mean(selected_features, axis=0)
    return label_attribute

def get_labeled_nodes_label_attribute(nodeids, labellist, features):
    '''# Return np[l, ft]
    '''
    label_attribute = get_label_attributes(nodeids, labellist, features)
    #print('label_attribute', label_attribute)
    res = np.zeros([len(nodeids), features.shape[1]])
    for i, labels in enumerate(labellist):
        c = len(labels)
        #print(nodeids[i], 'label number is', c, labels)
        temp = np.zeros([c, features.shape[1]])
        for ii, label in enumerate(labels):
            temp[ii, :] = label_attribute[int(label)]
        temp = np.mean(temp, axis=0)
        res[i, :] = temp
    #print('get_labeled_nodes_label_attribute', res.shape)
    return res