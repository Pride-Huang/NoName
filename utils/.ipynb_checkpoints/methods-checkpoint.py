import torch
from sklearn import svm
from sklearn.metrics import accuracy_score, normalized_mutual_info_score

def linear_decision(features, labels):
    model_SVM = svm.SVC(kernel='poly', degree=3)
    model_SVM.fit(features, labels)
    return model_SVM
    
def cluster(features):
    kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(features)
    pred = kmeans.labels_
    return pred
    
def accuracy(labels, predictions):
    count = 0
    for l, p in zip(labels, predictions):
        l_class = torch.where(l==torch.max(l))[0]
        p_class = torch.where(p==torch.max(p))[0]
        if l_class[0] == p_class[0]:
            count = count+1
    acc = count / len(labels)
    return acc

def pred_label(labels, predictions):
    pred = []
    for l, p in zip(labels, predictions):
        p_class = torch.where(p==torch.max(p))[0]
        pred.append(p_class)
    return pred


