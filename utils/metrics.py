from sklearn.metrics import log_loss
import torch
import numpy as np
import pennylane as qml
import random
import torch.nn.functional as F

def information_image(input_data):
    quantumInfo = []
    for i in range(X_train_FRQI.shape[0]):
        quantumInfo.append(qml.math.vn_entropy(X_train_FRQI[i,:], indices=[i for i in range(8)]))
    quantumInfo_list = ["{:.2e}".format(tensor.item()) for tensor in quantumInfo]
    
    return quantumInfo_list

def image_entropy(image):
    hist, _ = np.histogram(image.flatten(), bins=256, range=[0,1])
    probabilities = hist / np.sum(hist)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    return entropy

def decision_entropy(prediction):
    prediction = prediction.cpu().detach().numpy()
    entropy = -np.sum(prediction * np.log2(prediction + 1e-10))
    return torch.tensor(entropy)

def split_fsl_query(X_train_FRQI, Y_train, images_train, labels_train, indices):
    images_fsl, labels_fsl,  state_fsl, y_fsl = images_train[indices], labels_train[indices], X_train_FRQI[indices], Y_train[indices]
    not_indices = [i for i in range(images_train.shape[0]) if i not in indices]
    images_query, labels_query, state_query, y_query = images_train[not_indices], labels_train[not_indices], X_train_FRQI[not_indices], Y_train[not_indices]
    return images_fsl, labels_fsl, state_fsl, y_fsl, images_query, labels_query, state_query, y_query

def extract_max_indices(entropy_list, sublist_size=50, max_elements=3):
    indices = []

    for i in range(0, len(entropy_list), sublist_size):
        sublist = entropy_list[i:i + sublist_size]
        sorted_indices = sorted(range(len(sublist)), key=lambda k: sublist[k], reverse=True)
        selected_indices = sorted_indices[:max_elements]

        for idx in selected_indices:
            indices.append(i + idx)
            
    return indices

def info_measure(data, info_type):
    entropy_list = []
    
    if info_type=='classical':
        for i in range(data.shape[0]):
            entropy_list.append(image_entropy(data[i,:]))
            
    elif info_type=='quantum':
        entropy_list_1 = []
        for i in range(data.shape[0]):
            entropy_list_1.append(qml.math.vn_entropy(data[i,:], indices=[i for i in range(8)]))
        entropy_list = ["{:.2e}".format(tensor.item()) for tensor in entropy_list_1]
            
    return entropy_list

def random_sample(state_fsl, y_fsl, state_query, y_query, AdjMtx, args): 
    N = state_query.shape[0]
    random_indices = torch.randperm(N)

    selected_indices = random_indices[:args.N_ACTIVE]
    
    state_fsl = torch.cat([state_fsl, state_query[selected_indices]])
    y_fsl = torch.cat([y_fsl, y_query[selected_indices]])
    
    indices_all = [i for i in range(state_query.shape[0])]
    result_list = [x for x in indices_all if x not in selected_indices]
    
    AdjMtx = AdjMtx[:,result_list]
    AdjMtx = AdjMtx[result_list, :]
    
    state_query = torch.index_select(state_query, 0, torch.tensor(result_list))
    y_query = torch.index_select(y_query, 0, torch.tensor(result_list))

    return state_fsl, y_fsl, state_query, y_query, AdjMtx

def active_sample(state_fsl, y_fsl, state_query, y_query, entropy, AdjMtx, args):
    sorted_indices = torch.argsort(entropy, descending=True)

    selected_indices = sorted_indices[:args.N_ACTIVE]
    state_fsl = torch.cat([state_fsl, state_query[selected_indices]])
    y_fsl = torch.cat([y_fsl, y_query[selected_indices]])
    indices_all = [i for i in range(state_query.shape[0])]
    result_list = [x for x in indices_all if x not in selected_indices]
    AdjMtx = AdjMtx[:,result_list]
    AdjMtx = AdjMtx[result_list, :]
    state_query = torch.index_select(state_query, 0, torch.tensor(result_list))
    y_query = torch.index_select(y_query, 0, torch.tensor(result_list))
    entropy = torch.index_select(entropy, 0, torch.tensor(result_list))
    return state_fsl, y_fsl, state_query, y_query, AdjMtx            
def add_sample(qnn, weights, state_fsl, y_fsl, state_query, y_query, AdjMtx, args):
    if args.strategy=='RAND':
        state_fsl, y_fsl, state_query, y_query, AdjMtx = random_sample(state_fsl, y_fsl, state_query, y_query, AdjMtx, args)
        
    elif args.strategy=='ENTRO':
        entropy = torch.stack([decision_entropy(qnn.qnode_qnn(weights, x)) for x in state_query])
        state_fsl, y_fsl, state_query, y_query, AdjMtx = active_sample(state_fsl, y_fsl, state_query, y_query, entropy, AdjMtx, args)
        
    elif args.strategy=='QUANTUM':
        entropy = torch.stack([qnn.qnode_qnn_entropy(weights, x) for x in state_query])
        topo_list = F.softmax(torch.sum(AdjMtx.to(torch.float) > 0.7, dim=1).to(torch.float), dim=0)
        entropy_all = entropy 
        state_fsl, y_fsl, state_query, y_query, AdjMtx = active_sample(state_fsl, y_fsl, state_query, y_query, entropy_all, AdjMtx, args) 
        
    elif args.strategy=='QUANTUM_PLUS':
        entropy_1 = torch.stack([qnn.qnode_qnn_entropy(weights, x) for x in state_query])
        
        entropy_2 = []
        for i in range(data.shape[0]):
            entropy_2.append(qml.math.vn_entropy(data[i,:], indices=[i for i in range(8)]))
        entropy_2 = torch.stack(["{:.2e}".format(tensor.item()) for tensor in entropy_2])
        
        entropy = entropy_1 + entropy_2
            
        state_fsl, y_fsl, state_query, y_query, AdjMtx = active_sample(state_fsl, y_fsl, state_query, y_query, entropy, AdjMtx, args) 
        
    return state_fsl, y_fsl, state_query, y_query, AdjMtx
def return_entropy(qnn, weights, state_fsl, y_fsl, state_query, y_query, AdjMtx, args):
    if args.strategy=='ENTRO':
        entropy = torch.stack([decision_entropy(qnn.qnode_qnn(weights, x)) for x in state_query])
        state_fsl, y_fsl, state_query, y_query, AdjMtx = active_sample(state_fsl, y_fsl, state_query, y_query, entropy, AdjMtx, args)
        
    elif args.strategy=='QUANTUM':
        entropy = torch.stack([qnn.qnode_qnn_entropy(weights, x) for x in state_query])
        topo_list = F.softmax(torch.sum(AdjMtx.to(torch.float) > 0.7, dim=1).to(torch.float), dim=0)
        entropy_all = entropy 
        state_fsl, y_fsl, state_query, y_query, AdjMtx = active_sample(state_fsl, y_fsl, state_query, y_query, entropy_all, AdjMtx, args)         
    return state_fsl, y_fsl, state_query, y_query, AdjMtx, entropy
def add_sample(qnn, weights, state_fsl, y_fsl, state_query, y_query, AdjMtx, args):
    if args.strategy=='RAND':
        state_fsl, y_fsl, state_query, y_query, AdjMtx = random_sample(state_fsl, y_fsl, state_query, y_query, AdjMtx, args)
    elif args.strategy=='ENTRO':
        entropy = torch.stack([decision_entropy(qnn.qnode_qnn(weights, x)) for x in state_query])
        state_fsl, y_fsl, state_query, y_query, AdjMtx = active_sample(state_fsl, y_fsl, state_query, y_query, entropy, AdjMtx, args)
        
    elif args.strategy=='QUANTUM':
        entropy = torch.stack([qnn.qnode_qnn_entropy(weights, x) for x in state_query])
        topo_list = F.softmax(torch.sum(AdjMtx.to(torch.float) > 0.7, dim=1).to(torch.float), dim=0)
        entropy_all = entropy 
        state_fsl, y_fsl, state_query, y_query, AdjMtx = active_sample(state_fsl, y_fsl, state_query, y_query, entropy_all, AdjMtx, args) 
    elif args.strategy=='QUANTUM_PLUS':
        entropy_1 = torch.stack([qnn.qnode_qnn_entropy(weights, x) for x in state_query])
        entropy_2 = []
        for i in range(data.shape[0]):
            entropy_2.append(qml.math.vn_entropy(data[i,:], indices=[i for i in range(8)]))
        entropy_2 = torch.stack(["{:.2e}".format(tensor.item()) for tensor in entropy_2])
        
        entropy = entropy_1 + entropy_2
            
        state_fsl, y_fsl, state_query, y_query, AdjMtx = active_sample(state_fsl, y_fsl, state_query, y_query, entropy, AdjMtx, args) 
        
    return state_fsl, y_fsl, state_query, y_query, AdjMtx


def split_fsl_query_random(X_train_state, Y_train, images_train, labels_train, AdjMtx, args):
    result_indices = []    
    for class_label in range(args.N_WAY):
        class_indices = [i for i, label in enumerate(labels_train) if label == class_label]
        fsl_indices = random.sample(class_indices, args.N_SHOT)
        result_indices.extend(fsl_indices)
        
    images_fsl, labels_fsl, state_fsl, y_fsl, images_query, labels_query, state_query, y_query = split_fsl_query(X_train_state, Y_train, images_train, labels_train, result_indices)
    
    not_indices = [i for i in range(images_train.shape[0]) if i not in result_indices]
    AdjMtx = AdjMtx[:, not_indices]
    AdjMtx = AdjMtx[not_indices, :]
        
    return images_fsl, labels_fsl, state_fsl, y_fsl, images_query, labels_query, state_query, y_query, AdjMtx