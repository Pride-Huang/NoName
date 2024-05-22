import pennylane as qml
from pennylane.optimize import NesterovMomentumOptimizer, AdamOptimizer
from itertools import product

class GENERAL_RAM:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        
        self.device_frqi = qml.device("default.qubit", wires=num_qubits)
        self.qnode_frqi = qml.QNode(self.FRQI, self.device_frqi)        
        
        self.device_amplitude = qml.device("default.qubit", wires=num_qubits)
        self.qnode_amplitude = qml.QNode(self.AmplitudeEmbedding, self.device_amplitude) 
        
        self.device_qnn = qml.device("default.qubit", wires=num_qubits)
        self.qnode_qnn = qml.QNode(self.general_net, self.device_qnn)
        
        self.device_entropy = qml.device("default.qubit", wires=num_qubits)
        self.qnode_qnn_entropy = qml.QNode(self.general_net_entropy, self.device_entropy)
        
    def FRQI(self, input_data, gammas):
        location_strings = list(product([0, 1], repeat=self.num_qubits-1))
        location_wires = list(range(self.num_qubits-1))
        target_wire = self.num_qubits-1
        
        for i in range(self.num_qubits-1):
            qml.Hadamard(i)
        
        for i in range(input_data.shape[0]):
            qml.ctrl(qml.RY, control=location_wires, control_values=location_strings[i])(input_data[i], target_wire)
            qml.ctrl(qml.RZ, control=location_wires, control_values=location_strings[i])(gammas[i], target_wire)
        
        return qml.state()
    
    def AmplitudeEmbedding(self, input_data, gammas):
        qml.AmplitudeEmbedding(features=input_data, wires=range(self.num_qubits), normalize=True)
        return qml.state()
    
    def layer_basis(self, W):
        n = W.shape[0]
        for i in range(n):
            qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)
        
        for i in range(n):
            qml.CNOT(wires=[i, (i+1)%n])

    def layer_CCQC(self, W):
        n = W.shape[0]
        for i in range(n):
            qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)
        
        for i in range(n):
            qml.CNOT(wires=[(i*3)%n, (i-1)*3%n])
            
    def general_net(self, weights, state_FRQI):
        
        wires = list(range(self.num_qubits))
        qml.QubitStateVector(state_FRQI, wires)
        
        for W in weights:
            self.layer_basis(W)
        
        probs = qml.probs(wires=[0,1])
        return probs

    def general_net_entropy(self, weights, state_FRQI):
        
        wires = list(range(self.num_qubits))
        qml.QubitStateVector(state_FRQI, wires)
        
        for W in weights:
            self.layer_basis(W)
        
        entropy = qml.vn_entropy(wires=[0,1])
        return entropy
