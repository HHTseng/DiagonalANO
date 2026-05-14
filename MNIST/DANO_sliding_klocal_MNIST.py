import os
from pathlib import Path
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import pennylane as qml
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from datetime import datetime

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_H', type=float, default=1e-1)
# parser.add_argument('--n-qubits', type=int, default=16)
parser.add_argument('--resize-img', type=int, default=4)
parser.add_argument('--k-local', type=int, default=2)
parser.add_argument('--num-classes', type=int, default=10)
parser.add_argument('--vqc-depth', type=int, default=6)
parser.add_argument('--batch-size', type=int, default=200)
parser.add_argument('--epochs', type=int, default=30)
args = parser.parse_args()

n_qubits = args.resize_img **2


def H_layer(nqubits):
    """Layer of single-qubit Hadamard gates.
    """
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)

def RY_layer(w):
    """Layer of parametrized qubit rotations around the y axis."""
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)

def entangling_layer(nqubits):
    """ Layer of CNOTs followed by another shifted layer of CNOT."""
    # In other words it should apply something like :
    # CNOT  CNOT  CNOT  CNOT...  CNOT
    #   CNOT  CNOT  CNOT...  CNOT
    for i in range(0, nqubits - 1, 2):  # Loop over even indices: qubit = 0,2,4,...
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  qubit = 1,3,5,...
        qml.CNOT(wires=[i, i + 1])


# Define actual circuit architecture
def quantum_net(X, θ, H):
    """ The variational quantum circuit. """
    # Start from state |+> , unbiased w.r.t. |0> and |1>
    H_layer(n_qubits)

    # Embed features in the quantum node
    RY_layer(X)

    # Sequence of trainable variational layers
    for k in range(args.vqc_depth):
        entangling_layer(n_qubits)
        RY_layer(θ[k])

    # Compute Expectation values (for multi-class prediction) using n_local Hermitians
    exp_vals = [qml.expval(qml.Hermitian(H[q], wires=(np.arange(q, q + args.k_local) % n_qubits).tolist())) for q in range(n_qubits)]
    return exp_vals


class DANO_VQC_Model(nn.Module):
    '''VQC with adaptive nonlocal observables'''

    def __init__(self, img_size):
        super(DANO_VQC_Model, self).__init__()
        self.θ = nn.Parameter(0.01 * torch.randn(args.vqc_depth, n_qubits))  # VQC rotation params

        self.dev = qml.device("default.qubit", wires=n_qubits)  # Can use different simulation backend or quantum computers.
        self.VQC = qml.QNode(quantum_net, self.dev, interface = "torch")

        # diagonal elements (all real) of Hermitian matrices
        self.D = nn.ParameterList([nn.Parameter(torch.empty(2 ** args.k_local)) for _ in range(n_qubits)])

        for _ in range(n_qubits):
            nn.init.normal_(self.D[_], std=2.)

    def forward(self, X):
        z1 = X.reshape(-1, n_qubits)

        # create observable H here
        self.H = [torch.diag(self.D[_]) for _ in range(n_qubits)]  # first few observables: k-locals
        q_out = torch.stack([torch.stack(self.VQC(z, self.θ, self.H)).float() for z in z1])
        return q_out[:, :args.num_classes]

# Define a custom transformation
class NormalizeToPiTransform:
    def __call__(self, x):
        """
        Transform values from range [0, 1] to [-pi, pi].
        Assumes input x is already normalized to [0, 1].
        """
        return x * (2 * np.pi) - np.pi

# Compose transformations
transform = transforms.Compose([
    transforms.Resize((args.resize_img, args.resize_img)),
    transforms.ToTensor(),  # Convert image to tensor with range [0, 1]
    NormalizeToPiTransform(),  # Scale to [-pi, pi]
])

# Download the MNIST dataset
train_dataset = datasets.MNIST(
    root="mnist_data",  # Directory to save the dataset
    train=True,         # Download the training set
    transform=transform,  # Apply the transform
    download=True       # Download if not already downloaded
)

test_dataset = datasets.MNIST(
    root="mnist_data",
    train=False,        # Download the test set
    transform=transform,
    download=True
)


# Parameters
samples_per_class_train = 1000  # Number of samples per class in the subset
samples_per_class_test = 100

# Initialize a dictionary to store indices for each class
class_indices = {i: [] for i in range(args.num_classes)}

# Populate the dictionary with indices
for idx, (_, label) in enumerate(train_dataset):
    if len(class_indices[label]) < samples_per_class_train:  # Check if class is filled
        class_indices[label].append(idx)

# Combine all indices from each class
balanced_indices = [idx for indices in class_indices.values() for idx in indices]

# Create a subset using the balanced indices
balanced_train_subset = Subset(train_dataset, balanced_indices)

# Initialize a dictionary to store indices for each class
class_indices = {i: [] for i in range(args.num_classes)}

# Populate the dictionary with indices
for idx, (_, label) in enumerate(test_dataset):
    if len(class_indices[label]) < samples_per_class_test:  # Check if class is filled
        class_indices[label].append(idx)

# Combine all indices from each class
balanced_indices = [idx for indices in class_indices.values() for idx in indices]

# Create a subset using the balanced indices
balanced_test_subset = Subset(test_dataset, balanced_indices)

train_dataset_from_subset, valid_dataset_from_subset = train_test_split(balanced_train_subset, test_size=0.1, random_state=999, shuffle=True)


# Create DataLoader for training and testing
train_loader = torch.utils.data.DataLoader(dataset=train_dataset_from_subset, batch_size=args.batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset_from_subset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=balanced_test_subset, batch_size=args.batch_size, shuffle=False)

# Check the dataset
print(f"Number of training samples: {len(train_dataset_from_subset)}")
print(f"Number of validate samples: {len(valid_dataset_from_subset)}")
print(f"Number of testing samples: {len(balanced_test_subset)}")

# Example: Display one batch of images
data_iter = iter(train_loader)
images, labels = next(data_iter)

model = DANO_VQC_Model(img_size=args.resize_img).to(DEVICE)


# Split parameters into two groups
H_params = []
VQC_params = []
for name, param in model.named_parameters():
    if 'A' in name or 'B' in name or 'D' in name:
        H_params.append(param)  # Hermitian parameters
    else:
        VQC_params.append(param)

# initialize optimizer
H_optimizer = torch.optim.Adam(H_params, lr=args.lr_H)
optimizer = torch.optim.Adam(VQC_params, lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()

# Initialize dictionaries to store metrics
metrics = {
    'train_loss': [],
    'train_accuracy': [],
    'valid_loss': [],
    'valid_accuracy': [],
    'test_loss': [],
    'test_accuracy': []
}

H_epochs = []  # to record all non-local Hermitians
for epoch in range(args.epochs):
    print('*' * 30)
    print(f'Epoch {epoch + 1}'.center(30))
    print('*' * 30)

    model.train()
    total_loss = 0.0
    total_acc = 0.0
    N = 0  # total samples

    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs} - Training')
    for X, y in train_pbar:
        # send to device
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        N += len(y)  # accumulate batch sample size

        # set the gradient to zero
        optimizer.zero_grad()
        H_optimizer.zero_grad()

        # compute the vqc and the loss
        logits = model(X)
        loss = criterion(logits, y)

        # print(f'H: {vqc.H[0]}')

        total_loss += loss.item() * len(y)
        batch_acc = sum(torch.argmax(logits, dim=1) == y) / len(y)
        total_acc += sum(torch.argmax(logits, dim=1) == y)
        loss.backward()
        optimizer.step()
        H_optimizer.step()

        train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100 * batch_acc:.2f}%'})

    train_epoch_acc = total_acc / N
    train_epoch_loss = total_loss / N
    H_epochs.append(model.H)  # record non-local Hermitian of each epoch
    
    # Store training metrics
    metrics['train_loss'].append(train_epoch_loss)
    metrics['train_accuracy'].append(float(train_epoch_acc))

    tqdm.write(f'Epoch training loss {train_epoch_loss:.4f}, Train acc: {100 * train_epoch_acc:.2f}%')


    # validation
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    N = 0
    for X, y in tqdm(valid_loader):
        # send to device
        X = X.to(DEVICE)
        y = y.to(DEVICE)

        N += len(y)  # accumulate batch sample size

        # compute the vqc and the loss
        with torch.no_grad():
            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)
            total_acc += sum(torch.argmax(logits, dim=1) == y)

    valid_acc = total_acc / N
    valid_loss = total_loss / N
    
    # Store validation metrics
    metrics['valid_loss'].append(valid_loss)
    metrics['valid_accuracy'].append(float(valid_acc))
    
    tqdm.write(f'Valid loss: {valid_loss:.4f}, Valid acc: {100 * valid_acc:.2f}%')

    # Test vqc
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    N = 0
    for X, y in tqdm(test_loader):
        # send to device
        X = X.to(DEVICE)
        y = y.to(DEVICE)

        N += len(y)  # accumulate batch sample size

        # compute the vqc and the loss
        with torch.no_grad():
            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)
            total_acc += sum(torch.argmax(logits, dim=1) == y)

    test_acc = total_acc / N
    test_loss = total_loss / N
    
    # Store test metrics
    metrics['test_loss'].append(test_loss)
    metrics['test_accuracy'].append(float(test_acc))
    
    tqdm.write(f'Test loss: {test_loss:.4f}, Test acc: {100 * test_acc:.2f}%')

# Create experiment folder with descriptive name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_name = f"DANO_sliding_{args.k_local}local_{n_qubits}qubits_depth{args.vqc_depth}_lr{args.lr:.0e}_lrH{args.lr_H:.0e}_{args.epochs}epochs_{timestamp}"
experiment_path = f'./experiments/{experiment_name}'
Path(experiment_path).mkdir(parents=True, exist_ok=True)

# Save all metrics and experiment data
experiment_data = {
    'metrics': metrics,
    'H_epochs': H_epochs,
    'args': vars(args),
    'experiment_info': {
        'timestamp': timestamp,
        'device': str(DEVICE),
        'train_samples': len(train_dataset_from_subset),
        'valid_samples': len(valid_dataset_from_subset),
        'test_samples': len(balanced_test_subset)
    }
}

with open(os.path.join(experiment_path, 'experiment_results.pkl'), 'wb') as f:
    pickle.dump(experiment_data, f)

print(f"\nExperiment results saved to: {experiment_path}")
print(f"Final test accuracy: {100 * metrics['test_accuracy'][-1]:.2f}%")


