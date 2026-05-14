import os
from pathlib import Path
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_H', type=float, default=1e-1)
parser.add_argument('--k-local', type=int, default=2)
parser.add_argument('--vqc-depth', type=int, default=6)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--num-classes', type=int, default=10)
parser.add_argument('--train-ratio', type=float, default=0.8)
parser.add_argument('--valid-ratio', type=float, default=0.1)
parser.add_argument('--test-ratio', type=float, default=0.1)
parser.add_argument('--pca-path', type=str, default='yale_images_pca.npy')
parser.add_argument('--labels-path', type=str, default='yale_labels.npy')
args = parser.parse_args()

# Load PCA data to determine n_qubits and num_classes
print("="*60)
print("Loading Yale Face PCA data...")
print("="*60)

if not os.path.exists(args.pca_path):
    raise FileNotFoundError(f"PCA embeddings file not found: {args.pca_path}")
if not os.path.exists(args.labels_path):
    raise FileNotFoundError(f"Labels file not found: {args.labels_path}")

X_pca = np.load(args.pca_path)
y_labels = np.load(args.labels_path)

# Set n_qubits from PCA dimension
n_qubits = X_pca.shape[1]

# Encode labels and get num_classes
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_labels)
num_classes = len(label_encoder.classes_)

print(f"PCA embeddings shape: {X_pca.shape}")
print(f"Labels shape: {y_labels.shape}")
print(f"PCA dimension (n_qubits): {n_qubits}")
print(f"Number of unique classes: {num_classes}")
print(f"Class distribution: {dict(zip(*np.unique(y_labels, return_counts=True)))}")
print(f"Label mapping: {dict(zip(label_encoder.classes_, range(num_classes)))}")


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

    def __init__(self):
        super(DANO_VQC_Model, self).__init__()
        self.θ = nn.Parameter(0.01 * torch.randn(args.vqc_depth, n_qubits))  # VQC rotation params

        self.dev = qml.device("default.qubit", wires=n_qubits)  # Can use different simulation backend or quantum computers.
        self.VQC = qml.QNode(quantum_net, self.dev, interface = "torch")

        # diagonal elements (all real) of Hermitian matrices
        self.D = nn.ParameterList([nn.Parameter(torch.empty(2 ** args.k_local)) for _ in range(n_qubits)])

        for _ in range(n_qubits):
            nn.init.normal_(self.D[_], std=2.)

    def forward(self, X):
        # X is already in shape (batch_size, n_qubits) from PCA embeddings
        z1 = X.reshape(-1, n_qubits)

        # create observable H here
        self.H = [torch.diag(self.D[_]) for _ in range(n_qubits)]  # first few observables: k-locals
        q_out = torch.stack([torch.stack(self.VQC(z, self.θ, self.H)).float() for z in z1])
        return q_out[:, :args.num_classes]


# Normalize PCA embeddings to [-pi, pi] for quantum encoding
print("="*60)
print("Normalizing PCA data to [-pi, pi]...")
print("="*60)

X_min = X_pca.min(axis=0)
X_max = X_pca.max(axis=0)
X_normalized = 2 * np.pi * (X_pca - X_min) / (X_max - X_min + 1e-8) - np.pi

print(f"PCA data normalized to range: [{X_normalized.min():.3f}, {X_normalized.max():.3f}]")
print()

# Split data into train, validation, and test sets
print("="*60)
print("Splitting data...")
print("="*60)

assert abs(args.train_ratio + args.valid_ratio + args.test_ratio - 1.0) < 1e-6, \
    "Train, valid, and test ratios must sum to 1.0"

# First split: separate test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X_normalized, y_encoded, test_size=args.test_ratio, random_state=93, stratify=y_encoded
)

# Second split: separate train and validation from remaining data
valid_ratio_adjusted = args.valid_ratio / (args.train_ratio + args.valid_ratio)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_temp, y_temp, test_size=valid_ratio_adjusted, random_state=93, stratify=y_temp
)

print(f"Train set: {X_train.shape[0]} samples ({args.train_ratio*100:.0f}%)")
print(f"Valid set: {X_valid.shape[0]} samples ({args.valid_ratio*100:.0f}%)")
print(f"Test set:  {X_test.shape[0]} samples ({args.test_ratio*100:.0f}%)")
print()

# Convert to PyTorch tensors and create datasets
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_valid_tensor = torch.FloatTensor(X_valid)
y_valid_tensor = torch.LongTensor(y_valid)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create DataLoader for training, validation, and testing
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Check the dataset
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validate samples: {len(valid_dataset)}")
print(f"Number of testing samples: {len(test_dataset)}")
print(f"Number of classes: {num_classes}")
print(f"Number of qubits (PCA dim): {n_qubits}")
print()

# Example: Display one batch
data_iter = iter(train_loader)
features, labels_batch = next(data_iter)
print(f"Batch features shape: {features.shape}")
print(f"Batch labels shape: {labels_batch.shape}")
print()

model = DANO_VQC_Model().to(DEVICE)


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
    for X_batch, y_batch in train_pbar:
        # send to device
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        N += len(y_batch)  # accumulate batch sample size

        # set the gradient to zero
        optimizer.zero_grad()
        H_optimizer.zero_grad()

        # compute the vqc and the loss
        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        total_loss += loss.item() * len(y_batch)
        batch_acc = sum(torch.argmax(logits, dim=1) == y_batch) / len(y_batch)
        total_acc += sum(torch.argmax(logits, dim=1) == y_batch)
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
    for X_batch, y_batch in tqdm(valid_loader):
        # send to device
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        N += len(y_batch)  # accumulate batch sample size

        # compute the vqc and the loss
        with torch.no_grad():
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            total_acc += sum(torch.argmax(logits, dim=1) == y_batch)

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
    for X_batch, y_batch in tqdm(test_loader):
        # send to device
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        N += len(y_batch)  # accumulate batch sample size

        # compute the vqc and the loss
        with torch.no_grad():
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            total_acc += sum(torch.argmax(logits, dim=1) == y_batch)

    test_acc = total_acc / N
    test_loss = total_loss / N

    # Store test metrics
    metrics['test_loss'].append(test_loss)
    metrics['test_accuracy'].append(float(test_acc))

    tqdm.write(f'Test loss: {test_loss:.4f}, Test acc: {100 * test_acc:.2f}%')

# Create experiment folder with descriptive name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_name = f"DANO_sliding_{args.k_local}local_YaleFace_PCA{n_qubits}d_depth{args.vqc_depth}_lr{args.lr:.0e}_lrH{args.lr_H:.0e}_{args.epochs}epochs_{timestamp}"
experiment_path = f'./experiments/{experiment_name}'
Path(experiment_path).mkdir(parents=True, exist_ok=True)

# Save all metrics and experiment data
experiment_data = {
    'metrics': metrics,
    'H_epochs': H_epochs,
    'args': vars(args),
    'label_encoder': label_encoder,
    'experiment_info': {
        'timestamp': timestamp,
        'device': str(DEVICE),
        'dataset': 'Yale Extended Face Database B (PCA)',
        'pca_dimension': n_qubits,
        'num_classes': num_classes,
        'train_samples': len(train_dataset),
        'valid_samples': len(valid_dataset),
        'test_samples': len(test_dataset)
    }
}

with open(os.path.join(experiment_path, 'experiment_results.pkl'), 'wb') as f:
    pickle.dump(experiment_data, f)

# Save model checkpoint
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'H_optimizer_state_dict': H_optimizer.state_dict(),
    'args': vars(args),
    'label_encoder': label_encoder,
    'final_metrics': {
        'train_acc': metrics['train_accuracy'][-1],
        'valid_acc': metrics['valid_accuracy'][-1],
        'test_acc': metrics['test_accuracy'][-1]
    }
}, os.path.join(experiment_path, 'model_checkpoint.pth'))

print(f"\nExperiment results saved to: {experiment_path}")
print(f"Final test accuracy: {100 * metrics['test_accuracy'][-1]:.2f}%")
print("\nGenerated files:")
print(f"  - {experiment_path}/experiment_results.pkl")
print(f"  - {experiment_path}/model_checkpoint.pth")