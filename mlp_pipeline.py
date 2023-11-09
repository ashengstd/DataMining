import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tensorboardX import SummaryWriter


class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
def train_mlp(X_train, y_train, X_test, y_test, device='cuda:0', progress=False, num_epoch=100):
    input_dim = X_train.shape[1]
    hidden_size = 64


    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.fit_transform(X_test)

    if progress:
        writer = SummaryWriter()

    X_train_torch = torch.tensor(X_train, dtype=torch.float).to(device)
    X_test_torch = torch.tensor(X_test, dtype=torch.float).to(device)
    y_train_torch = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test_torch = torch.tensor(y_test, dtype=torch.long).to(device)

    num_classes = len(torch.unique(y_train_torch))
    mlp = MLPModel(input_dim, hidden_size, num_classes).to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)


    train_data = TensorDataset(X_train_torch, y_train_torch)
    test_data = TensorDataset(X_test_torch, y_test_torch)


    batch_size = 256
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()

    best_loss = 1e9
    best_epoch = 0

    for epoch in range(num_epoch):
        mlp.train()
        train_loss = 0.0
        train_num = len(train_loader.dataset)

        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epoch}', leave=False) as t:
            for data in t:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = mlp(inputs)
                loss = criterion(outputs, labels)

                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        train_loss /= train_num

        mlp.eval()
        test_loss = 0.0
        test_num = len(test_loader.dataset)
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = mlp(inputs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                test_loss += loss.item()

        test_loss /= test_num
        test_accuracy = correct / total

        if progress:
            writer.add_scalar('Loss/Train', train_loss, epoch + 1)
            writer.add_scalar('Loss/Test', test_loss, epoch + 1)
            writer.add_scalar('Acc/Test', test_accuracy, epoch + 1)

        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch

            # torch.save(mlp.state_dict(), 'best_model.pth')

    if progress:
        writer.close()
    print('Best epoch:', best_epoch)
    return mlp