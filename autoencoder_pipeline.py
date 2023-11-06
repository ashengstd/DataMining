import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class Autoencoder(nn.Module):

    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 12),  # 编码器的第一层
            nn.ReLU(True),
            nn.Linear(12, 8),  # 编码器的第二层
            nn.ReLU(True),
            nn.Linear(8, encoding_dim)  # 编码器的输出层
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 8),  # 解码器的第一层
            nn.ReLU(True),
            nn.Linear(8, 12),  # 解码器的第二层
            nn.ReLU(True),
            nn.Linear(12, input_dim)  # 解码器的输出层
        )

    def extract_features(self, X):
        if isinstance(X, torch.Tensor):
            with torch.no_grad():
                return self.encoder(X)
        else:
            X = torch.Tensor(X).to('cuda:0')
            with torch.no_grad():
                return np.array(self.encoder(X).cpu())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_autoencoder(X_train, X_test, device='cuda:0', progress=False,num_epoch=300):
    input_dim = X_train.shape[1]  # 输入特征的维度
    encoding_dim = 8  # 编码器的输出维度

    autoencoder = Autoencoder(input_dim, encoding_dim).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if progress:
        writer = SummaryWriter()

    # 转换为PyTorch张量
    X_train_torch = torch.tensor(X_train, dtype=torch.float).to(device)
    X_test_torch = torch.tensor(X_test, dtype=torch.float).to(device)

    batch_size = 256  # 可以根据数据大小调整批大小
    train_data = torch.utils.data.TensorDataset(X_train_torch, X_train_torch)
    test_loader = torch.utils.data.TensorDataset(X_test_torch, X_test_torch)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_loader,
                                              batch_size=batch_size,
                                              shuffle=True)
    for epoch in range(num_epoch):
        autoencoder.train()  # 将模型设置为训练模式
        train_loss = 0.0
        # 使用 tqdm 添加进度条
        with tqdm(train_loader,
                  desc=f'Epoch {epoch + 1}/{num_epoch}',
                  leave=False) as t:
            for data in t:
                inputs, _ = data
                optimizer.zero_grad()
                outputs = autoencoder(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

        train_loss /= len(train_loader)

        # 在每个 epoch 结束后评估模型性能
        autoencoder.eval()  # 将模型设置为评估模式
        test_loss = 0.0
        with torch.no_grad():
            for data in test_loader:
                inputs, _ = data
                outputs = autoencoder(inputs)
                loss = criterion(outputs, inputs)
                test_loss += loss.item()

        test_loss /= len(test_loader)

        if progress:
            writer.add_scalar('Loss/Train', train_loss, epoch + 1)
            writer.add_scalar('Loss/Test', test_loss, epoch + 1)
        # if epoch % 10 ==0:
        #     print(
        #         f'Epoch {epoch + 1}/{num_epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}'
        #     )
    if progress:
        writer.close()
    print('Train Done!')
    return autoencoder
