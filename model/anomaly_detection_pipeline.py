from cgi import test
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm


class VAEAnomalyDetection(nn.Module):

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAEAnomalyDetection, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)  # 输出为两倍
        )
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, input_dim),
                                     nn.Sigmoid())

    def reparameterize(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, x):
        org_size = x.size()
        batch = org_size[0]
        x = x.view(batch, -1)

        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z).view(size=org_size)

        return recon_x, mu, logvar

    def anomaly_score(self, x):
        with torch.no_grad():
            x_recon, mu, logvar = self(x)
            recon_loss = nn.MSELoss(reduction='none')(x_recon, x).sum(dim=1)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),
                                      dim=1)
            total_loss = recon_loss + kl_div
        return total_loss

    def predict_anomaly(self, x, threshold=0.5):
        scores = self.anomaly_score(x)
        predictions = scores > threshold
        return predictions

    def accuracy(self, x, threshold=0.5):
        with torch.no_grad():
            scores = self.anomaly_score(x)
            true_labels = (scores > threshold).to(torch.float)
            acc = torch.sum(true_labels == x) / x.numel()
        return acc.item()



def train_vae_anomaly_detection(X_train,
                                X_test,
                                device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                                progress=False,
                                num_epoch=100):
    input_dim = X_train.shape[1]  # 输入特征的维度
    latent_dim = 8
    hidden_dim = 32

    vae_anomaly = VAEAnomalyDetection(input_dim, hidden_dim,
                                      latent_dim).to(device)
    optimizer = torch.optim.Adam(vae_anomaly.parameters(), lr=0.001)

    if progress:
        writer = SummaryWriter()

    # 转换为PyTorch张量
    X_train_torch = torch.tensor(X_train, dtype=torch.float).to(device)
    X_test_torch = torch.tensor(X_test, dtype=torch.float).to(device)

    batch_size = 4096  # 可以根据数据大小调整批大小
    train_data = torch.utils.data.TensorDataset(X_train_torch, X_train_torch)
    test_loader = torch.utils.data.TensorDataset(X_test_torch, X_test_torch)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_loader,
                                              batch_size=batch_size,
                                              shuffle=True)

    best_loss = 1e9
    best_epoch = 0

    for epoch in range(num_epoch):
        vae_anomaly.train()  # 将模型设置为训练模式
        train_loss = 0.0
        train_num = len(train_loader.dataset)

        # 使用 tqdm 添加进度条
        with tqdm(train_loader,
                  desc=f'Epoch {epoch + 1}/{num_epoch}',
                  leave=False) as t:
            for data in t:
                inputs, _ = data
                inputs = inputs.to(device)  # 将输入数据移到GPU上
                optimizer.zero_grad()
                inputs_recon, mu, logvar = vae_anomaly(inputs)
                recon_loss = nn.MSELoss()(inputs_recon, inputs)
                kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) -
                                          logvar.exp())
                loss = recon_loss + kl_div

                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        train_loss /= train_num

        # 在每个 epoch 结束后评估模型性能
        vae_anomaly.eval()  # 将模型设置为评估模式
        test_loss = 0.0
        test_num = len(test_loader.dataset)
        with torch.no_grad():
            for data in test_loader:
                inputs, _ = data
                inputs = inputs.to(device)  # 将输入数据移到GPU上
                inputs_recon, mu, logvar = vae_anomaly(inputs)
                recon_loss = nn.MSELoss()(inputs_recon, inputs)
                kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) -
                                          logvar.exp())
                loss = recon_loss + kl_div

                test_loss += loss.item()

        test_loss /= test_num

        if progress:
            writer.add_scalar('Loss/Train', train_loss, epoch + 1)
            writer.add_scalar('Loss/Test', test_loss, epoch + 1)

        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch

    if progress:
        writer.close()
    print('Best epoch:', best_epoch)
    return vae_anomaly
