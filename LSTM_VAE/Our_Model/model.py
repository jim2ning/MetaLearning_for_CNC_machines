import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(LSTMVAE, self).__init__()
        
        # 인코더
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # 잠재 공간으로의 매핑
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_var = nn.Linear(hidden_dim * 2, latent_dim)
        
        # 디코더
        self.decoder = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # 출력 레이어
        self.fc_out = nn.Linear(hidden_dim * 2, input_dim)
        
    def encode(self, x):
        # LSTM 인코더
        _, (hidden, _) = self.encoder(x)
        hidden = hidden[-2:].mean(0)  # 양방향 LSTM의 마지막 은닉 상태 평균
        
        # 잠재 공간 분포 파라미터
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        # 재매개변수화 트릭
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        # 잠재 벡터를 시퀀스로 확장
        z = z.unsqueeze(1).repeat(1, 100, 1)  # 시퀀스 길이를 100으로 가정
        
        # LSTM 디코더
        output, _ = self.decoder(z)
        
        # 출력 레이어
        x_recon = self.fc_out(output)
        return x_recon
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var
    
    def compute_loss(self, x, x_recon, mu, log_var):
        # 재구성 손실
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        
        # KL 발산
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # 전체 손실
        total_loss = recon_loss + kl_loss
        return total_loss 