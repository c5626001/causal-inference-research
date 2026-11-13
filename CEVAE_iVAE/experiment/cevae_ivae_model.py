"""CEVAE(iVAE ver): CEVAE/experiment/cevae_model.py の prior だけを p(z|u) に変えたもの。

変更点はこれだけ:
- prior: 固定 N(0,I) → PriorNet(u) が出す N(mu_p(u), var_p(u))
- KL: KL(q(z|x,t,y,u) || N(0,I)) → KL(q(z|x,t,y,u) || p(z|u))
- encoder_z は u も見るようにする(真の事後分布p(z|x,t,y,u)がuに依存するため)
- encoder_t, encoder_y(CEVAE独自の補助NW、q(t|x), q(y|t,x)にuを入れる理論的必要はない。
  実際にuはTの情報をほとんど持たない(確認済み)ので、xだけを見る元の形のままにする
- decoder_x, decoder_t, decoder_y は変更なし(zだけから生成。iVAE論文と同じ、uを見るのはprior/encoder_zだけ)
"""
from dataclasses import dataclass

import torch
import torch.nn as nn


class DecoderX(nn.Module):
    def __init__(self, z_dim, d, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim, hidden), nn.ReLU(), nn.Linear(hidden, d))

    def forward(self, z):
        return self.net(z)


class DecoderT(nn.Module):
    def __init__(self, z_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, z):
        return self.net(z).squeeze(-1)


class DecoderY(nn.Module):
    def __init__(self, z_dim, hidden):
        super().__init__()
        self.head0 = nn.Sequential(nn.Linear(z_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.head1 = nn.Sequential(nn.Linear(z_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, z):
        return self.head0(z).squeeze(-1), self.head1(z).squeeze(-1)


class EncoderT(nn.Module):
    """q(t|x)"""

    def __init__(self, d, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class EncoderY(nn.Module):
    """q(y|t,x)"""

    def __init__(self, d, hidden):
        super().__init__()
        self.head0 = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.head1 = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.head0(x).squeeze(-1), self.head1(x).squeeze(-1)


class EncoderZ(nn.Module):
    """q(z|x,t,y,u)"""

    def __init__(self, d, n_segments, z_dim, hidden):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d + n_segments, hidden), nn.ReLU())
        self.head0 = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.ReLU())
        self.head1 = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.ReLU())
        self.mu0 = nn.Linear(hidden, z_dim)
        self.logvar0 = nn.Linear(hidden, z_dim)
        self.mu1 = nn.Linear(hidden, z_dim)
        self.logvar1 = nn.Linear(hidden, z_dim)

    def forward(self, x, t, y, u_onehot):
        h = self.trunk(torch.cat([x, u_onehot], dim=-1))
        hy = torch.cat([h, y.unsqueeze(-1)], dim=-1)
        h0, h1 = self.head0(hy), self.head1(hy)
        mu0, lv0 = self.mu0(h0), self.logvar0(h0)
        mu1, lv1 = self.mu1(h1), self.logvar1(h1)
        tc = t.unsqueeze(-1)
        return torch.where(tc == 1, mu1, mu0), torch.where(tc == 1, lv1, lv0)


class EncoderZShared(nn.Module):
    """q(z|x,t,y,u): S-learner型。tをスイッチ(head0/head1の選択)ではなく、
    xやyと同列のただの入力として1本のネットワークに混ぜる。"""

    def __init__(self, d, n_segments, z_dim, hidden):
        super().__init__()
        d_in = d + n_segments + 2  # x, u, t, y
        self.trunk = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)

    def forward(self, x, t, y, u_onehot):
        h = self.trunk(torch.cat([x, u_onehot, t.unsqueeze(-1), y.unsqueeze(-1)], dim=-1))
        return self.mu(h), self.logvar(h)


class PriorNet(nn.Module):
    """p(z|u): iVAE/experiment/models.py の PriorNet と同じ役割"""

    def __init__(self, n_segments, z_dim, hidden):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(n_segments, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)

    def forward(self, u_onehot):
        h = self.trunk(u_onehot)
        return self.mu(h), self.logvar(h)


def reparameterize(mu, logvar):
    return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)


def kl_divergence(mu_q, logvar_q, mu_p, logvar_p):
    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    kl = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1)
    return kl.sum(dim=-1).mean()


def linear_kl_anneal(step, total_steps):
    return 1.0 if total_steps <= 0 else min(1.0, step / total_steps)


@dataclass
class CEVAEiVAEConfig:
    z_dim: int = 1
    n_segments: int = 20
    hidden: int = 64
    beta_kl_max: float = 1.0
    kl_anneal_epochs: int = 75
    w_recon_y: float = 1.0
    w_aux_y: float = 1.0
    w_recon_t: float = 1.0  # decoder_tのrecon_t項の重み(0にするとTのショートカット圧力を切れる)
    recon_t_anneal_epochs: int = 0  # 0=アニーリングなし(最初からw_recon_t)。>0ならbeta_klと同じ形で徐々に強める
    epochs: int = 150
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 0
    encoder_z_type: str = "branching"  # "branching"(元のT-learner型, head0/head1) / "shared"(S-learner型, 1本)


def build_cevae_ivae(cfg: CEVAEiVAEConfig, d: int):
    torch.manual_seed(cfg.seed)
    if cfg.encoder_z_type == "shared":
        encoder_z = EncoderZShared(d, cfg.n_segments, cfg.z_dim, cfg.hidden)
    elif cfg.encoder_z_type == "branching":
        encoder_z = EncoderZ(d, cfg.n_segments, cfg.z_dim, cfg.hidden)
    else:
        raise ValueError(f"unknown encoder_z_type: {cfg.encoder_z_type!r}")
    return {
        "decoder_x": DecoderX(cfg.z_dim, d, cfg.hidden),
        "decoder_t": DecoderT(cfg.z_dim, cfg.hidden),
        "decoder_y": DecoderY(cfg.z_dim, cfg.hidden),
        "encoder_t": EncoderT(d, cfg.hidden),
        "encoder_y": EncoderY(d, cfg.hidden),
        "encoder_z": encoder_z,
        "prior_net": PriorNet(cfg.n_segments, cfg.z_dim, cfg.hidden),
    }


def cevae_ivae_loss_fn(nets, x, t, y, u_onehot, beta_kl, w_recon_y=1.0, w_aux_y=1.0, w_recon_t=1.0):
    mu_z, logvar_z = nets["encoder_z"](x, t, y, u_onehot)
    z = reparameterize(mu_z, logvar_z)

    x_hat = nets["decoder_x"](z)
    t_logit_p = nets["decoder_t"](z)
    y0p, y1p = nets["decoder_y"](z)
    y_hat_p = torch.where(t == 1, y1p, y0p)

    recon_x = nn.functional.mse_loss(x_hat, x)
    recon_t = nn.functional.binary_cross_entropy_with_logits(t_logit_p, t)
    recon_y = nn.functional.mse_loss(y_hat_p, y)

    mu_p, logvar_p = nets["prior_net"](u_onehot)
    kl = kl_divergence(mu_z, logvar_z, mu_p, logvar_p)

    t_logit_q = nets["encoder_t"](x)
    y0q, y1q = nets["encoder_y"](x)
    y_hat_q = torch.where(t == 1, y1q, y0q)
    aux_t = nn.functional.binary_cross_entropy_with_logits(t_logit_q, t)
    aux_y = nn.functional.mse_loss(y_hat_q, y)

    return recon_x + w_recon_t * recon_t + w_recon_y * recon_y + beta_kl * kl + aux_t + w_aux_y * aux_y


def train_cevae_ivae(nets, cfg: CEVAEiVAEConfig, x_train, t_train, y_train, u_train):
    torch.manual_seed(cfg.seed)
    all_params = [p for net in nets.values() for p in net.parameters()]
    optimizer = torch.optim.Adam(all_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    n = x_train.shape[0]
    n_batches = max(1, n // cfg.batch_size)
    for epoch in range(cfg.epochs):
        beta_kl = cfg.beta_kl_max * linear_kl_anneal(epoch, cfg.kl_anneal_epochs)
        w_recon_t = cfg.w_recon_t * linear_kl_anneal(epoch, cfg.recon_t_anneal_epochs)
        perm = torch.randperm(n)
        for b in range(n_batches):
            idx = perm[b * cfg.batch_size:(b + 1) * cfg.batch_size]
            optimizer.zero_grad()
            loss = cevae_ivae_loss_fn(
                nets, x_train[idx], t_train[idx], y_train[idx], u_train[idx],
                beta_kl, cfg.w_recon_y, cfg.w_aux_y, w_recon_t,
            )
            loss.backward()
            optimizer.step()


def compute_val_elbo(nets, x_val, t_val, y_val, u_val):
    with torch.no_grad():
        return cevae_ivae_loss_fn(nets, x_val, t_val, y_val, u_val, beta_kl=1.0).item()


def estimate_ite(nets, x, u_onehot):
    """テスト時のITE推定: 真のt, yは一切使わない。uは処置・結果と無関係な外生変数なので
    そのまま使ってよい(反実仮想を考える場面でもuは変わらない)。"""
    with torch.no_grad():
        t_hat = (torch.sigmoid(nets["encoder_t"](x)) > 0.5).float()
        y0q, y1q = nets["encoder_y"](x)
        y_hat = torch.where(t_hat == 1, y1q, y0q)
        mu_z, _ = nets["encoder_z"](x, t_hat, y_hat, u_onehot)
        y0p, y1p = nets["decoder_y"](mu_z)
        return (y1p - y0p).numpy()
