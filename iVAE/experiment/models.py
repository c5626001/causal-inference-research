"""VAE(prior固定)とiVAE(p(z|u)で条件付けたprior)。

encoder/decoderの構造はVAE/iVAEで完全に共通。違いはpriorだけ:
- VAE: p(z) = N(0, I)固定(uを見ない)
- iVAE: p(z|u)をuから出すprior_net(uを見る)

`ModelConfig.use_prior_net` でこの2モデルを切り替える。
"""
from dataclasses import dataclass

import torch
import torch.nn as nn


def _make_trunk(d_in, hidden, n_layers):
    """n_layers個のLinear+ReLUを積んだ共通の隠れ層部分。"""
    layers = [nn.Linear(d_in, hidden), nn.ReLU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    """q(z|x)、あるいはn_segments>0ならq(z|x,u)。
    uはスイッチではなく、xと同列のただの入力として連結する(S-learner型の条件付けと同じ発想)。
    """

    def __init__(self, d_x, z_dim, hidden, n_layers=2, n_segments=0):
        super().__init__()
        self.n_segments = n_segments
        self.trunk = _make_trunk(d_x + n_segments, hidden, n_layers)
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)

    def forward(self, x, u_onehot=None):
        h = x if self.n_segments == 0 else torch.cat([x, u_onehot], dim=-1)
        h = self.trunk(h)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, z_dim, d_x, hidden, n_layers=2):
        super().__init__()
        self.trunk = _make_trunk(z_dim, hidden, n_layers)
        self.out = nn.Linear(hidden, d_x)

    def forward(self, z):
        return self.out(self.trunk(z))


class PriorNet(nn.Module):
    """iVAE用: p(z|u)の平均・分散をu(one-hot)から出す。"""

    def __init__(self, n_segments, z_dim, hidden, n_layers=1):
        super().__init__()
        self.trunk = _make_trunk(n_segments, hidden, n_layers)
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)

    def forward(self, u_onehot):
        h = self.trunk(u_onehot)
        return self.mu(h), self.logvar(h)


def reparameterize(mu, logvar):
    return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)


def kl_divergence(mu_q, logvar_q, mu_p, logvar_p):
    """KL(q(z)=N(mu_q,var_q) || p(z)=N(mu_p,var_p))。VAEはmu_p=0, logvar_p=0を渡す。"""
    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    kl = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1)
    return kl.sum(dim=-1).mean()


@dataclass
class ModelConfig:
    z_dim: int = 2
    hidden: int = 64
    n_layers: int = 2  # encoder/decoderの隠れ層の数(ハイパラ探索で振るパラメータ)
    use_prior_net: bool = True  # False=VAE(prior固定), True=iVAE(prior条件付け)
    encoder_sees_u: bool = False  # True: q(z|x,u)。iVAE本来の定式化はこちら(論文準拠)
    obs_sigma: float = 0.1  # 観測ノイズの大きさ(dgp.generate_dataのsigma_xと合わせる)
    epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-3
    seed: int = 0


def build_model(cfg: ModelConfig, d_x, n_segments):
    torch.manual_seed(cfg.seed)
    enc_n_segments = n_segments if (cfg.use_prior_net and cfg.encoder_sees_u) else 0
    nets = {
        "encoder": Encoder(d_x, cfg.z_dim, cfg.hidden, cfg.n_layers, n_segments=enc_n_segments),
        "decoder": Decoder(cfg.z_dim, d_x, cfg.hidden, cfg.n_layers),
    }
    if cfg.use_prior_net:
        nets["prior_net"] = PriorNet(n_segments, cfg.z_dim, cfg.hidden)
    return nets


def loss_fn(nets, x, u_onehot, cfg: ModelConfig):
    """負のELBO。p(x|z) = N(decoder(z), obs_sigma^2 I) として正しくスケールしたNLLを使う
    (単純にmse_lossだけだとobs_sigma=1を仮定したことになり、ノイズが小さいデータでは
    再構成項の勾配がKL項に負けてしまい、zが情報を持たなくなる)。
    """
    mu_q, logvar_q = nets["encoder"](x, u_onehot)
    z = reparameterize(mu_q, logvar_q)
    x_hat = nets["decoder"](z)
    recon = 0.5 * ((x - x_hat) ** 2).sum(dim=-1).mean() / (cfg.obs_sigma ** 2)

    if cfg.use_prior_net:
        mu_p, logvar_p = nets["prior_net"](u_onehot)
    else:
        mu_p = torch.zeros_like(mu_q)
        logvar_p = torch.zeros_like(logvar_q)
    kl = kl_divergence(mu_q, logvar_q, mu_p, logvar_p)
    return recon + kl


def train_model(nets, cfg: ModelConfig, x_train, u_onehot_train):
    torch.manual_seed(cfg.seed)
    all_params = [p for net in nets.values() for p in net.parameters()]
    optimizer = torch.optim.Adam(all_params, lr=cfg.lr)

    n = x_train.shape[0]
    n_batches = max(1, n // cfg.batch_size)
    for epoch in range(cfg.epochs):
        perm = torch.randperm(n)
        for b in range(n_batches):
            idx = perm[b * cfg.batch_size:(b + 1) * cfg.batch_size]
            optimizer.zero_grad()
            loss = loss_fn(nets, x_train[idx], u_onehot_train[idx], cfg)
            loss.backward()
            optimizer.step()


def encode(nets, x, u_onehot=None):
    with torch.no_grad():
        mu_q, _ = nets["encoder"](x, u_onehot)
        return mu_q.numpy()


def compute_val_loss(nets, cfg: ModelConfig, x_val, u_onehot_val):
    """検証データでの負のELBO。モデル選択に使う(真のsは一切見ない、教師なしの基準)。"""
    with torch.no_grad():
        return loss_fn(nets, x_val, u_onehot_val, cfg).item()


def search_best_config(x_train, u_train, x_val, u_val, d_x, n_segments, use_prior_net,
                        hidden_options, n_layers_options, epochs, lr, batch_size, obs_sigma, seed=0,
                        encoder_sees_u=False):
    """hidden幅・層数を軽く振って、検証loss(負のELBO)が一番良い設定を選ぶ。
    (真のsは使わないので、本番(実データ)でも同じやり方が使える)
    """
    results = []
    best = None
    for hidden in hidden_options:
        for n_layers in n_layers_options:
            cfg = ModelConfig(z_dim=x_train.shape[1], hidden=hidden, n_layers=n_layers,
                               use_prior_net=use_prior_net, encoder_sees_u=encoder_sees_u,
                               obs_sigma=obs_sigma,
                               epochs=epochs, batch_size=batch_size, lr=lr, seed=seed)
            nets = build_model(cfg, d_x=d_x, n_segments=n_segments)
            train_model(nets, cfg, x_train, u_train)
            val_loss = compute_val_loss(nets, cfg, x_val, u_val)
            results.append({"hidden": hidden, "n_layers": n_layers, "val_loss": val_loss})
            if best is None or val_loss < best[0]:
                best = (val_loss, cfg, nets)
    return best[1], best[2], results
