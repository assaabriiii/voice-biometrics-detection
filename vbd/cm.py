"""Countermeasure (liveness) models: LFCC-LCNN and RawNet2. Both take raw 16 kHz waveforms (B, T) and return 2 logits
(index 1 = bona fide). The liveness score used downstream is logit[1] - logit[0] (higher = more likely live)."""
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchaudio

N_SAMPLES = 64000  # 4 s


def fix_len(x, n=N_SAMPLES, rng=None):
    """Repeat-pad or crop a 1-D numpy waveform to n samples (random crop if rng is given, else from the start)."""
    if len(x) >= n:
        s = 0 if rng is None else int(rng.integers(0, len(x) - n + 1))
        return x[s:s + n]
    return np.tile(x, int(math.ceil(n / len(x))))[:n]


# ---------------------------------------------------------------- LFCC-LCNN
class MFM(nn.Module):
    """Max-feature-map: halves the channels by an element-wise max over channel pairs."""
    def forward(self, x):
        a, b = torch.chunk(x, 2, dim=1)
        return torch.max(a, b)


def _conv(i, o, k, s=1, p=0):
    return nn.Sequential(nn.Conv2d(i, o * 2, k, s, p), MFM())


class LCNN(nn.Module):
    def __init__(self, n_lfcc=20, dropout=0.5):
        super().__init__()
        self.lfcc = torchaudio.transforms.LFCC(sample_rate=16000, n_filter=20, n_lfcc=n_lfcc,
                                               speckwargs=dict(n_fft=512, win_length=320, hop_length=160))
        self.net = nn.Sequential(
            _conv(1, 32, 5, 1, 2), nn.MaxPool2d(2, 2),
            _conv(32, 32, 1), nn.BatchNorm2d(32), _conv(32, 48, 3, 1, 1), nn.MaxPool2d(2, 2), nn.BatchNorm2d(48),
            _conv(48, 48, 1), nn.BatchNorm2d(48), _conv(48, 64, 3, 1, 1), nn.MaxPool2d(2, 2),
            _conv(64, 64, 1), nn.BatchNorm2d(64), _conv(64, 32, 3, 1, 1), nn.BatchNorm2d(32),
            _conv(32, 32, 1), nn.BatchNorm2d(32), _conv(32, 32, 3, 1, 1), nn.MaxPool2d(2, 2))
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(32 * (3 * n_lfcc // 16), 64, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, 2)

    def feats(self, wav):
        with torch.no_grad():
            x = self.lfcc(wav)                                   # (B, n_lfcc, T)
            d1 = torchaudio.functional.compute_deltas(x); d2 = torchaudio.functional.compute_deltas(d1)
            x = torch.cat([x, d1, d2], 1)
            return x - x.mean(-1, keepdim=True)

    def forward(self, wav):
        x = self.net(self.feats(wav).unsqueeze(1))               # (B, 32, F/16, T/16)
        x = self.drop(x).permute(0, 3, 1, 2).flatten(2)          # (B, T', 32*F')
        x, _ = self.lstm(x)
        return self.fc(x.mean(1))


# ---------------------------------------------------------------- RawNet2
class SincConv(nn.Module):
    """Fixed mel-spaced band-pass sinc filters applied to the raw waveform."""
    def __init__(self, out_ch=20, k=1024, sr=16000):
        super().__init__()
        k += 1 - k % 2
        mel = np.linspace(0, 2595 * np.log10(1 + (sr / 2) / 700), out_ch + 1)
        hz = 700 * (10 ** (mel / 2595) - 1)
        t = np.arange(-(k - 1) / 2, (k - 1) / 2 + 1)
        filt = np.zeros((out_ch, k), np.float32)
        for i in range(out_ch):
            lo, hi = hz[i], hz[i + 1]
            h = (2 * hi / sr) * np.sinc(2 * hi * t / sr) - (2 * lo / sr) * np.sinc(2 * lo * t / sr)
            filt[i] = np.hamming(k) * h
        self.register_buffer("filt", torch.from_numpy(filt).unsqueeze(1)); self.k = k

    def forward(self, x):
        return F.conv1d(x.unsqueeze(1), self.filt, padding=self.k // 2)


class ResBlock(nn.Module):
    def __init__(self, i, o, first=False):
        super().__init__()
        self.first = first
        self.bn1 = nn.BatchNorm1d(i); self.c1 = nn.Conv1d(i, o, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(o); self.c2 = nn.Conv1d(o, o, 3, padding=1)
        self.down = nn.Conv1d(i, o, 1) if i != o else None
        self.pool = nn.MaxPool1d(3); self.fms = nn.Linear(o, o)

    def forward(self, x):
        y = x if self.first else F.leaky_relu(self.bn1(x), 0.3)
        y = self.c2(F.leaky_relu(self.bn2(self.c1(y)), 0.3))
        y = self.pool(y + (x if self.down is None else self.down(x)))
        s = torch.sigmoid(self.fms(y.mean(-1))).unsqueeze(-1)   # feature-map scaling
        return y * s + s


class RawNet2(nn.Module):
    def __init__(self, ch=(20, 20, 128), gru=1024, emb=1024):
        super().__init__()
        self.sinc = SincConv(ch[0]); self.bn0 = nn.BatchNorm1d(ch[0])
        self.blocks = nn.Sequential(ResBlock(ch[0], ch[1], first=True), ResBlock(ch[1], ch[1]),
                                    ResBlock(ch[1], ch[2]), ResBlock(ch[2], ch[2]), ResBlock(ch[2], ch[2]), ResBlock(ch[2], ch[2]))
        self.bn = nn.BatchNorm1d(ch[2]); self.gru = nn.GRU(ch[2], gru, num_layers=3, batch_first=True)
        self.fc1 = nn.Linear(gru, emb); self.fc2 = nn.Linear(emb, 2)

    def forward(self, wav):
        x = F.leaky_relu(self.bn0(F.max_pool1d(torch.abs(self.sinc(wav)), 3)), 0.3)
        x = F.leaky_relu(self.bn(self.blocks(x)), 0.3).transpose(1, 2)
        self.gru.flatten_parameters(); x, _ = self.gru(x)
        return self.fc2(self.fc1(x[:, -1]))


def build(name):
    return {"lcnn": LCNN, "rawnet2": RawNet2}[name]()


@torch.no_grad()
def score(model, wavs, bs=64, device="cuda"):
    """Liveness scores for a list of numpy waveforms (first 4 s, repeat-padded)."""
    model.eval(); out = []
    for i in range(0, len(wavs), bs):
        x = torch.from_numpy(np.stack([fix_len(w) for w in wavs[i:i + bs]]).astype(np.float32)).to(device)
        lg = model(x).float(); out.append((lg[:, 1] - lg[:, 0]).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)
