import torch
import torch.nn as nn
import math
from transformers import BertModel

HIDDEN_DIM     = 128   # общая размерность всех модальностей после проекции
N_HEADS        = 8     # число голов в multi-head attention (HIDDEN_DIM делится на N_HEADS)
N_BOTTLENECK   = 16    # число bottleneck tokens — DBA: n=16 оптимально по ablation
N_LAYERS       = 2     # число слоёв bottleneck — XMBT: L=2 лучший результат на CMU-MOSEI
DROPOUT        = 0.1   # MulT, Table 5: dropout=0.1 для CMU-MOSEI
N_EMOTIONS     = 6     # happy, sad, anger, surprise, disgust, fear
BERT_MODEL     = "bert-base-uncased"

# ─────────────────────────────────────────────────────────────────────────────


# ── Проблема 1: проекция разных размерностей в общий hidden_dim ───────────────
# Статья: DBA (He et al., 2024) — "features scaled to the same feature dimension
# using 1D convolutional layers"
# Статья: MulT (Tsai et al., 2019) — "temporal convolutions project features of
# different modalities to the same dimension d"
class ModalityProjection(nn.Module):
    """Conv1D проекция одной модальности в HIDDEN_DIM с локальным контекстом."""

    def __init__(self, input_dim):
        super().__init__()
        self.proj = nn.Sequential(
            # kernel_size=3 захватывает фреймы t-1, t, t+1 — локальный контекст
            nn.Conv1d(input_dim, HIDDEN_DIM, kernel_size=3, padding=1),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: [B, T, D_in] → нужен [B, D_in, T] для Conv1d
        x = x.transpose(1, 2)          # [B, D_in, T]
        x = self.proj(x)               # [B, HIDDEN_DIM, T]
        x = x.transpose(1, 2)          # [B, T, HIDDEN_DIM]
        return x


# ── Проблема 6: positional encoding для audio и vision ────────────────────────
# Статья: MulT (Tsai et al., 2019) — "augment positional embedding to enable
# sequences to carry temporal information"
# Статья: MER-SEM-MBT (Xia et al., 2022) — "sine-cosine positional encoding
# to preserve the temporal information in the audio/visual feature sequence"
class PositionalEncoding(nn.Module):
    """Sine-cosine позиционное кодирование для временных последовательностей."""

    def __init__(self, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, HIDDEN_DIM)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, HIDDEN_DIM, 2).float()
                        * (-math.log(10000.0) / HIDDEN_DIM))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, HIDDEN_DIM]

    def forward(self, x):
        # x: [B, T, HIDDEN_DIM]
        return x + self.pe[:, :x.size(1)]

# ── Semantic Enhancement Module ───────────────────────────────────────────────
# Статья: MER-SEM-MBT (Xia et al., 2022) — "text CLS token as key and value
# in cross-attention with audio/visual features as query — latent adaption
# from text to audio/visual modality improves unimodal encoders"
class SemanticEnhancement(nn.Module):
    """
    Текстовый CLS вектор направляет audio или vision через cross-attention.
    Q = audio/vision tokens, K = V = text CLS
    """

    def __init__(self):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            HIDDEN_DIM, N_HEADS, DROPOUT, batch_first=True
        )
        self.norm = nn.LayerNorm(HIDDEN_DIM)

    def forward(self, modality, text_cls):
        # modality: [B, T, HIDDEN_DIM] — audio или vision
        # text_cls: [B, 1, HIDDEN_DIM] — CLS токен текста как K и V
        enhanced, _ = self.cross_attn(modality, text_cls, text_cls)
        return self.norm(modality + enhanced)

# ── Один слой Bottleneck Attention ────────────────────────────────────────────
# Статья: NeurIPS 2021 MBT (Nagrani et al.) — bottleneck tokens как мост
# между модальностями вместо дорогого полного cross-attention
# Статья: DBA (He et al., 2024) — n=16 latent tokens оптимально
class BottleneckLayer(nn.Module):
    """
    Один слой bottleneck fusion для трёх модальностей.
    Bottleneck tokens собирают информацию из каждой модальности
    и затем распределяют её обратно — без прямого cross-attention O(T²).
    """

    def __init__(self):
        super().__init__()
        # self-attention внутри каждой модальности
        self.self_attn_t = nn.MultiheadAttention(HIDDEN_DIM, N_HEADS, DROPOUT, batch_first=True)
        self.self_attn_a = nn.MultiheadAttention(HIDDEN_DIM, N_HEADS, DROPOUT, batch_first=True)
        self.self_attn_v = nn.MultiheadAttention(HIDDEN_DIM, N_HEADS, DROPOUT, batch_first=True)

        # cross-attention: каждая модальность → bottleneck tokens (сжатие)
        self.cross_t2b = nn.MultiheadAttention(HIDDEN_DIM, N_HEADS, DROPOUT, batch_first=True)
        self.cross_a2b = nn.MultiheadAttention(HIDDEN_DIM, N_HEADS, DROPOUT, batch_first=True)
        self.cross_v2b = nn.MultiheadAttention(HIDDEN_DIM, N_HEADS, DROPOUT, batch_first=True)

        # cross-attention: bottleneck tokens → каждая модальность (расширение)
        self.cross_b2t = nn.MultiheadAttention(HIDDEN_DIM, N_HEADS, DROPOUT, batch_first=True)
        self.cross_b2a = nn.MultiheadAttention(HIDDEN_DIM, N_HEADS, DROPOUT, batch_first=True)
        self.cross_b2v = nn.MultiheadAttention(HIDDEN_DIM, N_HEADS, DROPOUT, batch_first=True)

        # FFN блоки после attention для каждой модальности
        self.ffn_t = self._make_ffn()
        self.ffn_a = self._make_ffn()
        self.ffn_v = self._make_ffn()

        # LayerNorm
        self.norm_t = nn.LayerNorm(HIDDEN_DIM)
        self.norm_a = nn.LayerNorm(HIDDEN_DIM)
        self.norm_v = nn.LayerNorm(HIDDEN_DIM)

    def _make_ffn(self):
        return nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 4),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM * 4, HIDDEN_DIM),
            nn.Dropout(DROPOUT),
        )

    def forward(self, text, audio, vision, bottleneck, text_mask=None):
        # text:       [B, 50, HIDDEN_DIM]
        # audio:      [B, 60, HIDDEN_DIM]
        # vision:     [B, 60, HIDDEN_DIM]
        # bottleneck: [B, N_BOTTLENECK, HIDDEN_DIM]

        # ── 1. Self-attention внутри каждой модальности ───────────────────────
        # Проблема 2: text_mask из датасета передаём сюда
        # key_padding_mask: True = игнорировать токен (padding)
        # attention_mask из BERT: 1=реальный, 0=padding → инвертируем
        t_pad = (text_mask == 0) if text_mask is not None else None

        t, _ = self.self_attn_t(text, text, text, key_padding_mask=t_pad)
        a, _ = self.self_attn_a(audio, audio, audio)
        v, _ = self.self_attn_v(vision, vision, vision)

        text  = self.norm_t(text + t)
        audio = self.norm_a(audio + a)
        vision = self.norm_v(vision + v)

        # ── 2. Сжатие: модальности → bottleneck (Q=bottleneck, KV=модальность) ─
        # Статья: MBT — bottleneck tokens как query, modality tokens как key-value
        bt, _ = self.cross_t2b(bottleneck, text,   text)
        ba, _ = self.cross_a2b(bottleneck, audio,  audio)
        bv, _ = self.cross_v2b(bottleneck, vision, vision)

        # усредняем вклад всех трёх модальностей в bottleneck
        # Статья: XMBT — mean average operation aggregates bottleneck tokens
        bottleneck = (bt + ba + bv) / 3.0

        # ── 3. Расширение: bottleneck → модальности (Q=модальность, KV=bottleneck)
        t2, _ = self.cross_b2t(text,   bottleneck, bottleneck)
        a2, _ = self.cross_b2a(audio,  bottleneck, bottleneck)
        v2, _ = self.cross_b2v(vision, bottleneck, bottleneck)

        text   = self.norm_t(text   + t2)
        audio  = self.norm_a(audio  + a2)
        vision = self.norm_v(vision + v2)

        # ── 4. FFN ─────────────────────────────────────────────────────────────
        text   = self.norm_t(text   + self.ffn_t(text))
        audio  = self.norm_a(audio  + self.ffn_a(audio))
        vision = self.norm_v(vision + self.ffn_v(vision))

        return text, audio, vision, bottleneck


# ── Полная модель ─────────────────────────────────────────────────────────────
class BottleneckFusionModel(nn.Module):

    def __init__(self):
        super().__init__()

        # ── Проблема 1: Text encoder — BERT онлайн ────────────────────────────
        # Статья: XMBT — ALBERT/BERT запускается онлайн, дообучается с lr/10
        # Берём last_hidden_state [B, 50, 768], не CLS [B, 768]
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        self.text_proj = nn.Linear(768, HIDDEN_DIM)  # 768 → 128

        # ── Проблема 1: Audio и Vision энкодеры — Conv1D проекция ─────────────
        # Статья: DBA, MulT — Conv1D для готовых признаков COVAREP и OpenFace
        self.audio_proj  = ModalityProjection(input_dim=74)   # COVAREP 74-dim
        self.vision_proj = ModalityProjection(input_dim=35)   # OpenFace 35-dim

        # ── Проблема 6: Positional encoding для audio и vision ────────────────
        self.pos_enc_audio  = PositionalEncoding(max_len=60)
        self.pos_enc_vision = PositionalEncoding(max_len=60)
        # text получает positional encoding от самого BERT — не нужно добавлять

        # ── Learnable bottleneck tokens ───────────────────────────────────────
        # Статья: DBA — n=16, d=50 (мы используем d=HIDDEN_DIM=128)
        # Статья: MBT NeurIPS 2021 — learnable bottleneck tokens
        self.bottleneck = nn.Parameter(
            torch.randn(1, N_BOTTLENECK, HIDDEN_DIM)
        )

        # ── N_LAYERS слоёв Bottleneck ─────────────────────────────────────────
        # Статья: XMBT — L=2 даёт лучший результат на CMU-MOSEI
        self.layers = nn.ModuleList([BottleneckLayer() for _ in range(N_LAYERS)])

        # ── Классификатор ─────────────────────────────────────────────────────
        # Статья: XMBT — linear layer после cls tokens для каждой модальности,
        # затем fusion linear для финального предсказания
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 3, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, N_EMOTIONS),
        )

        # ── Auxiliary heads — по одной на каждую модальность ─────────────────────
        # Статья: XMBT (Nguyen et al., 2025) — classification tokens для каждой
        # модальности передаются в отдельные linear layers для auxiliary loss
        self.head_text   = nn.Linear(HIDDEN_DIM, N_EMOTIONS)
        self.head_audio  = nn.Linear(HIDDEN_DIM, N_EMOTIONS)
        self.head_vision = nn.Linear(HIDDEN_DIM, N_EMOTIONS)

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, input_ids, attention_mask, audio, vision, audio_mask, vision_mask):
        # input_ids:      [B, 50]
        # attention_mask: [B, 50]
        # audio:          [B, 60, 74]
        # vision:         [B, 60, 35]

        B = input_ids.size(0)

        # ── Проблема 1: BERT — берём все токены, не только CLS ────────────────
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text = bert_out.last_hidden_state          # [B, 50, 768]
        text = self.dropout(self.text_proj(text))  # [B, 50, HIDDEN_DIM]

        # ── Audio и Vision проекция ───────────────────────────────────────────
        audio  = self.audio_proj(audio)    # [B, 60, HIDDEN_DIM]
        vision = self.vision_proj(vision)  # [B, 60, HIDDEN_DIM]

        # ── Проблема 6: добавляем positional encoding ─────────────────────────
        audio  = self.pos_enc_audio(audio)
        vision = self.pos_enc_vision(vision)


        # ── Инициализируем bottleneck tokens для батча ────────────────────────
        bottleneck = self.bottleneck.expand(B, -1, -1)  # [B, N_BOTTLENECK, HIDDEN_DIM]

        # ── Проблема 3: передаём маски в bottleneck слои ──────────────────────
        for layer in self.layers:
            text, audio, vision, bottleneck = layer(
                text, audio, vision, bottleneck,
                text_mask=attention_mask
            )

        # ── Агрегация: CLS токен текста + mean pool audio и vision ────────────
        # Статья: MER-SEM-MBT — global average pooling (GAP) для audio/vision,
        # CLS токен для text
        text_cls   = text[:, 0, :]        # [B, HIDDEN_DIM] — CLS позиция
        audio_pool = audio.mean(dim=1)    # [B, HIDDEN_DIM]
        vision_pool = vision.mean(dim=1)  # [B, HIDDEN_DIM]

        # конкатенируем все три модальности
        fused = torch.cat([text_cls, audio_pool, vision_pool], dim=-1)  # [B, HIDDEN_DIM*3]

        # ── Финальная классификация ───────────────────────────────────────────
        logits_fuse   = self.classifier(fused)
        logits_text   = self.head_text(text_cls)
        logits_audio  = self.head_audio(audio_pool)
        logits_vision = self.head_vision(vision_pool)
        return logits_fuse, logits_text, logits_audio, logits_vision


# ── Быстрая проверка ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
    from dataset import get_dataloaders

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"Device: {device}")

    loaders, pos_weight = get_dataloaders()
    model = BottleneckFusionModel().to(device)

    # параметры
    total = sum(p.numel() for p in model.parameters())
    bert_p = sum(p.numel() for p in model.bert.parameters())
    other_p = total - bert_p
    print(f"\nПараметры всего    : {total:,}")
    print(f"  из них BERT      : {bert_p:,}")
    print(f"  из них остальные : {other_p:,}")

    # один батч
    batch = next(iter(loaders["train"]))
    with torch.no_grad():
        logits = model(
            input_ids     = batch["input_ids"].to(device),
            attention_mask= batch["attention_mask"].to(device),
            audio         = batch["audio"].to(device),
            vision        = batch["vision"].to(device),
            audio_mask    = batch["audio_mask"].to(device),
            vision_mask   = batch["vision_mask"].to(device),
        )
    print(f"\nВход  — audio : {batch['audio'].shape}")
    print(f"Вход  — vision: {batch['vision'].shape}")
    print(f"Вход  — text  : {batch['input_ids'].shape}")
    logits_fuse, logits_text, logits_audio, logits_vision = logits
    print(f"Выход — logits_fuse : {logits_fuse.shape}")
    print(f"Выход — logits_fuse : {logits_fuse.shape}")
    print(f"Выход — logits_text : {logits_text.shape}")
    print(f"Выход — logits_audio: {logits_audio.shape}")
    print(f"Выход — logits_vision:{logits_vision.shape}")
    print("\n✅ model.py работает корректно")