# ======================================================================================
# IKRE
# ======================================================================================
class InstancePromptGenerator(nn.Module):
    def __init__(self, in_channels, dim, seq_len, patch_size, prompt_len, heads):
        super().__init__()
        assert patch_size % 2 == 0 and seq_len % patch_size == 0
        self.num_tokens = seq_len // patch_size
        self.embed_h = nn.Conv1d(in_channels, dim, kernel_size=patch_size // 2, stride=patch_size // 2)
        self.embed_l = nn.Conv1d(in_channels, dim, kernel_size=patch_size // 2, stride=patch_size // 2)
        self.norm_h, self.norm_l = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn_h = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.attn_l = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.conv_h = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.conv_l = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.gate = nn.Sequential(nn.Linear(2 * dim, dim // 2), nn.ReLU(inplace=True), nn.Linear(dim // 2, 1))
        self.B_H = nn.Parameter(torch.eye(dim) + 0.02 * torch.randn(dim, dim))
        self.B_L = nn.Parameter(torch.eye(dim) + 0.02 * torch.randn(dim, dim))
        self.out_conv = nn.Conv1d(2 * self.num_tokens, prompt_len, kernel_size=1)
        self.out_norm = nn.LayerNorm(dim)

    @staticmethod
    def haar_dwt(x):
        a, b = x[..., 0::2], x[..., 1::2]
        return (a + b) / (2 ** 0.5), (a - b) / (2 ** 0.5)  # 低频 cA, 高频 cD

    def forward(self, x):
        low, high = self.haar_dwt(x)
        H = self.norm_h(self.embed_h(high).transpose(1, 2))  # (B, T, h)
        L = self.norm_l(self.embed_l(low).transpose(1, 2))
        C_H = self.attn_h(H, L, L, need_weights=False)[0]
        C_L = self.attn_l(L, H, H, need_weights=False)[0]
        C_H = C_H + self.conv_h(C_H.transpose(1, 2)).transpose(1, 2)
        C_L = C_L + self.conv_l(C_L.transpose(1, 2)).transpose(1, 2)
        g = torch.sigmoid(self.gate(torch.cat([C_H.mean(1), C_L.mean(1)], dim=-1)))  # (B, 1)
        g = g.unsqueeze(-1)
        R_H = g * (C_H @ self.B_H)
        R_L = (1.0 - g) * (C_L @ self.B_L)
        prompt = self.out_conv(torch.cat([R_H, R_L], dim=1))  # (B, M, h)，式(3)
        return self.out_norm(prompt), g.squeeze(-1).squeeze(-1)


class IKRE(nn.Module):
    def __init__(self, num_classes, in_channels=1, num_domains=3, dim=128, depth=2, heads=4,
                 prompt_len=6, patch_size=64, seq_len=1024, dropout=0.1):
        super().__init__()
        self.stage = 1
        self.prompt_len = prompt_len
        self.num_patches = seq_len // patch_size
        self.patch_embed = nn.Conv1d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + self.num_patches, dim) * 0.02)
        self.domain_prompts = nn.Parameter(torch.randn(num_domains, prompt_len, dim) * 0.02)  # 域级提示
        self.prompt_generator = InstancePromptGenerator(in_channels, dim, seq_len, patch_size, prompt_len, heads)
        layer = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=2 * dim, dropout=dropout,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)
        self.feature_dim = dim

    def set_stage(self, stage):
        self.stage = stage
        self.domain_prompts.requires_grad_(stage == 1)

    def forward(self, x, domain=None):
        B = x.shape[0]
        if self.stage == 1:
            if domain is None:
                prompt = self.domain_prompts.mean(dim=0, keepdim=True).expand(B, -1, -1)
            else:
                prompt = self.domain_prompts[domain]
        else:
            prompt, _ = self.prompt_generator(x)
        tokens = self.patch_embed(x).transpose(1, 2)                      # (B, T, h)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed         # (B, 1+T, h)
        seq = torch.cat([tokens[:, :1], prompt, tokens[:, 1:]], dim=1)    # [CLS, 提示, patch tokens]
        feature = self.norm(self.encoder(seq)[:, 0])
        return self.classifier(feature), feature, prompt

