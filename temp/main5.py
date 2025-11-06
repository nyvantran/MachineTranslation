import torch.utils.checkpoint
from main1 import *
from main2 import *
from main3 import *
from main4 import *


# Encoder Layer
class OptimizedEncoderLayer(nn.Module):
    """
    Encoder layer với Pre-LayerNorm (hiệu quả hơn Post-LN)
    """

    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()

        # Pre-LayerNorm architecture
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = OptimizedMultiHeadAttention(d_model, num_heads, dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = OptimizedFeedforward(d_model, d_ff, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        # Pre-LN: Norm -> Attention -> Residual
        x = x + self.dropout(self.self_attn(self.norm1(x)))

        # Pre-LN: Norm -> FFN -> Residual
        x = x + self.dropout(self.ffn(self.norm2(x)))

        return x


class OptimizedDecoderLayer(nn.Module):
    """
    Decoder layer với Pre-LayerNorm và optimized cross-attention
    """

    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()

        # Self-attention
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = OptimizedMultiHeadAttention(
            d_model, num_heads, dropout, atmask=True
        )

        # Cross-attention
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = FusedCrossAttention(d_model, num_heads, dropout)

        # Feedforward
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = OptimizedFeedforward(d_model, d_ff, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Self-attention with causal mask
        x = x + self.dropout(self.self_attn(self.norm1(x)))

        # Cross-attention
        x = x + self.dropout(self.cross_attn(
            self.norm2(x), memory,
            key_padding_mask=memory_key_padding_mask
        ))

        # Feedforward
        x = x + self.dropout(self.ffn(self.norm3(x)))

        return x


class OptimizedTransformer(nn.Module):
    """
    Optimized Transformer với:
    - Pre-LayerNorm
    - Optimized components
    - Proper masking
    - Gradient checkpointing support
    - Mixed precision support
    """

    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 d_model=512,
                 num_heads=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 d_ff=2048,
                 dropout=0.1,
                 max_seq_len=5000,
                 pad_idx=0,
                 use_gradient_checkpointing=False):
        super().__init__()

        self.d_model = d_model
        self.pad_idx = pad_idx
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Embeddings
        self.src_embed = OptimizedEmbedding(src_vocab_size, d_model, max_seq_len, dropout)
        self.tgt_embed = OptimizedEmbedding(tgt_vocab_size, d_model, max_seq_len, dropout)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            OptimizedEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # Decoder
        self.decoder_layers = nn.ModuleList([
            OptimizedDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)

        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

        # Tie weights (share embedding and output weights)
        self.output_projection.weight = self.tgt_embed.token_emb.weight

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask for decoder"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def make_src_mask(self, src):
        """Create source padding mask"""
        # src: (batch_size, src_len)
        src_mask = (src == self.pad_idx)
        return src_mask

    def make_tgt_mask(self, tgt):
        """Create target padding mask"""
        # tgt: (batch_size, tgt_len)
        tgt_mask = (tgt == self.pad_idx)
        return tgt_mask

    def encode(self, src, src_mask=None):
        """Encode source sequence"""
        # Embedding
        x = self.src_embed(src)  # (batch_size, src_len, d_model)

        # Encoder layers
        for layer in self.encoder_layers:
            if self.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, src_mask, None
                )
            else:
                x = layer(x, src_key_padding_mask=src_mask)

        x = self.encoder_norm(x)
        return x

    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """Decode target sequence"""
        # Embedding
        x = self.tgt_embed(tgt)  # (batch_size, tgt_len, d_model)

        # Decoder layers
        for layer in self.decoder_layers:
            if self.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, memory, None, None, tgt_mask, memory_mask
                )
            else:
                x = layer(
                    x, memory,
                    tgt_key_padding_mask=tgt_mask,
                    memory_key_padding_mask=memory_mask
                )

        x = self.decoder_norm(x)
        return x

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            src: (batch_size, src_len)
            tgt: (batch_size, tgt_len)
        Returns:
            (batch_size, tgt_len, tgt_vocab_size)
        """
        # Create masks if not provided
        if src_mask is None:
            src_mask = self.make_src_mask(src)
        if tgt_mask is None:
            tgt_mask = self.make_tgt_mask(tgt)

        # Encode
        memory = self.encode(src, src_mask)

        # Decode
        output = self.decode(tgt, memory, tgt_mask, src_mask)

        # Project to vocabulary
        logits = self.output_projection(output)

        return logits

    @torch.no_grad()
    def generate(self, src, max_len=50, start_token=1, end_token=2,
                 temperature=1.0, top_k=None, top_p=None):
        """
        Greedy decoding with optional sampling
        """
        self.eval()
        device = src.device

        # Encode source
        if src.dim() == 1:
            src = src.unsqueeze(0)

        src_mask = self.make_src_mask(src)
        memory = self.encode(src, src_mask)

        # Start with start token
        tgt = torch.tensor([[start_token]], device=device)

        for _ in range(max_len):
            # Decode
            tgt_mask = self.make_tgt_mask(tgt)
            output = self.decode(tgt, memory, tgt_mask, src_mask)

            # Get logits for last token
            logits = self.output_projection(output[:, -1, :])  # (1, vocab_size)

            # Apply temperature
            logits = logits / temperature

            # Top-k sampling
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Top-p (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            tgt = torch.cat([tgt, next_token], dim=1)

            # Check for end token
            if next_token.item() == end_token:
                break

        return tgt.squeeze(0)

    @torch.no_grad()
    def beam_search(self, src, beam_size=5, max_len=50,
                    start_token=1, end_token=2, length_penalty=1.0):
        """
        Beam search decoding
        """
        self.eval()
        device = src.device

        if src.dim() == 1:
            src = src.unsqueeze(0)

        src_mask = self.make_src_mask(src)
        memory = self.encode(src, src_mask)

        # Initialize beam
        # sequences: (beam_size, seq_len)
        # scores: (beam_size,)
        sequences = torch.tensor([[start_token]], device=device)
        scores = torch.zeros(1, device=device)

        for step in range(max_len):
            # Expand sequences
            batch_size = sequences.size(0)

            # Decode all sequences in beam
            tgt_mask = self.make_tgt_mask(sequences)
            memory_expanded = memory.expand(batch_size, -1, -1)
            src_mask_expanded = src_mask.expand(batch_size, -1) if src_mask is not None else None

            output = self.decode(sequences, memory_expanded, tgt_mask, src_mask_expanded)
            logits = self.output_projection(output[:, -1, :])  # (beam_size, vocab_size)

            # Compute log probabilities
            log_probs = F.log_softmax(logits, dim=-1)

            # Add to existing scores
            # scores: (beam_size,) -> (beam_size, 1)
            # log_probs: (beam_size, vocab_size)
            scores_expanded = scores.unsqueeze(1) + log_probs  # (beam_size, vocab_size)

            # Flatten and get top beam_size
            scores_flat = scores_expanded.view(-1)  # (beam_size * vocab_size,)

            if step == 0:
                # First step, all beams are the same
                top_scores, top_indices = scores_flat[:logits.size(-1)].topk(beam_size)
            else:
                top_scores, top_indices = scores_flat.topk(beam_size)

            # Convert flat indices to (beam_idx, token_idx)
            beam_indices = top_indices // logits.size(-1)
            token_indices = top_indices % logits.size(-1)

            # Update sequences
            sequences = torch.cat([
                sequences[beam_indices],
                token_indices.unsqueeze(1)
            ], dim=1)

            scores = top_scores

            # Check if all beams have generated end token
            if (token_indices == end_token).all():
                break

        # Apply length penalty
        lengths = (sequences != end_token).sum(dim=1).float()
        scores = scores / (lengths ** length_penalty)

        # Return best sequence
        best_idx = scores.argmax()
        return sequences[best_idx]


class EfficientTransformer(nn.Module):
    """
    Efficient Transformer với shared weights và các tối ưu khác
    """

    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 d_model=512,
                 num_heads=8,
                 num_layers=6,
                 d_ff=2048,
                 dropout=0.1,
                 share_encoder_decoder=False,
                 share_all_embeddings=False):
        super().__init__()

        self.d_model = d_model
        self.share_all_embeddings = share_all_embeddings

        # Embeddings
        self.src_embed = OptimizedEmbedding(src_vocab_size, d_model, dropout=dropout)

        if share_all_embeddings and src_vocab_size == tgt_vocab_size:
            self.tgt_embed = self.src_embed
        else:
            self.tgt_embed = OptimizedEmbedding(tgt_vocab_size, d_model, dropout=dropout)

        # Encoder layers
        encoder_layer = OptimizedEncoderLayer(d_model, num_heads, d_ff, dropout)

        if share_encoder_decoder:
            # Share same layer across all positions (ALBERT-style)
            self.encoder_layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        else:
            self.encoder_layers = nn.ModuleList([
                OptimizedEncoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ])

        # Decoder layers
        decoder_layer = OptimizedDecoderLayer(d_model, num_heads, d_ff, dropout)

        if share_encoder_decoder:
            self.decoder_layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        else:
            self.decoder_layers = nn.ModuleList([
                OptimizedDecoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ])

        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)

        # Output
        self.output_projection = nn.Linear(d_model, tgt_vocab_size, bias=False)
        self.output_projection.weight = self.tgt_embed.token_emb.weight

    def forward(self, src, tgt):
        # Encode
        x = self.src_embed(src)
        for layer in self.encoder_layers:
            x = layer(x)
        memory = self.encoder_norm(x)

        # Decode
        x = self.tgt_embed(tgt)
        for layer in self.decoder_layers:
            x = layer(x, memory)
        x = self.decoder_norm(x)

        # Project
        logits = self.output_projection(x)
        return logits


# Original implementation (simplified)
class OriginalEncoder(nn.Module):
    def __init__(self, dmodel=512, num_heads=4):
        super().__init__()
        self.self_attn = OriginalMultiHeadAttention(dmodel, num_heads)
        self.ffn = OriginalFeedforward(dmodel)
        self.norm1 = nn.LayerNorm(dmodel)
        self.norm2 = nn.LayerNorm(dmodel)

    def forward(self, x):
        attn = self.self_attn(x)
        x = self.norm1(x + attn)
        ffn = self.ffn(x)
        x = self.norm2(x + ffn)
        return x


class OriginalDecoder(nn.Module):
    def __init__(self, dmodel=512, num_heads=4):
        super().__init__()
        self.self_attn = OriginalMultiHeadAttention(dmodel, num_heads, atmask=True)
        self.cross_attn = OriginalCrossMultiHeadAttention(dmodel, num_heads)
        self.ffn = OriginalFeedforward(dmodel)
        self.norm1 = nn.LayerNorm(dmodel)
        self.norm2 = nn.LayerNorm(dmodel)
        self.norm3 = nn.LayerNorm(dmodel)

    def forward(self, x, memory):
        attn = self.self_attn(x)
        x = self.norm1(x + attn)
        cross = self.cross_attn(x, memory)
        x = self.norm2(x + cross)
        ffn = self.ffn(x)
        x = self.norm3(x + ffn)
        return x


class OriginalPredictionHead(nn.Module):
    def __init__(self, dmodel=512, output_dim=10000):
        super().__init__()
        self.linear = nn.Linear(dmodel, output_dim)

    def forward(self, x):
        return self.linear(x)


class OriginalTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim=512, num_heads=4, num_layers=6):
        super().__init__()

        self.src_embed = OriginalEmbedding(input_dim, emb_dim)
        self.tgt_embed = OriginalEmbedding(output_dim, emb_dim)

        self.encoder_layers = nn.ModuleList([
            OriginalEncoder(emb_dim, num_heads) for _ in range(num_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            OriginalDecoder(emb_dim, num_heads) for _ in range(num_layers)
        ])

        self.prediction_head = OriginalPredictionHead(emb_dim, output_dim)

    def forward(self, src, tgt):
        src_emb = self.src_embed(src)
        tgt_emb = self.tgt_embed(tgt)

        x = src_emb
        for layer in self.encoder_layers:
            x = layer(x)

        memory = x
        x = tgt_emb
        for layer in self.decoder_layers:
            x = layer(x, memory)

        output = self.prediction_head(x)
        return output


class PyTorchNativeTransformer(nn.Module):
    """Wrapper cho nn.Transformer của PyTorch"""

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512,
                 num_heads=8, num_layers=6, d_ff=2048, dropout=0.1):
        super().__init__()

        self.d_model = d_model

        # Embeddings
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )

        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        # Embed
        src_emb = self.src_embed(src) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_embed(tgt) * math.sqrt(self.d_model)

        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)

        # Create causal mask
        tgt_len = tgt.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_len).to(tgt.device)

        # Transform
        output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)

        # Project
        logits = self.output_projection(output)
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_forward(model, src, tgt, num_iterations=100, warmup=10):
    """Benchmark forward pass"""
    device = next(model.parameters()).device
    src = src.to(device)
    tgt = tgt.to(device)
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(src, tgt)

    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            for _ in range(num_iterations):
                output = model(src, tgt)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        avg_time = elapsed_time / num_iterations
        peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
    else:
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                output = model(src, tgt)
        elapsed_time = (time.time() - start_time) * 1000
        avg_time = elapsed_time / num_iterations
        peak_memory = 0

    return avg_time, peak_memory


def benchmark_training(model, src, tgt, criterion, optimizer, num_iterations=100):
    """Benchmark training step"""
    device = next(model.parameters()).device
    src = src.to(device)
    tgt = tgt.to(device)
    model.train()

    # Warmup
    for _ in range(10):
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])
        loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()

    if device.type == 'cuda':
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_iterations):
            optimizer.zero_grad()
            output = model(src, tgt[:, :-1])
            loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
        end_event.record()

        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        avg_time = elapsed_time / num_iterations
    else:
        start_time = time.time()
        for _ in range(num_iterations):
            optimizer.zero_grad()
            output = model(src, tgt[:, :-1])
            loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
        elapsed_time = (time.time() - start_time) * 1000
        avg_time = elapsed_time / num_iterations

    return avg_time


def benchmark_generation(model, src, max_len=50, num_iterations=10):
    """Benchmark generation/inference"""
    device = next(model.parameters()).device
    src = src.to(device)
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(2):
            if isinstance(model, OptimizedTransformer):
                _ = model.generate(src[0], max_len=max_len)
            elif hasattr(model, 'greedy_decode'):
                _ = model.greedy_decode(src[0], max_len=max_len)

    if device.type == 'cuda':
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            for i in range(num_iterations):
                if isinstance(model, OptimizedTransformer):
                    output = model.generate(src[i % src.size(0)], max_len=max_len)
                elif hasattr(model, 'greedy_decode'):
                    output = model.greedy_decode(src[i % src.size(0)], max_len=max_len)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        avg_time = elapsed_time / num_iterations
    else:
        start_time = time.time()
        with torch.no_grad():
            for i in range(num_iterations):
                if isinstance(model, OptimizedTransformer):
                    output = model.generate(src[i % src.size(0)], max_len=max_len)
        elapsed_time = (time.time() - start_time) * 1000
        avg_time = elapsed_time / num_iterations

    return avg_time


def run_comprehensive_benchmark():
    """Comprehensive benchmark"""
    configs = [
        # (batch_size, src_len, tgt_len, src_vocab, tgt_vocab, d_model, num_heads, num_layers)
        (32, 32, 32, 10000, 10000, 512, 8, 6),
        (16, 64, 64, 10000, 10000, 512, 8, 6),
        (8, 128, 128, 10000, 10000, 512, 8, 6),
        (32, 32, 32, 30000, 30000, 768, 12, 6),  # BERT-base size
    ]

    print("=" * 150)
    print(
        f"{'Config':<30} {'Model':<30} {'Params (M)':<12} {'FWD (ms)':<12} {'Train (ms)':<12} {'Memory (MB)':<12} {'Speedup':<10}")
    print("=" * 150)

    for batch_size, src_len, tgt_len, src_vocab, tgt_vocab, d_model, num_heads, num_layers in configs:
        config_str = f"B{batch_size}_S{src_len}_T{tgt_len}_D{d_model}"

        # Create sample data
        src = torch.randint(1, src_vocab, (batch_size, src_len))
        tgt = torch.randint(1, tgt_vocab, (batch_size, tgt_len))

        models = {
            'Original': OriginalTransformer(src_vocab, tgt_vocab, d_model, num_heads, num_layers),
            'Optimized': OptimizedTransformer(src_vocab, tgt_vocab, d_model, num_heads,
                                              num_layers, num_layers, d_model * 4, 0.1),
            'Efficient (Shared)': EfficientTransformer(src_vocab, tgt_vocab, d_model, num_heads,
                                                       num_layers, d_model * 4, 0.1,
                                                       share_encoder_decoder=False),
            'PyTorch Native': PyTorchNativeTransformer(src_vocab, tgt_vocab, d_model,
                                                       num_heads, num_layers, d_model * 4, 0.1),
        }

        results = {}

        for name, model in models.items():
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = model.to(device)

                params = count_parameters(model) / 1e6

                # Forward pass
                fwd_time, memory = benchmark_forward(model, src, tgt, num_iterations=50)

                # Training
                criterion = nn.CrossEntropyLoss(ignore_index=0)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
                train_time = benchmark_training(model, src, tgt, criterion, optimizer, num_iterations=20)

                results[name] = {
                    'params': params,
                    'fwd': fwd_time,
                    'train': train_time,
                    'memory': memory
                }

                # Clean up
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"{config_str:<30} {name:<30} ERROR: {str(e)}")
                results[name] = {
                    'params': 0,
                    'fwd': float('inf'),
                    'train': float('inf'),
                    'memory': 0
                }

        # Calculate speedup vs Original
        if 'Original' in results:
            baseline_fwd = results['Original']['fwd']
        else:
            baseline_fwd = 1.0

        for name, res in results.items():
            speedup = baseline_fwd / res['fwd'] if res['fwd'] > 0 else 0
            print(f"{config_str:<30} {name:<30} {res['params']:>8.2f} M   "
                  f"{res['fwd']:>8.2f} ms   {res['train']:>8.2f} ms   "
                  f"{res['memory']:>8.2f} MB   {speedup:>6.2f}x")

        print("-" * 150)


def test_correctness():
    """Test output correctness"""
    batch_size, src_len, tgt_len = 4, 16, 16
    src_vocab, tgt_vocab = 1000, 1000
    d_model, num_heads, num_layers = 512, 8, 2

    torch.manual_seed(42)
    src = torch.randint(1, src_vocab, (batch_size, src_len))
    tgt = torch.randint(1, tgt_vocab, (batch_size, tgt_len))

    models = {
        'Original': OriginalTransformer(src_vocab, tgt_vocab, d_model, num_heads, num_layers),
        'Optimized': OptimizedTransformer(src_vocab, tgt_vocab, d_model, num_heads,
                                          num_layers, num_layers, d_model * 4, 0.0),
    }

    print("\n" + "=" * 80)
    print("CORRECTNESS TEST")
    print("=" * 80)

    outputs = {}
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            outputs[name] = model(src, tgt)

        print(f"\n{name}:")
        print(f"  Output shape: {outputs[name].shape}")
        print(f"  Output mean:  {outputs[name].mean().item():.6f}")
        print(f"  Output std:   {outputs[name].std().item():.6f}")
        print(f"  Output range: [{outputs[name].min().item():.2f}, {outputs[name].max().item():.2f}]")


def test_generation():
    """Test generation capabilities"""
    src_vocab, tgt_vocab = 1000, 1000
    d_model, num_heads, num_layers = 512, 8, 2

    print("\n" + "=" * 80)
    print("GENERATION TEST")
    print("=" * 80)

    # Create model
    model = OptimizedTransformer(
        src_vocab, tgt_vocab, d_model, num_heads,
        num_layers, num_layers, d_model * 4, 0.1
    )
    model.eval()

    # Source sentence
    src = torch.randint(1, src_vocab, (20,))

    print("\nGreedy Decoding:")
    start = time.time()
    output_greedy = model.generate(src, max_len=30, temperature=1.0)
    time_greedy = (time.time() - start) * 1000
    print(f"  Output length: {len(output_greedy)}")
    print(f"  Time: {time_greedy:.2f} ms")
    print(f"  Tokens: {output_greedy.tolist()[:10]}...")

    print("\nBeam Search (beam_size=5):")
    start = time.time()
    output_beam = model.beam_search(src, beam_size=5, max_len=30)
    time_beam = (time.time() - start) * 1000
    print(f"  Output length: {len(output_beam)}")
    print(f"  Time: {time_beam:.2f} ms")
    print(f"  Tokens: {output_beam.tolist()[:10]}...")

    print("\nSampling (temperature=0.8, top_p=0.9):")
    start = time.time()
    output_sample = model.generate(src, max_len=30, temperature=0.8, top_p=0.9)
    time_sample = (time.time() - start) * 1000
    print(f"  Output length: {len(output_sample)}")
    print(f"  Time: {time_sample:.2f} ms")
    print(f"  Tokens: {output_sample.tolist()[:10]}...")


def compare_architectures():
    """Compare different architectures"""
    batch_size, src_len, tgt_len = 32, 64, 64
    src_vocab, tgt_vocab = 10000, 10000
    d_model, num_heads, num_layers = 512, 8, 6

    src = torch.randint(1, src_vocab, (batch_size, src_len))
    tgt = torch.randint(1, tgt_vocab, (batch_size, tgt_len))

    print("\n" + "=" * 120)
    print("ARCHITECTURE VARIANTS COMPARISON")
    print("=" * 120)
    print(f"{'Architecture':<40} {'Params (M)':<15} {'FWD (ms)':<15} {'Memory (MB)':<15}")
    print("=" * 120)

    architectures = {
        'Standard (Post-LN)': OriginalTransformer(src_vocab, tgt_vocab, d_model, num_heads, num_layers),
        'Optimized (Pre-LN)': OptimizedTransformer(src_vocab, tgt_vocab, d_model, num_heads,
                                                   num_layers, num_layers, d_model * 4, 0.1),
        'Shared Weights': EfficientTransformer(src_vocab, tgt_vocab, d_model, num_heads,
                                               num_layers, d_model * 4, 0.1, share_encoder_decoder=True),
        'Gradient Checkpointing': OptimizedTransformer(src_vocab, tgt_vocab, d_model, num_heads,
                                                       num_layers, num_layers, d_model * 4, 0.1,
                                                       use_gradient_checkpointing=True),
        'PyTorch Native': PyTorchNativeTransformer(src_vocab, tgt_vocab, d_model, num_heads,
                                                   num_layers, d_model * 4, 0.1),
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for name, model in architectures.items():
        try:
            model = model.to(device)
            params = count_parameters(model) / 1e6
            fwd_time, memory = benchmark_forward(model, src, tgt, num_iterations=50)
            print(f"{name:<40} {params:>10.2f} M   {fwd_time:>10.2f} ms   {memory:>10.2f} MB")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"{name:<40} ERROR: {str(e)}")


def benchmark_batch_sizes():
    """Benchmark different batch sizes"""
    src_len, tgt_len = 64, 64
    src_vocab, tgt_vocab = 10000, 10000
    d_model, num_heads, num_layers = 512, 8, 6

    batch_sizes = [8, 16, 32, 64, 128]

    print("\n" + "=" * 100)
    print("BATCH SIZE SCALING")
    print("=" * 100)
    print(f"{'Batch Size':<15} {'Original (ms)':<18} {'Optimized (ms)':<18} {'Speedup':<10}")
    print("=" * 100)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for batch_size in batch_sizes:
        src = torch.randint(1, src_vocab, (batch_size, src_len))
        tgt = torch.randint(1, tgt_vocab, (batch_size, tgt_len))

        # Original
        model_orig = OriginalTransformer(src_vocab, tgt_vocab, d_model, num_heads, num_layers).to(device)
        time_orig, _ = benchmark_forward(model_orig, src, tgt, num_iterations=50)
        del model_orig

        # Optimized
        model_opt = OptimizedTransformer(src_vocab, tgt_vocab, d_model, num_heads,
                                         num_layers, num_layers, d_model * 4, 0.1).to(device)
        time_opt, _ = benchmark_forward(model_opt, src, tgt, num_iterations=50)
        del model_opt

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        speedup = time_orig / time_opt
        print(f"{batch_size:<15} {time_orig:>13.2f} ms   {time_opt:>13.2f} ms   {speedup:>6.2f}x")


def memory_profiling():
    """Detailed memory profiling"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory profiling")
        return

    batch_size, src_len, tgt_len = 32, 64, 64
    src_vocab, tgt_vocab = 10000, 10000
    d_model, num_heads, num_layers = 512, 8, 6

    src = torch.randint(1, src_vocab, (batch_size, src_len)).cuda()
    tgt = torch.randint(1, tgt_vocab, (batch_size, tgt_len)).cuda()

    models = {
        'Original': OriginalTransformer(src_vocab, tgt_vocab, d_model, num_heads, num_layers),
        'Optimized': OptimizedTransformer(src_vocab, tgt_vocab, d_model, num_heads,
                                          num_layers, num_layers, d_model * 4, 0.1),
        'Gradient Checkpoint': OptimizedTransformer(src_vocab, tgt_vocab, d_model, num_heads,
                                                    num_layers, num_layers, d_model * 4, 0.1,
                                                    use_gradient_checkpointing=True),
    }

    print("\n" + "=" * 120)
    print("MEMORY PROFILING")
    print("=" * 120)
    print(f"{'Model':<25} {'Params (MB)':<15} {'Forward (MB)':<15} {'Backward (MB)':<15} {'Total (MB)':<15}")
    print("=" * 120)

    for name, model in models.items():
        model = model.cuda()

        # Parameter memory
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2

        # Forward pass memory
        torch.cuda.reset_peak_memory_stats()
        model.eval()
        with torch.no_grad():
            output = model(src, tgt)
        torch.cuda.synchronize()
        forward_memory = torch.cuda.max_memory_allocated() / 1024 ** 2 - param_memory

        # Backward pass memory
        torch.cuda.reset_peak_memory_stats()
        model.train()
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])
        loss = output.sum()
        loss.backward()
        torch.cuda.synchronize()
        total_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
        backward_memory = total_memory - param_memory - forward_memory

        print(f"{name:<25} {param_memory:>10.2f} MB   {forward_memory:>10.2f} MB   "
              f"{backward_memory:>10.2f} MB   {total_memory:>10.2f} MB")

        del model, optimizer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print("=" * 80)
    print("TRANSFORMER MODEL OPTIMIZATION BENCHMARK")
    print("=" * 80)

    # Test correctness
    test_correctness()

    # Test generation
    test_generation()

    # Comprehensive benchmark
    run_comprehensive_benchmark()

    # Architecture comparison
    compare_architectures()

    # Batch size scaling
    benchmark_batch_sizes()

    # Memory profiling
    if torch.cuda.is_available():
        memory_profiling()
