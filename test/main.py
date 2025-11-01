import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import math
import time
from typing import Optional, Tuple
import gc

# ==================== MODEL ARCHITECTURE ====================
class TransformerTranslation(nn.Module):
    def __init__(
            self,
            src_vocab_size: int,
            tgt_vocab_size: int,
            d_model: int = 512,
            nhead: int = 8,
            num_encoder_layers: int = 6,
            num_decoder_layers: int = 6,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            max_seq_length: int = 512
    ):
        super().__init__()
        self.d_model = d_model

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Output layer
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
            self,
            src: torch.Tensor,
            tgt: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None,
            tgt_mask: Optional[torch.Tensor] = None,
            src_key_padding_mask: Optional[torch.Tensor] = None,
            tgt_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        # Embeddings - test
        src_emb = self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))

        # Transformer
        output = self.transformer(
            src_emb, tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        return self.fc_out(output)

    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Tạo mask để decoder không nhìn thấy future tokens"""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ==================== DATASET ====================
class TranslationDataset(Dataset):
    def __init__(self, src_data, tgt_data, src_vocab, tgt_vocab):
        self.src_data = src_data
        self.tgt_data = tgt_data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.src_data[idx], dtype=torch.long),
            torch.tensor(self.tgt_data[idx], dtype=torch.long)
        )


def collate_fn(batch, pad_idx=0):
    """Dynamic padding - chỉ pad đến max length trong batch"""
    src_batch, tgt_batch = zip(*batch)

    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)

    return src_batch, tgt_batch


# ==================== TRAINING FUNCTION ====================
class Trainer:
    def __init__(
            self,
            model: nn.Module,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader,
            optimizer: optim.Optimizer,
            criterion: nn.Module,
            device: torch.device,
            pad_idx: int = 0,
            accumulation_steps: int = 2,  # Gradient accumulation
            max_grad_norm: float = 1.0,
            use_amp: bool = True,  # Mixed Precision
            checkpoint_dir: str = './checkpoints'
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.pad_idx = pad_idx
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        self.checkpoint_dir = checkpoint_dir

        # Mixed Precision Scaler
        self.scaler = GradScaler() if use_amp else None

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )

        # Best model tracking
        self.best_val_loss = float('inf')

        import os
        os.makedirs(checkpoint_dir, exist_ok=True)

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train một epoch"""
        self.model.train()
        total_loss = 0
        total_tokens = 0

        start_time = time.time()

        # Zero gradient ban đầu
        self.optimizer.zero_grad()

        for batch_idx, (src, tgt) in enumerate(self.train_dataloader):
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            # Tạo input và target cho decoder
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # Tạo masks
            tgt_mask = self.model.generate_square_subsequent_mask(
                tgt_input.size(1), self.device
            )
            src_padding_mask = (src == self.pad_idx)
            tgt_padding_mask = (tgt_input == self.pad_idx)

            # Mixed Precision Training
            with autocast(enabled=self.use_amp):
                output = self.model(
                    src, tgt_input,
                    tgt_mask=tgt_mask,
                    src_key_padding_mask=src_padding_mask,
                    tgt_key_padding_mask=tgt_padding_mask
                )

                # Reshape để tính loss
                output = output.reshape(-1, output.shape[-1])
                tgt_output = tgt_output.reshape(-1)

                loss = self.criterion(output, tgt_output)

                # Normalize loss cho gradient accumulation
                loss = loss / self.accumulation_steps

            # Backward pass với mixed precision
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights sau mỗi accumulation_steps
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.use_amp:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()

                self.optimizer.zero_grad()

            # Tracking
            num_tokens = (tgt_output != self.pad_idx).sum().item()
            total_loss += loss.item() * self.accumulation_steps * num_tokens
            total_tokens += num_tokens

            # Clear cache định kỳ
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            # Logging
            if batch_idx % 100 == 0:
                avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
                elapsed = time.time() - start_time
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(self.train_dataloader)} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Speed: {batch_idx / elapsed:.2f} batches/s | "
                      f"GPU Mem: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        perplexity = math.exp(min(avg_loss, 100))  # Avoid overflow

        return avg_loss, perplexity

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0

        for src, tgt in self.val_dataloader:
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            tgt_mask = self.model.generate_square_subsequent_mask(
                tgt_input.size(1), self.device
            )
            src_padding_mask = (src == self.pad_idx)
            tgt_padding_mask = (tgt_input == self.pad_idx)

            with autocast(enabled=self.use_amp):
                output = self.model(
                    src, tgt_input,
                    tgt_mask=tgt_mask,
                    src_key_padding_mask=src_padding_mask,
                    tgt_key_padding_mask=tgt_padding_mask
                )

                output = output.reshape(-1, output.shape[-1])
                tgt_output = tgt_output.reshape(-1)

                loss = self.criterion(output, tgt_output)

            num_tokens = (tgt_output != self.pad_idx).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        perplexity = math.exp(min(avg_loss, 100))

        return avg_loss, perplexity

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'scheduler_state_dict': self.scheduler.state_dict(),
        }

        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # Save latest checkpoint
        torch.save(checkpoint, f'{self.checkpoint_dir}/checkpoint_latest.pt')

        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, f'{self.checkpoint_dir}/checkpoint_best.pt')
            print(f"✓ Saved best model with val_loss: {val_loss:.4f}")

    def train(self, num_epochs: int, early_stopping_patience: int = 5):
        """Main training loop"""
        print("=" * 80)
        print("Starting Training...")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Batch Size: {self.train_dataloader.batch_size}")
        print(f"Gradient Accumulation Steps: {self.accumulation_steps}")
        print(f"Effective Batch Size: {self.train_dataloader.batch_size * self.accumulation_steps}")
        print("=" * 80)

        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()

            # Train
            train_loss, train_ppl = self.train_epoch(epoch)

            # Validate
            val_loss, val_ppl = self.validate()

            # Update learning rate
            self.scheduler.step(val_loss)

            epoch_time = time.time() - epoch_start_time

            print("-" * 80)
            print(f"Epoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
            print(f"  Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
            print(f"  Time: {epoch_time:.2f}s | LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            print("-" * 80)

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            self.save_checkpoint(epoch, val_loss, is_best)

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n✗ Early stopping triggered after {epoch} epochs")
                break

            # Clear cache
            torch.cuda.empty_cache()
            gc.collect()

        print("\n" + "=" * 80)
        print("Training Completed!")
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")
        print("=" * 80)


# ==================== USAGE EXAMPLE ====================
def main():
    # Hyperparameters
    BATCH_SIZE = 16
    GRADIENT_ACCUMULATION = 2  # Effective batch size = 32
    LEARNING_RATE = 5e-4
    NUM_EPOCHS = 50

    SRC_VOCAB_SIZE = 32000
    TGT_VOCAB_SIZE = 32000
    D_MODEL = 512
    NHEAD = 8
    NUM_LAYERS = 6
    DIM_FEEDFORWARD = 2048
    DROPOUT = 0.1
    MAX_SEQ_LEN = 256
    PAD_IDX = 0

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Model
    model = TransformerTranslation(
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_LAYERS,
        num_decoder_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_seq_length=MAX_SEQ_LEN
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Dummy data (thay bằng data thật của bạn)
    # Giả sử bạn đã có src_train, tgt_train, src_val, tgt_val đã được tokenize
    train_dataset = TranslationDataset([], [], None, None)  # Load your data here
    val_dataset = TranslationDataset([], [], None, None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, PAD_IDX),
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, PAD_IDX),
        num_workers=2,
        pin_memory=True
    )

    # Loss (ignore padding)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.01
    )

    # Trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        pad_idx=PAD_IDX,
        accumulation_steps=GRADIENT_ACCUMULATION,
        use_amp=True,
        checkpoint_dir='./checkpoints'
    )

    # Train
    trainer.train(num_epochs=NUM_EPOCHS, early_stopping_patience=5)


if __name__ == "__main__":
    main()
