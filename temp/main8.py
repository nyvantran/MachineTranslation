import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
import time
import os
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import logging
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from main7 import *
from main6 import *
from main5 import *

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TransformerTrainer:
    """
    Complete training framework cho OptimizedTransformer
    """

    def __init__(
            self,
            model: nn.Module,
            train_dataset,
            val_dataset,
            config: Dict,
            save_dir: str = './checkpoints',
            device: Optional[torch.device] = None
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Device
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        logger.info(f"Using device: {self.device}")
        logger.info(f"Model parameters: {self.count_parameters():,}")

        # DataLoaders
        self.train_loader = self._create_dataloader(train_dataset, shuffle=True)
        self.val_loader = self._create_dataloader(val_dataset, shuffle=False)

        # Loss function
        self.criterion = self._create_loss_function()

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision training
        self.use_amp = config.get('use_amp', True) and self.device.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_bleu = 0.0
        self.patience_counter = 0

        # History
        self.history = {
            'train_loss': [], 'train_acc': [], 'train_ppl': [],
            'val_loss': [], 'val_acc': [], 'val_ppl': [], 'val_bleu': [],
            'learning_rates': []
        }

        # Save config
        self._save_config()

    def _create_dataloader(self, dataset, shuffle: bool) -> DataLoader:
        """Create DataLoader with optimized settings"""
        return DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=shuffle,
            num_workers=self.config.get('num_workers', 4),
            collate_fn=optimized_collate_fn,
            pin_memory=True if self.device.type == 'cuda' else False,
            prefetch_factor=2 if self.config.get('num_workers', 4) > 0 else None
        )

    def _create_loss_function(self):
        """Create optimized loss function"""
        loss_type = self.config.get('loss_type', 'optimized')
        vocab_size = self.train_dataset.get_vocab_size('vi')
        pad_idx = self.config.get('pad_idx', 0)
        label_smoothing = self.config.get('label_smoothing', 0.1)

        if loss_type == 'optimized':
            criterion = OptimizedCrossEntropyLoss(
                vocab_size=vocab_size,
                pad_idx=pad_idx,
                label_smoothing=label_smoothing
            )
        elif loss_type == 'translation':
            criterion = TranslationLoss(
                vocab_size=vocab_size,
                pad_idx=pad_idx,
                label_smoothing=label_smoothing,
                length_penalty=self.config.get('length_penalty', 0.0)
            )
        elif loss_type == 'focal':
            criterion = FocalLoss(
                vocab_size=vocab_size,
                pad_idx=pad_idx,
                gamma=self.config.get('focal_gamma', 2.0)
            )
        else:
            criterion = nn.CrossEntropyLoss(
                ignore_index=pad_idx,
                label_smoothing=label_smoothing
            )

        return criterion.to(self.device)

    def _create_optimizer(self):
        """Create optimizer with weight decay"""
        optimizer_type = self.config.get('optimizer', 'adamw')
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 0.01)

        # Separate parameters for weight decay
        no_decay = ['bias', 'LayerNorm.weight', 'norm']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        if optimizer_type == 'adam':
            optimizer = Adam(
                optimizer_grouped_parameters,
                lr=lr,
                betas=(0.9, 0.98),
                eps=1e-9
            )
        elif optimizer_type == 'adamw':
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=lr,
                betas=(0.9, 0.98),
                eps=1e-9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        return optimizer

    def _create_scheduler(self):
        """Create learning rate scheduler"""
        scheduler_type = self.config.get('scheduler', 'warmup')

        if scheduler_type == 'warmup':
            warmup_steps = self.config.get('warmup_steps', 4000)

            def lr_lambda(step):
                step = max(step, 1)
                d_model = self.config.get('d_model', 512)
                return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)

            scheduler = LambdaLR(self.optimizer, lr_lambda)

        elif scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('num_epochs', 100),
                eta_min=1e-6
            )

        elif scheduler_type == 'plateau':
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            scheduler = None

        return scheduler

    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _save_config(self):
        """Save configuration"""
        config_path = self.save_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Config saved to {config_path}")

    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()

        metrics = defaultdict(float)
        num_batches = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config['num_epochs']}"
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)

            # Forward pass
            loss, batch_metrics = self._train_step(src, tgt)

            # Accumulate metrics
            metrics['loss'] += loss
            for key, value in batch_metrics.items():
                metrics[key] += value
            num_batches += 1

            # Update progress bar
            if batch_idx % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'loss': f"{loss:.4f}",
                    'acc': f"{batch_metrics.get('accuracy', 0):.4f}",
                    'ppl': f"{batch_metrics.get('perplexity', 0):.2f}",
                    'lr': f"{current_lr:.2e}"
                })

            # Gradient accumulation
            if (batch_idx + 1) % self.config.get('gradient_accumulation_steps', 1) == 0:
                # Clip gradients
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get('max_grad_norm', 1.0)
                )

                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # Scheduler step (for warmup scheduler)
                if self.scheduler and self.config.get('scheduler') == 'warmup':
                    self.scheduler.step()

                self.optimizer.zero_grad()
                self.global_step += 1

        # Average metrics
        for key in metrics:
            metrics[key] /= num_batches

        return dict(metrics)

    def _train_step(self, src: torch.Tensor, tgt: torch.Tensor) -> Tuple[float, Dict]:
        """Single training step"""
        # Target input (shifted right) and output
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            output = self.model(src, tgt_input)

            # Compute loss
            if isinstance(self.criterion, (OptimizedCrossEntropyLoss, TranslationLoss)):
                loss, batch_metrics = self.criterion(output, tgt_output, return_metrics=True)
            else:
                batch_size, seq_len, vocab_size = output.size()
                loss = self.criterion(
                    output.reshape(-1, vocab_size),
                    tgt_output.reshape(-1)
                )
                batch_metrics = {}

        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.item(), batch_metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()

        metrics = defaultdict(float)
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validating"):
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                output = self.model(src, tgt_input)

                # Compute loss
                if isinstance(self.criterion, (OptimizedCrossEntropyLoss, TranslationLoss)):
                    loss, batch_metrics = self.criterion(output, tgt_output, return_metrics=True)
                else:
                    batch_size, seq_len, vocab_size = output.size()
                    loss = self.criterion(
                        output.reshape(-1, vocab_size),
                        tgt_output.reshape(-1)
                    )
                    batch_metrics = {}

            metrics['loss'] += loss.item()
            for key, value in batch_metrics.items():
                metrics[key] += value
            num_batches += 1

        # Average metrics
        for key in metrics:
            metrics[key] /= num_batches

        return dict(metrics)

    @torch.no_grad()
    def compute_bleu(self, num_samples: int = 100) -> float:
        """Compute BLEU score on validation set"""
        try:
            from sacrebleu import corpus_bleu
        except ImportError:
            logger.warning("sacrebleu not installed. Skipping BLEU computation.")
            return 0.0

        self.model.eval()

        references = []
        hypotheses = []

        # Sample from validation set
        indices = np.random.choice(len(self.val_dataset), min(num_samples, len(self.val_dataset)), replace=False)

        for idx in tqdm(indices, desc="Computing BLEU"):
            src, tgt = self.val_dataset[idx]
            src = src.unsqueeze(0).to(self.device)

            # Generate translation
            if hasattr(self.model, 'generate'):
                output = self.model.generate(
                    src[0],
                    max_len=self.config.get('max_gen_len', 75),
                    start_token=self.config.get('start_token', 1),
                    end_token=self.config.get('end_token', 2)
                )
            else:
                # Fallback to greedy decoding
                output = self._greedy_decode(src[0])

            # Decode
            hypothesis = self.val_dataset.decode(output, language='vi')
            reference = self.val_dataset.decode(tgt, language='vi')

            hypotheses.append(hypothesis)
            references.append([reference])  # BLEU expects list of references

        # Compute BLEU
        bleu = corpus_bleu(hypotheses, references)
        return bleu.score

    def _greedy_decode(self, src: torch.Tensor, max_len: int = 75) -> torch.Tensor:
        """Greedy decoding fallback"""
        start_token = self.config.get('start_token', 1)
        end_token = self.config.get('end_token', 2)

        src = src.unsqueeze(0)
        tgt = torch.tensor([[start_token]], device=self.device)

        for _ in range(max_len):
            output = self.model(src, tgt)
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_token], dim=1)

            if next_token.item() == end_token:
                break

        return tgt.squeeze(0)

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_loss': self.best_val_loss,
            'best_bleu': self.best_bleu,
            'history': self.history,
            'config': self.config
        }

        checkpoint_path = self.save_dir / filename
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

        if is_best:
            best_path = self.save_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved to {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_bleu = checkpoint.get('best_bleu', 0.0)
        self.history = checkpoint.get('history', self.history)

        logger.info(f"Checkpoint loaded from {checkpoint_path}")

    def plot_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Accuracy
        axes[0, 1].plot(self.history['train_acc'], label='Train')
        axes[0, 1].plot(self.history['val_acc'], label='Val')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Perplexity
        axes[1, 0].plot(self.history['train_ppl'], label='Train')
        axes[1, 0].plot(self.history['val_ppl'], label='Val')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Perplexity')
        axes[1, 0].set_title('Perplexity')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # BLEU
        axes[1, 1].plot(self.history['val_bleu'], label='BLEU')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('BLEU Score')
        axes[1, 1].set_title('BLEU Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plot_path = self.save_dir / 'training_history.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {plot_path}")
        plt.close()

    def train(self):
        """Main training loop"""
        logger.info("=" * 80)
        logger.info("Starting training")
        logger.info("=" * 80)

        num_epochs = self.config['num_epochs']
        early_stopping_patience = self.config.get('early_stopping_patience', 10)
        compute_bleu_every = self.config.get('compute_bleu_every', 5)
        save_every = self.config.get('save_every', 5)

        try:
            for epoch in range(self.current_epoch, num_epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()

                # Training
                train_metrics = self.train_epoch()

                # Validation
                val_metrics = self.validate()

                # Compute BLEU periodically
                bleu_score = 0.0
                if (epoch + 1) % compute_bleu_every == 0:
                    bleu_score = self.compute_bleu(num_samples=100)
                    val_metrics['bleu'] = bleu_score

                # Update learning rate (for non-warmup schedulers)
                if self.scheduler and self.config.get('scheduler') != 'warmup':
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step()

                # Record history
                self.history['train_loss'].append(train_metrics['loss'])
                self.history['train_acc'].append(train_metrics.get('accuracy', 0))
                self.history['train_ppl'].append(train_metrics.get('perplexity', 0))
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_acc'].append(val_metrics.get('accuracy', 0))
                self.history['val_ppl'].append(val_metrics.get('perplexity', 0))
                self.history['val_bleu'].append(bleu_score)
                self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

                # Logging
                epoch_time = time.time() - epoch_start_time
                logger.info(f"\nEpoch {epoch + 1}/{num_epochs} - {epoch_time:.2f}s")
                logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                            f"Acc: {train_metrics.get('accuracy', 0):.4f}, "
                            f"PPL: {train_metrics.get('perplexity', 0):.2f}")
                logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                            f"Acc: {val_metrics.get('accuracy', 0):.4f}, "
                            f"PPL: {val_metrics.get('perplexity', 0):.2f}, "
                            f"BLEU: {bleu_score:.2f}")

                # Save checkpoint
                if (epoch + 1) % save_every == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')

                # Check for best model
                is_best = False
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    is_best = True
                    self.patience_counter = 0
                    logger.info(f"New best validation loss: {self.best_val_loss:.4f}")
                else:
                    self.patience_counter += 1

                if bleu_score > self.best_bleu:
                    self.best_bleu = bleu_score
                    logger.info(f"New best BLEU score: {self.best_bleu:.2f}")

                # Save best model
                if is_best:
                    self.save_checkpoint('best_model.pt', is_best=True)

                # Early stopping
                if self.patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break

            # Final checkpoint
            self.save_checkpoint('final_model.pt')

            # Plot history
            self.plot_history()

            logger.info("=" * 80)
            logger.info("Training completed!")
            logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
            logger.info(f"Best BLEU score: {self.best_bleu:.2f}")
            logger.info("=" * 80)

        except KeyboardInterrupt:
            logger.info("\nTraining interrupted by user")
            self.save_checkpoint('interrupted_model.pt')
            self.plot_history()

        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise


def create_config(
        # Model architecture
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 75,

        # Training
        batch_size: int = 32,
        num_epochs: int = 100,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,

        # Optimization
        optimizer: str = 'adamw',
        scheduler: str = 'warmup',
        warmup_steps: int = 4000,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,

        # Loss
        loss_type: str = 'optimized',
        label_smoothing: float = 0.1,
        pad_idx: int = 0,

        # Mixed precision
        use_amp: bool = True,

        # Checkpointing
        save_every: int = 5,
        early_stopping_patience: int = 10,

        # Evaluation
        compute_bleu_every: int = 5,
        max_gen_len: int = 75,
        start_token: int = 1,
        end_token: int = 2,

        # Data loading
        num_workers: int = 4,

        **kwargs
) -> Dict:
    """Create training configuration"""
    config = {
        # Model
        'd_model': d_model,
        'num_heads': num_heads,
        'num_encoder_layers': num_encoder_layers,
        'num_decoder_layers': num_decoder_layers,
        'd_ff': d_ff,
        'dropout': dropout,
        'max_seq_len': max_seq_len,

        # Training
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,

        # Optimization
        'optimizer': optimizer,
        'scheduler': scheduler,
        'warmup_steps': warmup_steps,
        'max_grad_norm': max_grad_norm,
        'gradient_accumulation_steps': gradient_accumulation_steps,

        # Loss
        'loss_type': loss_type,
        'label_smoothing': label_smoothing,
        'pad_idx': pad_idx,

        # Mixed precision
        'use_amp': use_amp,

        # Checkpointing
        'save_every': save_every,
        'early_stopping_patience': early_stopping_patience,

        # Evaluation
        'compute_bleu_every': compute_bleu_every,
        'max_gen_len': max_gen_len,
        'start_token': start_token,
        'end_token': end_token,

        # Data loading
        'num_workers': num_workers,
    }

    config.update(kwargs)
    return config


def main():
    """Main training script"""

    # ============================================================================
    # 1. LOAD DATA
    # ============================================================================
    logger.info("Loading data...")

    # Load your translation data
    # Example format: [{"en": "Hello", "vi": "Xin chào"}, ...]
    train_data = load_your_data('train.json')  # Replace with your data loading
    val_data = load_your_data('val.json')

    # Create datasets
    train_dataset = FastMETTDataset(
        data=train_data,
        tokenizer_eng="bert-base-uncased",
        tokenizer_vie="vinai/phobert-base",
        max_length=75,
        cache_dir="./cache",
        num_workers=4
    )

    val_dataset = FastMETTDataset(
        data=val_data,
        tokenizer_eng="bert-base-uncased",
        tokenizer_vie="vinai/phobert-base",
        max_length=75,
        cache_dir="./cache",
        num_workers=4
    )

    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")

    # ============================================================================
    # 2. CREATE MODEL
    # ============================================================================
    logger.info("Creating model...")

    src_vocab_size = train_dataset.get_vocab_size('eng')
    tgt_vocab_size = train_dataset.get_vocab_size('vi')

    model = OptimizedTransformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        dropout=0.1,
        max_seq_len=512,
        pad_idx=0,
        use_gradient_checkpointing=False
    )

    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # ============================================================================
    # 3. CREATE CONFIG
    # ============================================================================
    config = create_config(
        # Model
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        dropout=0.1,

        # Training
        batch_size=32,
        num_epochs=100,
        learning_rate=1e-4,
        weight_decay=0.01,

        # Optimization
        optimizer='adamw',
        scheduler='warmup',
        warmup_steps=4000,
        max_grad_norm=1.0,
        gradient_accumulation_steps=1,

        # Loss
        loss_type='optimized',
        label_smoothing=0.1,

        # Mixed precision
        use_amp=True,

        # Checkpointing
        save_every=5,
        early_stopping_patience=10,

        # Evaluation
        compute_bleu_every=5,
    )

    # ============================================================================
    # 4. CREATE TRAINER
    # ============================================================================
    trainer = TransformerTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        save_dir='./checkpoints/transformer_en_vi'
    )

    # ============================================================================
    # 5. TRAIN
    # ============================================================================
    trainer.train()

    # ============================================================================
    # 6. EVALUATE
    # ============================================================================
    logger.info("\nFinal evaluation...")

    # Load best model
    best_model_path = './checkpoints/transformer_en_vi/best_model.pt'
    trainer.load_checkpoint(best_model_path)

    # Compute final metrics
    final_val_metrics = trainer.validate()
    final_bleu = trainer.compute_bleu(num_samples=500)

    logger.info("=" * 80)
    logger.info("FINAL RESULTS")
    logger.info("=" * 80)
    logger.info(f"Validation Loss: {final_val_metrics['loss']:.4f}")
    logger.info(f"Validation Accuracy: {final_val_metrics.get('accuracy', 0):.4f}")
    logger.info(f"Validation Perplexity: {final_val_metrics.get('perplexity', 0):.2f}")
    logger.info(f"BLEU Score: {final_bleu:.2f}")
    logger.info("=" * 80)


def load_your_data(file_path: str) -> List[Dict[str, str]]:
    """
    Load your translation data
    Replace this with your actual data loading logic
    """
    import json

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    else:
        # Generate dummy data for testing
        logger.warning(f"File {file_path} not found. Generating dummy data...")
        dummy_data = []
        for i in range(1000):
            en_text = f"This is English sentence number {i}."
            vi_text = f"Đây là câu tiếng Việt số {i}."
            dummy_data.append({"en": en_text, "vi": vi_text})
        return dummy_data


if __name__ == "__main__":
    main()
