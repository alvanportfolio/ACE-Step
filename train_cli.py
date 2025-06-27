#!/usr/bin/env python
"""
ACE-Step Training CLI Interface
A clean command-line interface for training ACE-Step models with LoRA
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.live import Live
from rich.layout import Layout
from rich import box
from loguru import logger
import warnings

# Import the trainer module
from trainer import Pipeline, ModelCheckpoint, TensorBoardLogger, Trainer

# Create console for rich output
console = Console()

class TrainingCLI:
    """Clean CLI interface for ACE-Step training"""
    
    def __init__(self):
        self.console = console
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging to be clean and informative"""
        # Remove default logger
        logger.remove()
        
        # Add custom logger with clean format
        logger.add(
            sys.stderr,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
            level="INFO",
            colorize=True
        )
        
        # Add file logger for detailed logs
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        logger.add(
            log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="100 MB"
        )
        
        # Suppress noisy warnings
        warnings.filterwarnings("ignore", message=".*Unsupported language.*")
        warnings.filterwarnings("ignore", message=".*find_unused_parameters=True.*")
        
        # Reduce verbosity of specific loggers
        import logging
        logging.getLogger("acestep.text2music_dataset").setLevel(logging.ERROR)
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    
    def create_argument_parser(self):
        """Create argument parser with organized groups"""
        parser = argparse.ArgumentParser(
            description="ACE-Step Training CLI - Train music generation models with LoRA",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Basic training with default settings
  python train_cli.py --dataset_path ./zh_lora_dataset --lora_config_path config/zh_rap_lora_config.json
  
  # Training with custom settings
  python train_cli.py --dataset_path ./zh_lora_dataset --lora_config_path config/zh_rap_lora_config.json \\
                      --learning_rate 5e-5 --batch_size 2 --max_steps 100000
  
  # Resume training from checkpoint
  python train_cli.py --dataset_path ./zh_lora_dataset --lora_config_path config/zh_rap_lora_config.json \\
                      --ckpt_path ./exps/logs/2024-01-01_12-00-00_chinese_rap_lora/checkpoints/last.ckpt
            """
        )
        
        # Required arguments
        required = parser.add_argument_group('Required Arguments')
        required.add_argument("--dataset_path", type=str, required=True,
                            help="Path to the Huggingface dataset (e.g., ./zh_lora_dataset)")
        required.add_argument("--lora_config_path", type=str, required=True,
                            help="Path to LoRA configuration JSON file")
        
        # Training parameters
        training = parser.add_argument_group('Training Parameters')
        training.add_argument("--learning_rate", type=float, default=1e-4,
                            help="Learning rate for optimization (default: 1e-4)")
        training.add_argument("--batch_size", type=int, default=1,
                            help="Training batch size (default: 1)")
        training.add_argument("--max_steps", type=int, default=2000000,
                            help="Maximum training steps (default: 2000000)")
        training.add_argument("--epochs", type=int, default=-1,
                            help="Number of epochs (-1 for unlimited, default: -1)")
        training.add_argument("--accumulate_grad_batches", type=int, default=1,
                            help="Gradient accumulation steps (default: 1)")
        training.add_argument("--gradient_clip_val", type=float, default=0.5,
                            help="Gradient clipping value (default: 0.5)")
        training.add_argument("--gradient_clip_algorithm", type=str, default="norm",
                            help="Gradient clipping algorithm (default: norm)")
        
        # Model parameters
        model = parser.add_argument_group('Model Parameters')
        model.add_argument("--shift", type=float, default=3.0,
                         help="Flow matching shift parameter (default: 3.0)")
        model.add_argument("--precision", type=str, default="32",
                         help="Training precision: 16, 32, or bf16 (default: 32)")
        
        # Hardware settings
        hardware = parser.add_argument_group('Hardware Settings')
        hardware.add_argument("--devices", type=int, default=1,
                            help="Number of GPUs to use (default: 1)")
        hardware.add_argument("--num_nodes", type=int, default=1,
                            help="Number of nodes for distributed training (default: 1)")
        hardware.add_argument("--num_workers", type=int, default=8,
                            help="Number of data loading workers (default: 8)")
        
        # Experiment settings
        experiment = parser.add_argument_group('Experiment Settings')
        experiment.add_argument("--exp_name", type=str, default="chinese_rap_lora",
                              help="Experiment name for logging (default: chinese_rap_lora)")
        experiment.add_argument("--logger_dir", type=str, default="./exps/logs/",
                              help="Directory for logs and checkpoints (default: ./exps/logs/)")
        experiment.add_argument("--checkpoint_dir", type=str, default=None,
                              help="Directory for model checkpoints (default: auto-download)")
        experiment.add_argument("--ckpt_path", type=str, default=None,
                              help="Path to resume training from checkpoint")
        
        # Logging and validation
        logging = parser.add_argument_group('Logging and Validation')
        logging.add_argument("--every_n_train_steps", type=int, default=2000,
                           help="Save checkpoint every N steps (default: 2000)")
        logging.add_argument("--every_plot_step", type=int, default=2000,
                           help="Generate evaluation samples every N steps (default: 2000)")
        logging.add_argument("--val_check_interval", type=int, default=None,
                           help="Validation interval in steps (default: None)")
        logging.add_argument("--reload_dataloaders_every_n_epochs", type=int, default=1,
                           help="Reload dataloaders every N epochs (default: 1)")
        
        # Display settings
        display = parser.add_argument_group('Display Settings')
        display.add_argument("--quiet", action="store_true",
                           help="Minimal output mode")
        display.add_argument("--verbose", action="store_true",
                           help="Verbose output mode")
        
        return parser
    
    def validate_arguments(self, args):
        """Validate and process arguments"""
        # Check dataset path
        if not Path(args.dataset_path).exists():
            self.console.print(f"[red]Error: Dataset path '{args.dataset_path}' does not exist![/red]")
            sys.exit(1)
        
        # Check LoRA config
        if not Path(args.lora_config_path).exists():
            self.console.print(f"[red]Error: LoRA config path '{args.lora_config_path}' does not exist![/red]")
            sys.exit(1)
        
        # Check GPU availability
        if not torch.cuda.is_available() and args.devices > 0:
            self.console.print("[yellow]Warning: CUDA not available, switching to CPU mode[/yellow]")
            args.devices = 0
        
        # Create directories
        Path(args.logger_dir).mkdir(parents=True, exist_ok=True)
        
        return args
    
    def display_training_config(self, args):
        """Display training configuration in a nice table"""
        table = Table(title="Training Configuration", box=box.ROUNDED)
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")
        
        # Dataset info
        table.add_section()
        table.add_row("Dataset Path", str(args.dataset_path))
        table.add_row("LoRA Config", str(args.lora_config_path))
        table.add_row("Experiment Name", args.exp_name)
        
        # Training params
        table.add_section()
        table.add_row("Learning Rate", f"{args.learning_rate:.2e}")
        table.add_row("Batch Size", str(args.batch_size))
        table.add_row("Max Steps", f"{args.max_steps:,}")
        table.add_row("Gradient Clip", str(args.gradient_clip_val))
        
        # Hardware
        table.add_section()
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            table.add_row("GPU", f"{gpu_name} ({gpu_memory:.1f}GB)")
        table.add_row("Devices", str(args.devices))
        table.add_row("Workers", str(args.num_workers))
        table.add_row("Precision", args.precision)
        
        # Checkpointing
        table.add_section()
        table.add_row("Checkpoint Every", f"{args.every_n_train_steps:,} steps")
        table.add_row("Evaluation Every", f"{args.every_plot_step:,} steps")
        
        panel = Panel(table, title="ACE-Step Training", border_style="blue")
        self.console.print(panel)
    
    def setup_training(self, args):
        """Setup training components"""
        logger.info("Initializing training components...")
        
        # Create model
        model = Pipeline(
            learning_rate=args.learning_rate,
            num_workers=args.num_workers,
            shift=args.shift,
            max_steps=args.max_steps,
            every_plot_step=args.every_plot_step,
            dataset_path=args.dataset_path,
            checkpoint_dir=args.checkpoint_dir,
            adapter_name=args.exp_name,
            lora_config_path=args.lora_config_path,
            batch_size=args.batch_size,
        )
        
        # Setup callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor=None,
            every_n_train_steps=args.every_n_train_steps,
            save_top_k=-1,
        )
        
        # Custom progress callback
        progress_callback = CleanProgressBar(
            max_steps=args.max_steps,
            quiet=args.quiet
        )
        
        # Logger
        logger_callback = TensorBoardLogger(
            version=datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + args.exp_name,
            save_dir=args.logger_dir,
        )
        
        # Trainer
        trainer = Trainer(
            accelerator="gpu" if args.devices > 0 else "cpu",
            devices=args.devices,
            num_nodes=args.num_nodes,
            precision=args.precision,
            accumulate_grad_batches=args.accumulate_grad_batches,
            strategy="ddp_find_unused_parameters_true" if args.devices > 1 else "auto",
            max_epochs=args.epochs,
            max_steps=args.max_steps,
            log_every_n_steps=1,
            logger=logger_callback,
            callbacks=[checkpoint_callback, progress_callback],
            gradient_clip_val=args.gradient_clip_val,
            gradient_clip_algorithm=args.gradient_clip_algorithm,
            reload_dataloaders_every_n_epochs=args.reload_dataloaders_every_n_epochs,
            val_check_interval=args.val_check_interval,
            enable_progress_bar=not args.quiet,
        )
        
        return model, trainer
    
    def run(self):
        """Main entry point"""
        parser = self.create_argument_parser()
        args = parser.parse_args()
        
        # Set logging level
        if args.quiet:
            logger.remove()
            logger.add(sys.stderr, level="WARNING")
        elif args.verbose:
            logger.remove()
            logger.add(sys.stderr, level="DEBUG")
        
        # Validate arguments
        args = self.validate_arguments(args)
        
        # Display configuration
        if not args.quiet:
            self.display_training_config(args)
        
        # Setup training
        try:
            model, trainer = self.setup_training(args)
            
            # Start training
            if not args.quiet:
                self.console.print("\n[green]Starting training...[/green]\n")
            
            trainer.fit(model, ckpt_path=args.ckpt_path)
            
            if not args.quiet:
                self.console.print("\n[green]Training completed successfully![/green]")
                
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Training interrupted by user[/yellow]")
            sys.exit(0)
        except Exception as e:
            self.console.print(f"\n[red]Training failed with error: {e}[/red]")
            logger.exception("Training error:")
            sys.exit(1)


class CleanProgressBar:
    """Clean progress bar for training"""
    
    def __init__(self, max_steps, quiet=False):
        self.max_steps = max_steps
        self.quiet = quiet
        self.last_update = datetime.now()
        self.update_interval = 1.0  # seconds
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Update progress on batch end"""
        if self.quiet:
            return
            
        # Only update at intervals to reduce overhead
        now = datetime.now()
        if (now - self.last_update).total_seconds() < self.update_interval:
            return
        self.last_update = now
        
        # Get metrics
        step = trainer.global_step
        progress = step / self.max_steps * 100
        
        # Get loss values
        metrics = trainer.callback_metrics
        loss = metrics.get('train/loss', 0.0)
        denoising_loss = metrics.get('train/denoising_loss', 0.0)
        lr = metrics.get('train/learning_rate', 0.0)
        
        # Create status line
        status = (
            f"Step: {step}/{self.max_steps} ({progress:.1f}%) | "
            f"Loss: {loss:.4f} | Denoising: {denoising_loss:.4f} | "
            f"LR: {lr:.2e}"
        )
        
        # Use carriage return to update the same line
        print(f"\r{status}", end='', flush=True)
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Print newline at epoch end"""
        if not self.quiet:
            print()  # New line after progress


def main():
    """Main entry point"""
    cli = TrainingCLI()
    cli.run()


if __name__ == "__main__":
    main() 