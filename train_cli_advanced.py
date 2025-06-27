#!/usr/bin/env python
"""
ACE-Step Advanced Training CLI Interface
A clean dashboard-style command-line interface for training ACE-Step models
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich import box
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.align import Align
from loguru import logger
import warnings
from pytorch_lightning.callbacks import Callback

# Import the trainer module
from trainer import Pipeline, ModelCheckpoint, TensorBoardLogger, Trainer

# Create console for rich output
console = Console()

class TrainingDashboard(Callback):
    """Real-time training dashboard using Rich"""
    
    def __init__(self, max_steps, refresh_rate=1.0):
        self.max_steps = max_steps
        self.refresh_rate = refresh_rate
        self.start_time = None
        self.current_step = 0
        self.current_epoch = 0
        self.losses = {'loss': 0.0, 'denoising_loss': 0.0, 'mert_loss': 0.0, 'm-hubert_loss': 0.0}
        self.learning_rate = 0.0
        self.last_update = time.time()
        
        # Create layout
        self.layout = Layout()
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        # Split body into panels
        self.layout["body"].split_row(
            Layout(name="progress", ratio=2),
            Layout(name="metrics", ratio=1)
        )
        
        # Create progress bar
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            MofNCompleteColumn(),
            "•",
            TimeRemainingColumn(),
            expand=True
        )
        self.progress_task = self.progress.add_task("Training Progress", total=max_steps)
        
        # Start live display
        self.live = Live(self.layout, console=console, refresh_per_second=2)
    
    def on_train_start(self, trainer, pl_module):
        """Initialize dashboard on training start"""
        self.start_time = datetime.now()
        self.live.start()
        self.update_display()
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Update dashboard on batch end"""
        # Rate limit updates
        current_time = time.time()
        if current_time - self.last_update < self.refresh_rate:
            return
        self.last_update = current_time
        
        # Update metrics
        self.current_step = trainer.global_step
        self.current_epoch = trainer.current_epoch
        
        # Get metrics from trainer
        metrics = trainer.callback_metrics
        self.losses['loss'] = metrics.get('train/loss', 0.0)
        self.losses['denoising_loss'] = metrics.get('train/denoising_loss', 0.0)
        self.losses['mert_loss'] = metrics.get('train/mert_loss', 0.0)
        self.losses['m-hubert_loss'] = metrics.get('train/m-hubert_loss', 0.0)
        self.learning_rate = metrics.get('train/learning_rate', 0.0)
        
        # Update progress
        self.progress.update(self.progress_task, completed=self.current_step)
        
        # Update display
        self.update_display()
    
    def update_display(self):
        """Update the dashboard display"""
        # Header
        header_text = Text("ACE-Step Training Dashboard", style="bold white on blue", justify="center")
        self.layout["header"].update(Panel(header_text, box=box.ROUNDED))
        
        # Progress panel
        progress_content = self.create_progress_panel()
        self.layout["progress"].update(progress_content)
        
        # Metrics panel
        metrics_content = self.create_metrics_panel()
        self.layout["metrics"].update(metrics_content)
        
        # Footer
        footer_content = self.create_footer_panel()
        self.layout["footer"].update(footer_content)
    
    def create_progress_panel(self):
        """Create progress information panel"""
        # Time calculations
        elapsed = datetime.now() - self.start_time if self.start_time else timedelta(0)
        if self.current_step > 0:
            time_per_step = elapsed.total_seconds() / self.current_step
            eta = timedelta(seconds=time_per_step * (self.max_steps - self.current_step))
        else:
            eta = timedelta(0)
        
        # Speed calculation
        if elapsed.total_seconds() > 0:
            steps_per_second = self.current_step / elapsed.total_seconds()
        else:
            steps_per_second = 0
        
        # Create content
        content = Table(box=None, show_header=False, padding=(0, 1))
        content.add_column(style="cyan", width=20)
        content.add_column(style="green")
        
        content.add_row("Step", f"{self.current_step:,} / {self.max_steps:,}")
        content.add_row("Epoch", str(self.current_epoch))
        content.add_row("Progress", f"{self.current_step / self.max_steps * 100:.1f}%")
        content.add_row("Speed", f"{steps_per_second:.2f} steps/s")
        content.add_row("Elapsed", str(elapsed).split('.')[0])
        content.add_row("ETA", str(eta).split('.')[0])
        
        # Add progress bar
        progress_panel = Panel(
            Align.center(content, vertical="middle"),
            title="[bold]Training Progress",
            box=box.ROUNDED
        )
        
        # Stack with actual progress bar
        from rich.console import Group
        return Panel(
            Group(
                self.progress,
                "",
                progress_panel
            ),
            box=box.ROUNDED,
            title="Progress"
        )
    
    def create_metrics_panel(self):
        """Create metrics panel"""
        table = Table(box=box.SIMPLE, show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")
        
        # Add loss metrics
        table.add_row("Total Loss", f"{self.losses['loss']:.4f}")
        table.add_row("Denoising Loss", f"{self.losses['denoising_loss']:.4f}")
        if self.losses['mert_loss'] > 0:
            table.add_row("MERT Loss", f"{self.losses['mert_loss']:.4f}")
        if self.losses['m-hubert_loss'] > 0:
            table.add_row("M-HuBERT Loss", f"{self.losses['m-hubert_loss']:.4f}")
        
        table.add_section()
        table.add_row("Learning Rate", f"{self.learning_rate:.2e}")
        
        return Panel(
            Align.center(table, vertical="middle"),
            title="[bold]Metrics",
            box=box.ROUNDED
        )
    
    def create_footer_panel(self):
        """Create footer panel with tips"""
        tips = [
            "Press Ctrl+C to stop training gracefully",
            "Logs are saved in the logs/ directory",
            "Checkpoints are saved every 2000 steps by default"
        ]
        
        tip = tips[self.current_step % len(tips)]
        footer_text = Text(f"💡 Tip: {tip}", style="dim", justify="center")
        
        return Panel(footer_text, box=box.ROUNDED, style="dim")
    
    def on_train_end(self, trainer, pl_module):
        """Clean up on training end"""
        self.live.stop()
        console.print("\n[green]✨ Training completed successfully![/green]")
    
    def on_exception(self, trainer, pl_module, exception):
        """Handle exceptions"""
        self.live.stop()
        console.print(f"\n[red]❌ Training failed: {exception}[/red]")


class AdvancedTrainingCLI:
    """Advanced CLI interface with dashboard for ACE-Step training"""
    
    def __init__(self):
        self.console = console
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging to file only (not console)"""
        logger.remove()
        
        # File logger for detailed logs
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        logger.add(
            log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="100 MB"
        )
        
        # Console logger only for errors
        logger.add(
            sys.stderr,
            format="<red>{message}</red>",
            level="ERROR",
            colorize=True
        )
        
        # Suppress warnings
        warnings.filterwarnings("ignore")
        import logging
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        logging.getLogger("acestep").setLevel(logging.ERROR)
    
    def print_banner(self):
        """Print welcome banner"""
        banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║           🎵 ACE-Step Training Dashboard 🎵               ║
    ║                                                           ║
    ║     Advanced Music Generation Model Training Suite        ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
        """
        self.console.print(banner, style="bold cyan", justify="center")
    
    def run_with_args(self, args):
        """Run training with parsed arguments"""
        # Print banner
        self.print_banner()
        
        # Validate paths
        if not Path(args.dataset_path).exists():
            self.console.print(f"\n[red]❌ Error: Dataset path '{args.dataset_path}' does not exist![/red]")
            sys.exit(1)
        
        if not Path(args.lora_config_path).exists():
            self.console.print(f"\n[red]❌ Error: LoRA config path '{args.lora_config_path}' does not exist![/red]")
            sys.exit(1)
        
        # Show configuration summary
        self.show_config_summary(args)
        
        # Ask for confirmation
        if not args.yes:
            confirm = console.input("\n[yellow]Start training with these settings? [y/N]:[/yellow] ")
            if confirm.lower() != 'y':
                console.print("[yellow]Training cancelled.[/yellow]")
                sys.exit(0)
        
        # Initialize components
        console.print("\n[cyan]Initializing training components...[/cyan]")
        
        try:
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
            
            # Callbacks
            checkpoint_callback = ModelCheckpoint(
                monitor=None,
                every_n_train_steps=args.every_n_train_steps,
                save_top_k=-1,
            )
            
            dashboard = TrainingDashboard(max_steps=args.max_steps)
            
            # Logger
            logger_callback = TensorBoardLogger(
                version=datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + args.exp_name,
                save_dir=args.logger_dir,
            )
            
            # Trainer
            trainer = Trainer(
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=args.devices,
                num_nodes=args.num_nodes,
                precision=args.precision,
                accumulate_grad_batches=args.accumulate_grad_batches,
                strategy="ddp_find_unused_parameters_true" if args.devices > 1 else "auto",
                max_epochs=args.epochs,
                max_steps=args.max_steps,
                log_every_n_steps=1,
                logger=logger_callback,
                callbacks=[checkpoint_callback, dashboard],
                gradient_clip_val=args.gradient_clip_val,
                gradient_clip_algorithm=args.gradient_clip_algorithm,
                reload_dataloaders_every_n_epochs=args.reload_dataloaders_every_n_epochs,
                val_check_interval=args.val_check_interval,
                enable_progress_bar=False,  # We use our own dashboard
            )
            
            # Start training
            console.print("\n[green]Starting training...[/green]\n")
            trainer.fit(model, ckpt_path=args.ckpt_path)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠️  Training interrupted by user[/yellow]")
            sys.exit(0)
        except Exception as e:
            console.print(f"\n[red]❌ Training failed: {e}[/red]")
            logger.exception("Training error:")
            sys.exit(1)
    
    def show_config_summary(self, args):
        """Show configuration summary"""
        # Create configuration table
        config_table = Table(title="Training Configuration", box=box.DOUBLE_EDGE, title_style="bold cyan")
        config_table.add_column("Category", style="cyan", width=20)
        config_table.add_column("Parameter", style="white", width=25)
        config_table.add_column("Value", style="green", width=30)
        
        # Dataset
        config_table.add_row("Dataset", "Path", str(args.dataset_path))
        config_table.add_row("", "LoRA Config", str(args.lora_config_path))
        config_table.add_row("", "Experiment Name", args.exp_name)
        
        config_table.add_section()
        
        # Training
        config_table.add_row("Training", "Learning Rate", f"{args.learning_rate:.2e}")
        config_table.add_row("", "Batch Size", str(args.batch_size))
        config_table.add_row("", "Max Steps", f"{args.max_steps:,}")
        config_table.add_row("", "Gradient Clip", str(args.gradient_clip_val))
        config_table.add_row("", "Precision", args.precision)
        
        config_table.add_section()
        
        # Hardware
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            config_table.add_row("Hardware", "GPU", f"{gpu_name}")
            config_table.add_row("", "GPU Memory", f"{gpu_memory:.1f} GB")
        else:
            config_table.add_row("Hardware", "Device", "CPU")
        config_table.add_row("", "Num Workers", str(args.num_workers))
        
        config_table.add_section()
        
        # Checkpointing
        config_table.add_row("Checkpointing", "Save Every", f"{args.every_n_train_steps:,} steps")
        config_table.add_row("", "Eval Every", f"{args.every_plot_step:,} steps")
        config_table.add_row("", "Log Directory", str(args.logger_dir))
        
        self.console.print(config_table)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="ACE-Step Advanced Training CLI - Train with real-time dashboard"
    )
    
    # Required
    parser.add_argument("--dataset_path", type=str, required=True,
                      help="Path to Huggingface dataset")
    parser.add_argument("--lora_config_path", type=str, required=True,
                      help="Path to LoRA config JSON")
    
    # Training
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=2000000)
    parser.add_argument("--epochs", type=int, default=-1)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=0.5)
    parser.add_argument("--gradient_clip_algorithm", type=str, default="norm")
    
    # Model
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--precision", type=str, default="32")
    
    # Hardware
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    
    # Experiment
    parser.add_argument("--exp_name", type=str, default="chinese_rap_lora")
    parser.add_argument("--logger_dir", type=str, default="./exps/logs/")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    
    # Logging
    parser.add_argument("--every_n_train_steps", type=int, default=2000)
    parser.add_argument("--every_plot_step", type=int, default=2000)
    parser.add_argument("--val_check_interval", type=int, default=None)
    parser.add_argument("--reload_dataloaders_every_n_epochs", type=int, default=1)
    
    # Other
    parser.add_argument("--yes", "-y", action="store_true",
                      help="Skip confirmation prompt")
    
    args = parser.parse_args()
    
    # Run CLI
    cli = AdvancedTrainingCLI()
    cli.run_with_args(args)


if __name__ == "__main__":
    main() 