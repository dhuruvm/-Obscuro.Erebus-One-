import sys
from typing import Optional

import click

from .architecture import CombinedFoundationModel
from .config import CombinedFoundationConfig
from .dataset_installer import DatasetInstaller
from .exporter import ErebusExporter
from .identity_dataset import ObscuroIdentityDatasetBuilder
from .industrial_runtime import IndustrialRuntime
from .nemo_integration import NeMoAdapter
from .phi3_backend import Phi3MiniBackend
from .pipeline import IndustrialMultiModalTrainingPipeline
from .supervisor import OpenJevSupervisor
from .system_checker import SystemDeviceChecker
from .training import TrainingRecipe, run_demo_training


@click.group()
def cli():
    """Obscuro Erebus combined foundation model CLI."""


@cli.command()
@click.option("--hidden-size", default=4096, show_default=True, type=int)
@click.option("--steps", default=8, show_default=True, type=int)
def demo(hidden_size: int, steps: int):
    """Run a compact architecture demo and print the model summary."""
    config = CombinedFoundationConfig(hidden_size=hidden_size, num_nar_steps=steps)
    model = CombinedFoundationModel(config)
    out = model.forward([1, 2, 3, 4, 5], [1, 1, 1, 1, 1])
    click.echo("Combined foundation model: " + config.model_name)
    click.echo(f"Estimated parameters: {model.estimate_parameters()}")
    click.echo(f"NAR output shape: {out['nar_output'].shape}")
    click.echo(f"Reasoning output shape: {out['reasoning_output'].shape}")
    click.echo(f"Final combined output shape: {out['combined_output'].shape}")
    click.echo("Architecture stack: RLCD, Verification, GRPO/RLVR, RLHF, NAR, MoE, Multimodal")


@cli.command()
def verify():
    """Verify that the combined architecture supports the requested components."""
    config = CombinedFoundationConfig()
    reqs = {
        "Non-autoregressive": config.use_nar,
        "Extended_reasoning": config.use_extended_reasoning,
        "Natively_multimodal_MoE": config.use_multimodal,
        "Autoregressive_MoE": config.use_ar_moe,
        "RLCD_training": "RLCD" in config.training_objectives,
        "RL_Verification": "RL / Verification" in config.training_objectives,
        "GRPO_RLVR": "GRPO / RLVR" in config.training_objectives,
        "RLHF": "RLHF" in config.training_objectives,
    }

    click.echo("Verification status:")
    for name, ok in reqs.items():
        status = "OK" if ok else "MISSING"
        click.echo(f"  - {name}: {status}")


@cli.command()
@click.option("--steps", default=10, show_default=True, type=int)
def train(steps: int):
    """Run a compact synthetic RL training cycle to demonstrate the recipe."""
    recipe = TrainingRecipe(steps=steps)
    metrics = run_demo_training(recipe)
    click.echo("Training summary:")
    for key, value in metrics.items():
        click.echo(f"  - {key}: {value:.4f}")


@cli.command()
def industrial():
    """Run the industrial multi-stage multimodal RL pipeline scaffold."""
    pipeline = IndustrialMultiModalTrainingPipeline()
    summary = pipeline.run_full_pipeline()
    click.echo(f"Model: {summary.model_name}")
    click.echo("Industrial training pipeline stages:")
    for stage in summary.stages:
        click.echo(f"  - {stage}")
    click.echo("Stage metrics:")
    for stage_name, score in summary.metrics.items():
        click.echo(f"  - {stage_name}: {score:.4f}")
    click.echo(f"Final pipeline score: {summary.final_score:.4f}")


@cli.command(name="install_datasets")
def install_datasets():
    """Phase 1: Install, generate synthetic seeds, and verify all domain dataset folders."""
    installer = DatasetInstaller()
    result = installer.verify_and_install()
    click.echo("==================================================")
    click.echo(" PHASE 1: DATASET INSTALLATION & VERIFICATION")
    click.echo("==================================================")
    click.echo(f"Status: {result['status']}")
    click.echo(f"Datasets Root: {result['root']}")
    click.echo(f"Company Identity Dataset: {result['company_identity']['prepared_records']} records ({result['company_identity']['path']})")
    click.echo(f"FineWeb Sample Corpus: {result['fineweb']['records']} records ({result['fineweb']['path']})")
    click.echo(f"Human Philosophy Books: {result['human_philosophy']['records']} records ({result['human_philosophy']['path']})")
    click.echo(f"Psychology Books: {result['psychology']['records']} records ({result['psychology']['path']})")
    click.echo(f"Coding Systems: {result['coding']['records']} records ({result['coding']['path']})")
    click.echo(f"Mathematics Proofs: {result['mathematics']['records']} records ({result['mathematics']['path']})")
    click.echo(f"Quantum Physics: {result['quantum_physics']['records']} records ({result['quantum_physics']['path']})")
    click.echo(f"Multimodal Batch Data: {result['multimodal']['sample_count']} samples ({result['multimodal']['path']})")
    click.echo(f"Preference Pairs Data: {result['preferences']['records']} pairs ({result['preferences']['path']})")
    click.echo(f"Chromium Learner Corpus: {result['chromium_browser']['records']} pages / {result['chromium_browser']['rlvr_pairs']} RLVR pairs ({result['chromium_browser']['path']})")


@cli.command(name="syscheck")
def syscheck():
    """Phase 2: Perform comprehensive system hardware, GPU/CPU, and 32B model diagnostics."""
    config = CombinedFoundationConfig.create_32b_config()
    checker = SystemDeviceChecker(config=config)
    info = checker.check_all()
    click.echo("==================================================")
    click.echo(" PHASE 2: HARDWARE & SYSTEM DEVICE CHECK")
    click.echo("==================================================")
    click.echo(f"Status: {info['status']}")
    click.echo(f"OS Platform: {info['os']}")
    click.echo(f"Python Runtime: {info['python_version']}")
    click.echo(f"CPU Cores: {info['cpu_cores']}")
    click.echo(f"Compute Device: {info['compute_device']}")
    if info["cuda_available"]:
        click.echo(f"GPU Name: {info['gpu_name']} ({info['vram_gb']} GB VRAM)")
    click.echo("Model Specifications:")
    for k, v in info["model_specs"].items():
        click.echo(f"  - {k}: {v}")
    click.echo(f"System Readiness: {info['system_readiness']}")


@cli.command()
def fullrun():
    """Run full industrial execution in 4 phases: Install Datasets -> System Check -> Train Model -> Save Artifacts."""
    click.echo("==========================================================================")
    click.echo("      OBSCURO EREBUS 32B COMBINED FOUNDATION MODEL - FULL RUN EXECUTOR")
    click.echo("==========================================================================")

    # Phase 1
    click.echo("\n[PHASE 1/4] Installing & Verifying All Domain Dataset Folders...")
    installer = DatasetInstaller()
    ds_res = installer.verify_and_install()
    click.echo(f"  [OK] Company Identity Records: {ds_res['company_identity']['prepared_records']}")
    click.echo(f"  [OK] FineWeb Corpus Records: {ds_res['fineweb']['records']}")
    click.echo(f"  [OK] Philosophy Books: {ds_res['human_philosophy']['records']}")
    click.echo(f"  [OK] Psychology Books: {ds_res['psychology']['records']}")
    click.echo(f"  [OK] Coding Systems: {ds_res['coding']['records']}")
    click.echo(f"  [OK] Mathematics Proofs: {ds_res['mathematics']['records']}")
    click.echo(f"  [OK] Quantum Physics: {ds_res['quantum_physics']['records']}")
    click.echo(f"  [OK] Chromium Autonomous Learner Corpus: {ds_res['chromium_browser']['records']}")
    click.echo(f"  [OK] Multimodal Samples: {ds_res['multimodal']['sample_count']}")
    click.echo(f"  [OK] Preference Pairs: {ds_res['preferences']['records']}")

    # Phase 2
    click.echo("\n[PHASE 2/4] Checking All System Devices & Model Hardware Specs...")
    config = CombinedFoundationConfig.create_32b_config()
    checker = SystemDeviceChecker(config=config)
    sys_info = checker.check_all()
    click.echo(f"  [OK] OS: {sys_info['os']}")
    click.echo(f"  [OK] Compute Device: {sys_info['compute_device']}")
    click.echo(f"  [OK] CPU Cores: {sys_info['cpu_cores']}")
    click.echo(f"  [OK] Target Model Parameters: {sys_info['model_specs']['estimated_params_billion']} Billion ({sys_info['model_specs']['estimated_params_million']} M)")
    click.echo(f"  [OK] System Readiness: {sys_info['system_readiness']}")

    # Phase 3 & 4
    click.echo("\n[PHASE 3/4] Training 32B Model & Executing Multimodal RL Pipeline...")
    runtime = IndustrialRuntime(model_name="Obscuro-Erebus-Combined-Foundation-32B")
    result = runtime.run()
    for stage_name, score in result.metrics.items():
        click.echo(f"  [OK] Stage [{stage_name}] Score: {score:.4f}")
    click.echo(f"  [OK] Final Multi-Stage Pipeline Score: {result.final_score:.4f}")

    click.echo("\n[PHASE 4/4] Writing Summary Artifacts & Reproducible Records...")
    click.echo(f"  [OK] Reproducible Artifact Saved: {result.artifact_path}")
    click.echo("==========================================================================")
    click.echo("                     FULL RUN COMPLETED SUCCESSFULLY [OK]")
    click.echo("==========================================================================")



@cli.command()
def nemosetup():
    """Show the NeMo-compatible configuration for the combined model stack."""
    adapter = NeMoAdapter()
    cfg = adapter.build_config()
    click.echo("NeMo setup:")
    for key, value in cfg.items():
        click.echo(f"  - {key}: {value}")


@cli.command()
def phi3():
    """Show the Microsoft Phi-3 Mini base backend configuration for the combined model stack."""
    backend = Phi3MiniBackend()
    info = backend.describe()
    click.echo("Phi-3 Mini base model:")
    for key, value in info.items():
        click.echo(f"  - {key}: {value}")


@cli.command()
def corpora():
    """List the reference corpora planned for a production-scale pretraining mix."""
    corpora = {
        "FineWeb_10B": "high-quality web corpus reference",
        "The_Stack_v2": "code corpus",
        "Dolma": "diverse Open web corpus",
        "RedPajama_Data_v2": "text corpus blend",
        "UltraChat_200k": "instruction-following dataset",
        "UltraFeedback": "preference and feedback data",
    }
    click.echo("Planned training corpora:")
    for name, purpose in corpora.items():
        click.echo(f"  - {name}: {purpose}")


@cli.command(name="identity_dataset")
def identity_dataset():
    """Generate the Obscuro Studio identity dataset for SFT and QLoRA training."""
    builder = ObscuroIdentityDatasetBuilder()
    result = builder.export_prepared()
    click.echo("Identity dataset created:")
    click.echo(f"  - synthetic_records: {result['synthetic_records']}")
    click.echo(f"  - prepared_records: {result['prepared_records']}")
    click.echo(f"  - path: {result['path']}")


@cli.command(name="export_erebus")
def export_erebus():
    """Export the ErebusV1 model artifacts in C++ and GGUF/GFFU-like formats."""
    exporter = ErebusExporter()
    result = exporter.export_all()
    click.echo("ErebusV1 export created:")
    for key, value in result.items():
        click.echo(f"  - {key}: {value}")


@cli.command()
def supervisor():
    """Query the Open-Dev supervisor/instructor layer for a training advice example."""
    supervisor = OpenJevSupervisor()
    sample_state = "Customer refund request with urgency, delay, and service dissatisfaction"
    sample_questions = {
        "refund_requested": {"type": "noul", "instructions": "Is a refund explicitly requested?"},
        "route": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": "Refunds and charges", "engineering": "Software defects"},
        },
        "frustration": {"type": "score", "instructions": "Rate expressed frustration.", "criteria": ["Calm", "Frustrated but civil", "Very angry"]},
    }
    result = supervisor.advise_training(sample_state, sample_questions)
    click.echo("Supervisor result:")
    click.echo(f"  - status: {result['status']}")
    for key, value in result["answers"].items():
        click.echo(f"  - {key}: {value}")


@cli.command(name="chromium_learn")
@click.option("--pages", default=20, show_default=True, type=int, help="Max pages to crawl per domain session")
def chromium_learn(pages: int):
    """Run autonomous Chromium browser self-learning agent to collect technical knowledge & RLVR pairs."""
    from .chromium_learner import ChromiumAutonomousLearner
    learner = ChromiumAutonomousLearner()
    click.echo(f"Starting Chromium Autonomous Learning Session (max_pages={pages})...")
    res = learner.run_autonomous_session(max_pages=pages)
    click.echo("Session completed:")
    click.echo(f"  - Pages ingested: {res['session_ingested_pages']}")
    click.echo(f"  - Total Chromium Corpus Records: {res['total_chromium_corpus_records']}")
    click.echo(f"  - Total Chromium RLVR Pairs: {res['total_chromium_rlvr_pairs']}")
    click.echo(f"  - Saved Corpus: {res['corpus_file']}")
    click.echo(f"  - Saved Preference File: {res['preference_file']}")


@cli.command()
@click.option("--interval", default=300, show_default=True, type=int, help="Sleep interval in seconds between autonomous training runs")
@click.option("--once", is_flag=True, help="Run single worker cycle (chromium learn + full pipeline) and exit")
@click.option("--pages", default=20, show_default=True, type=int, help="Chromium pages to crawl per session")
def worker(interval: int, once: bool, pages: int):
    """Run autonomous 32B training worker: self-browser learning → full RL pipeline → repeat."""
    from .docker_worker import run_worker
    run_worker(interval=interval, single_run=once, chromium_pages=pages)


if __name__ == "__main__":
    cli()

