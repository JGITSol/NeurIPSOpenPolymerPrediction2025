#!/usr/bin/env python3
"""
Example script demonstrating how to use the production-ready polymer prediction pipeline.

This script shows various ways to run the production pipeline with different configurations
and command-line options.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle the output."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        print(f"✅ Command completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)
        return False


def main():
    """Run various examples of the production pipeline."""
    
    # Check if the main script exists
    main_script = Path("main_production.py")
    if not main_script.exists():
        print(f"❌ Main script not found: {main_script}")
        return 1
    
    print("🚀 Production Pipeline Examples")
    print("This script demonstrates various ways to run the production pipeline.")
    
    # Example 1: Basic run with default configuration
    print("\n📋 Example 1: Basic run with default configuration")
    cmd1 = [sys.executable, "main_production.py"]
    success1 = run_command(cmd1, "Basic run with defaults")
    
    # Example 2: Run with custom configuration file
    print("\n📋 Example 2: Run with custom configuration file")
    cmd2 = [
        sys.executable, "main_production.py",
        "--config", "config_production.json",
        "--model-type", "stacking_ensemble"
    ]
    success2 = run_command(cmd2, "Run with custom config")
    
    # Example 3: Run with command-line parameter overrides
    print("\n📋 Example 3: Run with command-line parameter overrides")
    cmd3 = [
        sys.executable, "main_production.py",
        "--epochs", "30",
        "--batch-size", "16",
        "--learning-rate", "0.0005",
        "--log-level", "DEBUG",
        "--force-cpu"
    ]
    success3 = run_command(cmd3, "Run with parameter overrides")
    
    # Example 4: Dry run to validate configuration
    print("\n📋 Example 4: Dry run to validate configuration")
    cmd4 = [
        sys.executable, "main_production.py",
        "--config", "config_production.json",
        "--dry-run",
        "--debug"
    ]
    success4 = run_command(cmd4, "Dry run validation")
    
    # Example 5: Run with custom output settings
    print("\n📋 Example 5: Run with custom output settings")
    cmd5 = [
        sys.executable, "main_production.py",
        "--output-dir", "custom_outputs",
        "--submission-filename", "my_submission.csv",
        "--log-file", "my_pipeline.log",
        "--structured-logging"
    ]
    success5 = run_command(cmd5, "Run with custom output settings")
    
    # Example 6: Run with performance optimizations
    print("\n📋 Example 6: Run with performance optimizations")
    cmd6 = [
        sys.executable, "main_production.py",
        "--enable-caching",
        "--memory-monitoring",
        "--batch-size", "64",
        "--model-type", "gcn"
    ]
    success6 = run_command(cmd6, "Run with performance optimizations")
    
    # Summary
    examples = [
        ("Basic run", success1),
        ("Custom config", success2),
        ("Parameter overrides", success3),
        ("Dry run", success4),
        ("Custom outputs", success5),
        ("Performance optimized", success6)
    ]
    
    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print(f"{'='*60}")
    
    successful = 0
    for name, success in examples:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{name:20} : {status}")
        if success:
            successful += 1
    
    print(f"\nTotal: {successful}/{len(examples)} examples completed successfully")
    
    if successful == len(examples):
        print("🎉 All examples completed successfully!")
        return 0
    else:
        print("⚠️  Some examples failed. Check the logs for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)