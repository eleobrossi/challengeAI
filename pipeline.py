#!/usr/bin/env python3
"""
Unified Data Pipeline Orchestrator

This script runs the complete data processing pipeline:
1. Create unified dataset from raw JSON/CSV files
2. Enrich transactions with location data
3. Compute transaction statistics
4. Run fraud detection analysis

Starting from raw data in The Truman Show_train/public folder
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
import json

class PipelineOrchestrator:
    def __init__(self):
        self.workspace_dir = Path(__file__).parent
        self.scripts = [
            'create_unified_dataset.py',
            'enrich_transactions.py',
            'compute_transaction_stats.py',
            'data_analysis_enhanced.py',
            'generate_suspicious_list.py'
        ]
        self.step_names = [
            'Unified Dataset Creation',
            'Transaction Enrichment',
            'Statistical Computation',
            'Fraud Detection Analysis',
            'Suspicious List Generation'
        ]
        self.start_time = None
        self.results = {}
        
    def log(self, message, level='INFO'):
        """Print formatted log message"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] [{level}] {message}")
    
    def run_script(self, script_name: str, step_number: int, step_name: str) -> bool:
        """Execute a single pipeline script"""
        script_path = self.workspace_dir / script_name
        
        if not script_path.exists():
            self.log(f"Script not found: {script_path}", 'ERROR')
            return False
        
        self.log(f"Step {step_number}: {step_name}", 'INFO')
        self.log(f"Executing: {script_name}", 'INFO')
        
        try:
            # Run the script using the current Python interpreter
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=self.workspace_dir,
                capture_output=False,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                self.log(f"✓ {step_name} completed successfully", 'SUCCESS')
                self.results[script_name] = 'SUCCESS'
                return True
            else:
                self.log(f"✗ {step_name} failed with exit code {result.returncode}", 'ERROR')
                self.results[script_name] = f'FAILED (exit code: {result.returncode})'
                return False
                
        except Exception as e:
            self.log(f"✗ Error executing {script_name}: {str(e)}", 'ERROR')
            self.results[script_name] = f'ERROR: {str(e)}'
            return False
    
    def verify_outputs(self) -> dict:
        """Check if expected output files exist"""
        expected_files = {
            'create_unified_dataset.py': [
                'unified_dataset_complete.json',
                'user_profiles.csv'
            ],
            'enrich_transactions.py': [
                'transactions_enriched_with_locations.csv'
            ],
            'compute_transaction_stats.py': [
                'transactions_with_stats.csv'
            ],
            'data_analysis_enhanced.py': [
                'transactions_flagged.csv',
                'fraud_analysis_detailed.json'
            ],
            'generate_suspicious_list.py': [
                'suspicious.txt'
            ]
        }
        
        verification = {}
        for script, files in expected_files.items():
            verification[script] = {}
            for file in files:
                file_path = self.workspace_dir / file
                exists = file_path.exists()
                verification[script][file] = exists
                status = '✓' if exists else '✗'
                self.log(f"  {status} {file}", 'INFO' if exists else 'WARN')
        
        return verification
    
    def run_full_pipeline(self):
        """Execute the complete pipeline"""
        self.start_time = datetime.now()
        self.log("=" * 70, 'INFO')
        self.log("PIPELINE EXECUTION STARTED", 'INFO')
        self.log(f"Working Directory: {self.workspace_dir}", 'INFO')
        self.log("=" * 70, 'INFO')
        print()
        
        failed_steps = []
        
        # Execute each step
        for i, (script, step_name) in enumerate(zip(self.scripts, self.step_names), 1):
            success = self.run_script(script, i, step_name)
            if not success:
                failed_steps.append(f"Step {i}: {step_name}")
            print()
        
        # Verify outputs
        self.log("Verifying output files...", 'INFO')
        print()
        verification = self.verify_outputs()
        print()
        
        # Summary
        self.log("=" * 70, 'INFO')
        self.log("PIPELINE EXECUTION SUMMARY", 'INFO')
        self.log("=" * 70, 'INFO')
        
        elapsed_time = datetime.now() - self.start_time
        self.log(f"Execution Time: {elapsed_time.total_seconds():.2f} seconds", 'INFO')
        print()
        
        self.log("Step Results:", 'INFO')
        for i, (script, step_name) in enumerate(zip(self.scripts, self.step_names), 1):
            status = self.results.get(script, 'NOT RUN')
            symbol = '✓' if status == 'SUCCESS' else '✗'
            self.log(f"  {symbol} Step {i}: {step_name} - {status}", 'INFO')
        
        print()
        
        if failed_steps:
            self.log(f"FAILED: {len(failed_steps)} step(s) did not complete successfully", 'ERROR')
            for step in failed_steps:
                self.log(f"  - {step}", 'ERROR')
            print()
            self.log("=" * 70, 'INFO')
            return False
        else:
            self.log("✓ PIPELINE COMPLETED SUCCESSFULLY", 'SUCCESS')
            self.log("All steps executed and outputs generated", 'SUCCESS')
            self.log("=" * 70, 'INFO')
            return True

def main():
    """Main entry point"""
    try:
        orchestrator = PipelineOrchestrator()
        success = orchestrator.run_full_pipeline()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
