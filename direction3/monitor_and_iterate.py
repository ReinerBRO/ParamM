#!/usr/bin/env python3
"""
Continuous monitoring and iteration for Direction 3 pipeline.
Implements Codex supervisor pattern: actively monitor, validate, iterate.
"""
import subprocess
import time
import os
import sys

def log(msg):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

def check_training_status():
    """Check if training is still running."""
    result = subprocess.run(['ps', '-p', '1361157'], capture_output=True)
    return result.returncode == 0

def get_training_progress():
    """Extract latest training progress from log."""
    try:
        with open('router_training.log', 'r') as f:
            lines = f.readlines()
            
        # Find last epoch
        for line in reversed(lines[-100:]):
            if 'Epoch' in line and '/30' in line:
                epoch = line.strip()
                break
        else:
            epoch = "Unknown"
        
        # Find last best model
        for line in reversed(lines[-100:]):
            if 'best model' in line.lower():
                best = line.strip()
                break
        else:
            best = "No best model yet"
        
        return epoch, best
    except:
        return "Unknown", "Unknown"

def check_checkpoint():
    """Check if checkpoint was created."""
    ckpt_path = "router_checkpoints/transformer_mbpp_v1/router.ckpt"
    if os.path.exists(ckpt_path):
        size = os.path.getsize(ckpt_path) / (1024 * 1024)
        return True, f"{size:.2f} MB"
    return False, None

def check_validation_jobs():
    """Check if validation jobs are running."""
    result = subprocess.run(['squeue', '-u', 'user06', '-h'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        return len([l for l in lines if l.strip()])
    return 0

def find_latest_results():
    """Find latest validation results."""
    try:
        result = subprocess.run(
            ['find', 'results/mbpp_router_runs', '-name', 'results.txt', '-type', 'f'],
            capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            files = result.stdout.strip().split('\n')
            # Get most recent
            latest = max(files, key=os.path.getmtime)
            return latest
    except:
        pass
    return None

def extract_phase2_acc(results_file):
    """Extract phase2_acc from results file."""
    try:
        with open(results_file, 'r') as f:
            for line in f:
                if 'phase2_acc' in line:
                    # Extract number
                    import re
                    match = re.search(r'(\d+\.\d+)', line)
                    if match:
                        return float(match.group(1))
    except:
        pass
    return None

def main():
    log("=== Continuous Supervisor Started ===")
    log("Implementing Codex supervisor pattern")
    log("")
    
    # Phase 1: Monitor training
    log("Phase 1: Monitoring training...")
    while check_training_status():
        epoch, best = get_training_progress()
        log(f"Training: {epoch}")
        if "best model" in best.lower():
            log(f"  {best}")
        time.sleep(60)
    
    log("Training completed!")
    log("")
    
    # Phase 2: Verify checkpoint
    log("Phase 2: Verifying checkpoint...")
    time.sleep(5)
    
    has_ckpt, size = check_checkpoint()
    if has_ckpt:
        log(f"Checkpoint found: {size}")
    else:
        log("ERROR: Checkpoint not found!")
        return 1
    
    # Phase 3: Wait for auto-integration and validation launch
    log("Phase 3: Waiting for validation launch...")
    
    # Wait up to 5 minutes for job to appear
    for i in range(30):
        jobs = check_validation_jobs()
        if jobs > 0:
            log(f"Validation jobs launched: {jobs} active")
            break
        time.sleep(10)
    else:
        log("WARNING: No validation jobs found after 5 minutes")
        return 2
    
    # Phase 4: Monitor validation
    log("Phase 4: Monitoring validation run...")
    
    while check_validation_jobs() > 0:
        jobs = check_validation_jobs()
        log(f"Validation running: {jobs} jobs active")
        time.sleep(120)
    
    log("Validation completed!")
    log("")
    
    # Phase 5: Check results
    log("Phase 5: Checking results...")
    time.sleep(5)
    
    results_file = find_latest_results()
    if not results_file:
        log("ERROR: No results file found")
        return 3
    
    log(f"Results file: {results_file}")
    
    phase2_acc = extract_phase2_acc(results_file)
    if phase2_acc is None:
        log("ERROR: Could not extract phase2_acc")
        return 4
    
    log(f"phase2_acc: {phase2_acc}%")
    log("")
    
    # Phase 6: Validate against target
    TARGET = 75.31
    if phase2_acc > TARGET:
        log(f"SUCCESS: Target achieved! ({phase2_acc}% > {TARGET}%)")
        log("Direction 3 validation passed!")
        return 0
    else:
        log(f"Target not达标: {phase2_acc}% <= {TARGET}%")
        log("Architectural iteration needed")
        log("Next: Delegate to Codex for improvements")
        return 10  # Special code for iteration needed

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        log("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        log(f"ERROR: {e}")
        sys.exit(1)
