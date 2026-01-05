import subprocess
import json
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def get_metrics(file_path):
    """Extract radon metrics from a Python file"""
    try:
        result = subprocess.run(['radon', 'cc', str(file_path), '-j'], 
                              capture_output=True, text=True, timeout=5)
        if result.stdout:
            data = json.loads(result.stdout)
            return {
                'complexity': data.get('complexity', 0),
                'loc': data.get('loc', 0),
            }
    except:
        return None

def process_single_bug(project, bug_id, workspace):
    """Process one bug - designed for parallel execution"""
    bug_path = f"{workspace}/{project}_bug{bug_id}"
    
    try:
        # Checkout buggy version
        checkout = subprocess.run([
            'bugsinpy-checkout', 
            '-p', project,
            '-v', '0',
            '-i', str(bug_id),
            '-w', bug_path
        ], capture_output=True, timeout=120)
        
        if checkout.returncode != 0:
            return []
        
        # Extract metrics from Python files
        results = []
        py_files = list(Path(bug_path).rglob('*.py'))[:5]
        
        for py_file in py_files:
            metrics = get_metrics(py_file)
            if metrics:
                results.append({
                    'project': project,
                    'bug_id': bug_id,
                    'file_name': py_file.name,
                    **metrics,
                    'defect': 1
                })
        
        # Cleanup
        subprocess.run(['rm', '-rf', bug_path], timeout=30)
        
        return results
        
    except Exception as e:
        # Cleanup on error
        subprocess.run(['rm', '-rf', bug_path], capture_output=True)
        return []

# ============================================================================
# MAIN EXECUTION - FULL PARALLEL MODE
# ============================================================================

workspace = '/home/workspace'

# Define projects and number of bugs to extract
projects_config = {
    'pandas': 30,
    'keras': 20,
    'scrapy': 20,
    'black': 15,
    'fastapi': 15,
}

# Build list of all tasks
all_tasks = [
    (project, bug_id) 
    for project, num_bugs in projects_config.items()
    for bug_id in range(1, num_bugs + 1)
]

total_bugs = len(all_tasks)
print("=" * 70)
print(f"🚀 PARALLEL BUG EXTRACTION")
print("=" * 70)
print(f"Total bugs to process: {total_bugs}")
print(f"Projects: {list(projects_config.keys())}")
print(f"Parallel workers: 20 (adjust based on your Mac's performance)")
print("=" * 70)

# Process ALL bugs in parallel
all_data = []
start_time = time.time()
completed = 0
errors = 0

with ThreadPoolExecutor(max_workers=20) as executor:
    # Submit all tasks
    futures = {
        executor.submit(process_single_bug, proj, bug_id, workspace): (proj, bug_id)
        for proj, bug_id in all_tasks
    }
    
    # Process results as they complete
    for future in as_completed(futures):
        proj, bug_id = futures[future]
        try:
            results = future.result()
            all_data.extend(results)
            completed += 1
            
            # Progress update
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total_bugs - completed) / rate if rate > 0 else 0
            
            print(f"✅ {proj:12} bug {bug_id:3} | {len(results):2} files | "
                  f"Progress: {completed}/{total_bugs} ({completed*100//total_bugs}%) | "
                  f"ETA: {eta/60:.1f}m")
            
            # Save progress every 20 bugs
            if completed % 20 == 0:
                df_temp = pd.DataFrame(all_data)
                df_temp.to_csv(f'{workspace}/dataset_progress.csv', index=False)
                print(f"💾 Progress saved: {len(all_data)} examples")
                
        except Exception as e:
            errors += 1
            print(f"❌ {proj:12} bug {bug_id:3} | Error: {str(e)[:50]}")

# Final save
elapsed_time = time.time() - start_time
df = pd.DataFrame(all_data)
df.to_csv(f'{workspace}/bugsinpy_dataset.csv', index=False)

print("\n" + "=" * 70)
print(f"✅ EXTRACTION COMPLETE!")
print("=" * 70)
print(f"Total time: {elapsed_time/60:.1f} minutes")
print(f"Bugs processed: {completed}/{total_bugs}")
print(f"Errors: {errors}")
print(f"Total examples: {len(df)}")
print(f"Examples per project:")
print(df['project'].value_counts())
print(f"\nSaved to: {workspace}/bugsinpy_dataset.csv")
print("=" * 70)

# Quick stats
print("\n📊 Dataset Preview:")
print(df.head(10))
print(f"\nShape: {df.shape}")
print(f"Columns: {list(df.columns)}")