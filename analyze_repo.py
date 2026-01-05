# In Bolt, when user submits a repo
def analyze_repo(github_url):
    # 1. Clone the repo
    subprocess.run(['git', 'clone', github_url, '/tmp/user_repo'])
    
    # 2. Extract metrics (same as training)
    for py_file in Path('/tmp/user_repo').rglob('*.py'):
        metrics = get_metrics(py_file)  # Same radon extraction
        
        # 3. Predict with trained model
        prediction = model.predict([metrics])
        
        # 4. Return results
        results.append({
            'file': py_file.name,
            'defect_probability': prediction,
            'complexity': metrics['complexity'],
            'loc': metrics['loc']
        })