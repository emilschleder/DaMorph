from template import template

def generate_job_name(model_name: str) -> str:
    name = "sv_"
    if 'LLAMA' in model_name:
        name += 'll'
    if 'CEREBRAS' in model_name:
        name += 'ce'
    if 'STD' in model_name:
        name += '_std'
    elif 'BPE' in model_name:
        name += '_bpe'
    elif 'MIXED' in model_name:
        name += '_mix'
    elif 'MORPH' in model_name:
        name += '_mor'
    return name
    
def generate_jobs(
    tasks: list[str], 
    env: str, 
    hf_token: str,
    eval_models: str,
    verbose: bool = False,
) -> list[str]:    

    scripts = []
    # Read models and job names from the text file
    with open(eval_models, 'r') as f:
        for line in f:
            
            # Skiping empty lines and comments
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('--'):
                continue
            
            model_name = line
            job_name = generate_job_name(model_name)
            print(f'Processing model {model_name} with job name {job_name}')
            
            # Generating tasks list
            if tasks:
                tasks_list = ' '.join([f'--task {task}' for task in tasks])
            else:
                tasks_list = ''
            print(f'Tasks: {tasks_list}')

            # Printing all output if verbose
            if verbose:
                verbose_list = '--verbose --debug'
            else:
                verbose_list = ''
            print(f'Verbose: {verbose_list}')
                
            # Replace placeholders in the template
            script_content = template.format(
                model_name=model_name, 
                job_name=job_name,
                tasks=tasks_list,
                env=env,
                hf_token=hf_token,
                verbose=verbose_list
            )

            # Creating script file
            script_filename = f'_4_Evaluation/scandeval/jobs/{job_name}.sh'
            with open(script_filename, 'w') as script_file:
                script_file.write(script_content)
            
            print(f'Generated script {script_filename}')
            scripts.append(script_filename)
        
    return scripts