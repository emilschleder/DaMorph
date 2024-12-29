import os

directory = '_5_Analysis'

output_file = os.path.join(directory, 'results.csv')
files = [file for file in os.listdir(directory) if file.endswith('_test.out.txt')]

for file in files:
    
    with open(os.path.join(directory, file), 'r') as f:
        content = f.readlines()
        model = file.split('.')[0]
        
        for i, line in enumerate(content):
            
            if line.startswith('Prediction:'):
                
                if model == 'DA-MORPH-CEREBRAS':
                    sentence = 'Sætning: ' + content[i-1].strip()
                else:    
                    sentence = content[i-2].strip()
                    
                prediction = content[i].strip().split(': ')[1]
                label = content[i+1].strip().split(': ')[1]
                
                with open(output_file, 'a') as csvfile:
                    csvfile.write(f"{model}|{sentence}|{prediction}|{label}\n")
                i = i + 2
            
            else:
                continue
            