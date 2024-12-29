from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from _2_Tokenizer.DaMorph_tokenizers.DaMorph_raw import MorfessorTokenizer
from _2_Tokenizer.DaMorph_tokenizers.DaMorph_mixed import MorfessorBPETokenizer 
import pandas as pd 
import argparse
from models import models  
from prompts import prompts

parser = argparse.ArgumentParser(description="Generate text continuations using various models and tokenizers.")
parser.add_argument('--output_file', type=str, default="generated_continuations.csv", help='Output file to save generated continuations')
args = parser.parse_args()

output_file = args.output_file

def predict(prompt, tokenizer, model, device, model_name, tokenizer_name, generation_params):
   
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    # Set pad_token_id and eos_token_id in the model if not already set
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id

    # Generate text with category-specific parameters
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=200,
        do_sample=True,
        no_repeat_ngram_size=3,
        top_k=generation_params['top_k'],
        top_p=generation_params['top_p'],
        temperature=generation_params['temperature'],
    )

    generated_text = tokenizer.decode(
        output[0], 
        skip_special_tokens=True
    )
    continuation = generated_text[len(prompt):].strip()
    
    data = {
        'Model': model_name,
        'Tokenizer': tokenizer_name,
        'Category': generation_params['category'],
        'Prompt': prompt,
        'Continuation': continuation,
        'Top_K': generation_params['top_k'],
        'Top_P': generation_params['top_p'],
        'Temperature': generation_params['temperature']
    }
    return data

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    all_data = []
    for model_info in models:
        model_name = model_info['name']
        tokenizer_name = model_info['tokenizer']
        
        try:
            # Load tokenizer based on tokenizer type
            if 'MORPH' in tokenizer_name.upper():
                tokenizer = MorfessorTokenizer.from_pretrained(
                    tokenizer_name, 
                    morfessor_model=True
                )
            elif 'MIXED' in tokenizer_name.upper():
                tokenizer = MorfessorBPETokenizer.from_pretrained(
                    tokenizer_name,
                    morfessor_model=True
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name
                )
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            model.eval()
            
        except Exception as e:
            print(f"Error loading model/tokenizer for {model_name}: {e}")
            continue 

        print(f"\n=== Evaluating Model: {model_name} ===\n")
        
        for prompt in prompts:
            prompt_text = prompt['text']
            generation_params = prompt['generation_params']

            data = predict(
                prompt=prompt_text,
                tokenizer=tokenizer,
                model=model,
                device=device,
                model_name=model_name,
                tokenizer_name=tokenizer_name,
                generation_params=generation_params
            )
            all_data.append(data)
    
    df = pd.DataFrame(all_data)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\nAll generated continuations have been saved to '{output_file}'.")

if __name__ == "__main__":
    main()
