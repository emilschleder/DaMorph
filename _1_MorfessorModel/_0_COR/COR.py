import argparse

parser = argparse.ArgumentParser(description='Process and clean a file.')
parser.add_argument('--input_file', type=str, help='The input file to process')
parser.add_argument('--output_file', type=str, help='The final output file')

args = parser.parse_args()
input_file = args.input_file
output_file = args.output_file
    
def process_file(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            data = infile.readlines()
        
        extracted_words = []
        for line in data:
            parts = line.split("\t")
            if len(parts) > 4:
                word1 = parts[1].strip()
                word4 = parts[4].strip()
                
                if word1.startswith('-'):
                    word1 = word1[1:]
                if word4.startswith('-'):
                    word4 = word4[1:] 
                    
                extracted_words.append(word1)
                extracted_words.append(word4)
        
        unique_words = sorted(set(extracted_words))
        cleaned_words = [word for word in unique_words if ' ' not in word]
        
        with open(output_file, 'w', encoding='utf-8') as outfile:
            outfile.write('\n'.join(cleaned_words))
        
        print(f"Output: {output_file}")
    
    except Exception as e:
        print(f"Error: {e}")

def main():
    process_file(input_file, output_file)

if __name__ == '__main__':
    main()