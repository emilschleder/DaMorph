from collections import OrderedDict
import re, json, morfessor, os, torch
from transformers import PreTrainedTokenizer
from huggingface_hub import hf_hub_download, HfApi
from tokenizers import Tokenizer, models, pre_tokenizers, decoders

class MorfessorBPETokenizer(PreTrainedTokenizer):
    def __init__(
        self, 
        morfessor_model_path: str = None, 
        vocab_file: str = None,
        max_length: int = 512,
        special_tokens: dict = None, 
        bpe_tokenizer_path: str = None, 
        **kwargs
    ):
        if morfessor_model_path:
            self.model = morfessor.MorfessorIO().read_binary_model_file(morfessor_model_path)
        else:
            self.model = None
        
        self.model_max_length = max_length
        
        # Pretrained tokenizer - initiated with vocab
        if vocab_file is not None:         
            _vocab = json.load(open(vocab_file, 'r'))
            self.vocab =  _vocab["vocab"]
            self.inv_vocab = {v: k for k, v in self.vocab.items()}
            self.morph_table = _vocab["morph_table"]
            self.special_tokens = _vocab["special_tokens"]

            # Load merges from vocab file
            self.merges = _vocab["merges"]
            # Convert merges from list of lists/strings to list of tuples if necessary
            if isinstance(self.merges[0], str):
                self.merges = [tuple(merge.split()) for merge in self.merges]  
            elif isinstance(self.merges[0], list):
                self.merges = [tuple(merge) for merge in self.merges]
        else:
            self.vocab = {}
            self.inv_vocab = {}
            self.morph_table = {}
            self.merges = []
            self.special_tokens = {
                "[UNK]": 0,
                "[PAD]": 1,
                "[CLS]": 2,
                "[SEP]": 3,
                "[MASK]": 4
            }

        # Load BPE tokenizer if provided
        if bpe_tokenizer_path:
            self.bpe_tokenizer = Tokenizer.from_pretrained(bpe_tokenizer_path)
        
        # Initialize BPE tokenizer from scratch if no path provided
        elif self.vocab:
            # Get first index that starts with '§BPE§' to determine where to start indexing BPE tokens
            self.bpe_cutoff = min([v for k, v in self.vocab.items() if k.startswith('§BPE§')], default=0)
            bpe_vocab = {k[5:]: v-self.bpe_cutoff for k, v in self.vocab.items() if k.startswith('§BPE§')}
         
            self.bpe_tokenizer = Tokenizer(
                models.BPE(vocab=bpe_vocab, merges=self.merges)
            )
            self.bpe_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
                add_prefix_space=False
            )
            self.bpe_tokenizer.decoder = decoders.ByteLevel()
        else: 
            self.bpe_tokenizer = None
            self.bpe_cutoff = 0

        super().__init__(
            **kwargs
        )
        
        self._pad_token = "[PAD]"
        self.pad_token_id = self.special_tokens.get("[PAD]", 1) 
        self._mask_token = "[MASK]"
        self.mask_token_id = self.special_tokens.get("[MASK]", 4)
        self._eos_token = "[SEP]"
        self.eos_token_id = self.special_tokens.get("[SEP]", 3)
    
    
    def get_vocab(self) -> dict:
        return self.vocab


    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
    
    
    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        elif len(tokens) > 0:
            return [self._convert_token_to_id(token) for token in tokens]
        else:
            return [self.special_tokens['[UNK]']]
     
        
    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.special_tokens['[UNK]'])


    def _convert_id_to_token(self, index: int) -> str:
        return self.inv_vocab.get(index, "[UNK]")
     
        
    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        elif isinstance(ids, torch.Tensor):
            ids = ids.tolist()
            if isinstance(ids, int):
                return self._convert_id_to_token(ids)
            elif isinstance(ids, list):
                return [self._convert_id_to_token(idx) for idx in ids]
            else:
                raise TypeError(f"Expected int or list after converting tensor to list, got {type(ids)}")
        elif isinstance(ids, list):
            return [self._convert_id_to_token(idx) for idx in ids]
        else:
            raise TypeError(f"Expected an int, list of ints, or tensor, got {type(ids)}")


    def _tokenize(self, text: str) -> list:
        return self.tokenize(text)
    
    
    def _morph_table_lookup(self, word: str) -> list:
        return self.morph_table.get(word, None)
   
        
    def encode(
        self, 
        word: str, 
        split_special_tokens=False, 
        add_special_tokens=False
    ):
        word_init = word.removeprefix(" ")
        if word_init in self.special_tokens:
            return [word_init.lstrip()]
        
        processed_morphemes = []

        # Morph table lookup
        morphemes = self._morph_table_lookup(word)
        if morphemes and isinstance(morphemes, list):
            processed_morphemes.extend(morphemes)

         # Otherwise, use Morfessor model
        elif self.model:
            morphemes = self.model.viterbi_segment(word)[0]
            tmp = []
            for morpheme in morphemes:
                if morpheme in self.vocab:
                    tmp.append(morpheme)
            if len(tmp) == len(morphemes):    
                processed_morphemes.extend(tmp)
                
            else:
                # BPE tokenization
                encoding = self.bpe_tokenizer.encode(
                    word, 
                    add_special_tokens=True
                ) 
                bpe_tokens = encoding.tokens
                processed_morphemes.extend(["§BPE§" + token for token in bpe_tokens])
        
        # Last resort if no model - UNK token
        else:
            processed_morphemes.append('[UNK]')
        
        # Add special tokens if requested
        if add_special_tokens:
            if "[CLS]" in self.special_tokens:
                token_ids = [self.special_tokens["[CLS]"]] + token_ids
            if "[SEP]" in self.special_tokens:
                token_ids = token_ids + [self.special_tokens["[SEP]"]]
                
        return processed_morphemes


    def _split_into_sentences(self, long_string) -> list:
        pattern = r'(?<=[.!?])(?=\s)'
        sentences = re.split(pattern, long_string)
        return sentences


    def tokenize(
        self, 
        text: str, 
        add_special_tokens: bool = False, 
        split_special_tokens=False
    ) -> list:
        
        sentences = self._split_into_sentences(text)
        tokenized_text = []
    
        for sentence in sentences: 
            # Add special token for beginning of sentence if specified
            if add_special_tokens and "[CLS]" in self.special_tokens:
                tokenized_text.append("[CLS]")
            
            # Split by whitespace and special characters into chunks
            chunk = re.split(r'(?=\s)', sentence)
            for c in chunk:
                morphemes = self.encode(c)
                word_tokens = []
                for morpheme in morphemes:
                    word_tokens.append(morpheme)
                tokenized_text.extend(word_tokens)
            
            # Add special token for end of sentence if specified
            if add_special_tokens and "[SEP]" in self.special_tokens:
                tokenized_text.append("[SEP]")
        
        return tokenized_text
    
     
    def decode(
        self, 
        token_ids: list, 
        skip_special_tokens: bool = True, 
        clean_up_tokenization_spaces: bool = False
    ) -> str:
 
        decoded_text = "" 
        decoded_tokens = []
        bpe_token_ids = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in self.special_tokens.values():
                continue
            if token_id >= self.bpe_cutoff:
                bpe_token_ids.append(token_id - self.bpe_cutoff)
            else:
                # Decode any collected BPE tokens before processing the Morfessor token
                if bpe_token_ids:
                    decoded_bpe = self.bpe_tokenizer.decode(bpe_token_ids)
                    decoded_tokens.append(decoded_bpe)
                    bpe_token_ids = []
                    
                # Then decode the Morfessor token
                decoded_morf = self.convert_ids_to_tokens(token_id)
                decoded_tokens.append(decoded_morf)

        # Decode any remaining BPE tokens 
        if bpe_token_ids:
            decoded_bpe = self.bpe_tokenizer.decode(bpe_token_ids)
            decoded_tokens.append(decoded_bpe)

        decoded_text = ''.join(decoded_tokens)
        return decoded_text

    def _build_morph_vocab(
        self, 
        morph_table_path: str, 
        morph_vocab_size: int
    ) -> dict:
        
        # Build a list of special tokens in order of their indices
        special_tokens_list = sorted(self.special_tokens.items(), key=lambda x: x[1])
        special_tokens_list = [token for token, index in special_tokens_list]
                
        # Build vocab_init as an ordered list
        vocab_init = special_tokens_list 
        new_morpheme_table = {}
        
        with open(morph_table_path, 'r') as f:
            morph_table = json.load(f, object_pairs_hook=OrderedDict)
            
        # OrderedDict to store unique segments
        vocab = OrderedDict()
        for token in vocab_init:
            vocab[token] = None  

        vocab_full = False
        # Loop through each word in the morpheme table
        for word, data in morph_table.items():
            
            segments = data['segment'] 

            if not vocab_full:
                # Check if adding these segments would exceed the max vocab size
                new_segments = [seg for seg in segments if seg not in vocab]
                if len(vocab) + len(new_segments) > morph_vocab_size:
                    print(f"Max vocabulary size of {morph_vocab_size} reached. Stopping.")
                    vocab_full = True
                    continue

                # Add the word and its segments to the new morpheme table
                new_morpheme_table[word] = segments

                # Add the segments to the vocabulary
                for seg in segments:
                    if seg not in vocab:
                        vocab[seg] = None 
            else:
                new_segments_in_vocab = [seg for seg in segments if seg in vocab]
                if len(new_segments_in_vocab) == len(segments):
                    new_morpheme_table[word] = segments
                
        # Now assign indices to tokens
        final_vocab = {}
        index = 0
        for token in vocab:
            if token in self.special_tokens:
                final_vocab[token] = self.special_tokens[token]
                index = max(index, self.special_tokens[token] + 1)
            else:
                final_vocab[token] = index
                index += 1
        
        result = {
            "vocab": final_vocab, 
            "morph_table": new_morpheme_table, 
            "special_tokens": self.special_tokens
        }
        return result
  
    
    def _extract_bpe_vocab(
        self, 
        bpe_tokenizer_path: str, 
        start_index: int
    ) -> dict:
        
        print(f"Loading BPE tokenizer from {bpe_tokenizer_path}")
        print(f"bpe_start_index: {start_index}")

        vocab_file = hf_hub_download(
            repo_id=bpe_tokenizer_path, 
            filename="tokenizer.json", 
        )
        with open(vocab_file, 'r') as f:
            bpe_file = json.load(f)
            bpe_vocab = bpe_file["model"]["vocab"]
            bpe_merges = bpe_file["model"]["merges"]
            
        # Get the list of tokens from the BPE tokenizer's vocabulary
        tokens = list(bpe_vocab.keys())
        print(f"Number of BPE tokens: {len(tokens)}")
        
        # Assign indices starting from start_index
        bpe_vocab = {}
        index = start_index
        
        for token in tokens:
            prefixed_token = f'§BPE§{token}'
            bpe_vocab[prefixed_token] = index
            index += 1
        
        result = {
            "vocab": bpe_vocab,
            "merges": bpe_merges
        }
        return result
    
    
    def build_vocab(
        self, 
        morph_table_path: str, 
        morph_vocab_size: int, 
        bpe_tokenizer_path: str 
    ) -> None:
        
        # Morpheme vocabulary
        morph_dict = self._build_morph_vocab(
            morph_table_path=morph_table_path, 
            morph_vocab_size=morph_vocab_size
        )
        morph_vocab = morph_dict["vocab"]
        morph_table = morph_dict["morph_table"]
        max_morph_index = max(morph_vocab.values())+1
        
        # BPE vocabulary
        bpe_dict = self._extract_bpe_vocab(
            bpe_tokenizer_path=bpe_tokenizer_path, 
            start_index=max_morph_index
        )
        bpe_vocab = bpe_dict["vocab"]
        bpe_merges = bpe_dict["merges"]
        
        print(f"Final Morpheme Table size: {len(morph_table)}")
        print(f"Final vocabulary size: {len(morph_vocab)}")
        
        # Combine the two vocabularies
        final_vocab = {}
        final_vocab.update(morph_vocab)
        final_vocab.update(bpe_vocab)

        # Setting the final vocab, merges and morf table
        self.vocab = final_vocab
        self.merges = bpe_merges
        self.morph_table = morph_table
        print(f"Final vocabulary size: {len(final_vocab)}")


    def save_vocabulary(
        self, 
        save_directory: str,
        push_to_hub: bool = False, 
        hf_repo_name: str = None,  
        filename_prefix: str = "",
        morfessor_model_path: str = None,
        token: str = None
    ) -> None: 
        
        vocab_file = os.path.join(
            save_directory, 
            filename_prefix + "vocab.json"
        )
       
        with open(vocab_file, 'w') as f:
            obj = {
                "vocab": self.vocab,
                "morph_table": self.morph_table,
                "merges": self.merges,
                "special_tokens": self.special_tokens
            }
            json.dump(obj, f)
            
        print(f"Vocabulary saved to {vocab_file}")
        
        if push_to_hub:
            # Authenticate with Hugging Face Hub
            if token is None:
                raise ValueError("You need to provide a valid token to push to the Hugging Face Hub.")
            if hf_repo_name:
                api = HfApi()
                if not api.repo_exists(repo_id=hf_repo_name):
                    api.create_repo(
                        repo_id=hf_repo_name, 
                        private=True, 
                        token=token
                    )
                    print(f"Repository {hf_repo_name} created")
                else:
                    print(f"Repository {hf_repo_name} already exists")
                
                if vocab_file:
                    api.upload_file(
                        repo_id=hf_repo_name,
                        path_or_fileobj=vocab_file,
                        path_in_repo="vocab.json",
                    )
                if morfessor_model_path:
                    api.upload_file(
                        repo_id=hf_repo_name,
                        path_or_fileobj=morfessor_model_path,
                        path_in_repo="morfessor_model.bin",
                    )
                print(f"Tokenizer uploaded to: https://huggingface.co/{hf_repo_name}")
            else:
                raise ValueError("No repo_name specified.")


    def push_to_hub(
        self, 
        repo_name: str, 
        token: str,
        morfessor_model_path: str = None
    ):
        self.save_vocabulary(
            save_directory=".",
            push_to_hub=True,
            hf_repo_name=repo_name,
            morfessor_model_path=morfessor_model_path,
            token=token
        )
        
    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path: str, 
        bpe_tokenizer_path: str = None, 
        morfessor_model: bool = None, 
        **kwargs
    ):
        cache_dir = kwargs.get("cache_dir", None)


        if pretrained_model_name_or_path:
            vocab_file = hf_hub_download(
                repo_id=pretrained_model_name_or_path, 
                filename="vocab.json", 
                cache_dir=cache_dir
            )
        else: 
            vocab_file = None

        if morfessor_model:            
            morfessor_model_file = hf_hub_download(
                repo_id=pretrained_model_name_or_path, 
                filename="morfessor_model.bin", 
                cache_dir=cache_dir
            )
        else:
            morfessor_model_file = None
            
        # Instantiate the tokenizer
        tokenizer = cls(
            vocab_file=vocab_file,
            bpe_tokenizer_path=bpe_tokenizer_path,
            morfessor_model_path=morfessor_model_file,
            **kwargs
        )
        return tokenizer


    def save_pretrained(self, save_directory: str) -> None: 
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        vocab_file = os.path.join(save_directory, "vocab.json")
        special_tokens_file = os.path.join(save_directory, "special_tokens.json")

        with open(vocab_file, 'w') as f:
            json.dump(self.vocab, f)

        with open(special_tokens_file, 'w') as f:
            json.dump(self.special_tokens, f)

        print(f"Tokenizer saved to {save_directory}")