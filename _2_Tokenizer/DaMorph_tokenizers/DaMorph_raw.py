from collections import OrderedDict
import re, json, morfessor, os, sys, torch, logging
from transformers import PreTrainedTokenizer
from huggingface_hub import hf_hub_download, HfApi

class MorfessorTokenizer(PreTrainedTokenizer):
       
    def __init__(
        self, 
        morfessor_model_path: str = None, 
        vocab_file: str = None, 
        max_length: int = 512,
        special_tokens: dict = None,
        **kwargs
    ):
        if morfessor_model_path:
            self.model = morfessor.MorfessorIO().read_binary_model_file(morfessor_model_path)
        else:
            self.model = None
        
        self.model_max_length = max_length

        if vocab_file is not None:         
            _vocab = json.load(open(vocab_file, 'r'))
            self.vocab = _vocab["vocab"]
            self.inv_vocab = {v: k for k, v in self.vocab.items()}
            self.morph_table = _vocab["morph_table"]
            self.special_tokens = _vocab["special_tokens"]
        else:
            self.vocab = {}
            self.inv_vocab = {}
            self.morph_table = {}
            self.special_tokens = {
                "[UNK]": 0,
                "[PAD]": 1,
                "[CLS]": 2,
                "[SEP]": 3,
                "[MASK]": 4
            }
        
        super().__init__(
            **kwargs
        )
        self._pad_token = "[PAD]"
        self.pad_token_id = self.special_tokens.get("[PAD]", 1)       
        self._mask_token = "[MASK]"
        self.mask_token_id = self.special_tokens.get("[MASK]", 4) 
        
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

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.special_tokens['[UNK]'])

    def _convert_id_to_token(self, index):
        return self.inv_vocab.get(index, "[UNK]")
        
    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        elif isinstance(ids, torch.Tensor):
            ids = ids.tolist()
            return [self._convert_id_to_token(idx) for idx in ids]
        elif isinstance(ids, list):
            return [self._convert_id_to_token(idx) for idx in ids]
        else:
            raise TypeError(f"Expected an int, list of ints, or tensor, got {type(ids)}")

    def _tokenize(self, text):
        return self.tokenize(text)
    
    def _morph_table_lookup(self, word):
        return self.morph_table.get(word, None)
        
    def encode(self, word: str, split_special_tokens=False, add_special_tokens=False):
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
            for morpheme in morphemes:
                if morpheme in self.vocab:
                    processed_morphemes.append(morpheme)
                else:
                    processed_morphemes.append('[UNK]')
        
        # Last resort - UNK token
        else:
            processed_morphemes.append('[UNK]')
        
        # Ensure at least one morpheme is returned
        if not processed_morphemes:
            processed_morphemes.append('[UNK]')
        
        # Add special tokens if requested
        if add_special_tokens:
            if "[CLS]" in self.special_tokens:
                token_ids = [self.special_tokens["[CLS]"]] + token_ids
            if "[SEP]" in self.special_tokens:
                token_ids = token_ids + [self.special_tokens["[SEP]"]]
                
        return processed_morphemes

    def _split_into_sentences(self, long_string):
        pattern = r'(?<=[.!?])(?=\s)'
        sentences = re.split(pattern, long_string)
        return sentences

    def tokenize(
        self, 
        text: str, 
        add_special_tokens: bool = False, 
        split_special_tokens=False
    ):
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
        token_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    ):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        tokens = self.convert_ids_to_tokens(token_ids)
        
        # Filter out special tokens if requested
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.special_tokens]
        
        text = ''.join(tokens)
        
        if clean_up_tokenization_spaces:
            text = self._clean_up_tokenization(text)
            
        return text

    def _clean_up_tokenization(self, text):
        return text.strip()
    
    def _build_morph_vocab(
        self, 
        morph_table_path: str, 
        morph_vocab_size: int, 
    ):
        # Build a list of special tokens in order of their indices
        special_tokens_list = sorted(self.special_tokens.items(), key=lambda x: x[1])
        special_tokens_list = [token for token, index in special_tokens_list]

        vocab_init = special_tokens_list
        new_morpheme_table = {}
        
        with open(morph_table_path, 'r') as f:
            morph_table = json.load(f, object_pairs_hook=OrderedDict)
            
        # OrderedDict to store unique segments
        vocab = OrderedDict()
        for token in vocab_init:
            vocab[token] = None  

        vocab_full = False

        for word, data in morph_table.items():
            segments = data['segment']  # Extract the segments of the word

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
                
        final_vocab = {}
        index = 0
        for token in vocab:
            if token in self.special_tokens:
                final_vocab[token] = self.special_tokens[token]
                index = max(index, self.special_tokens[token] + 1)
            else:
                final_vocab[token] = index
                index += 1

        return {"vocab": final_vocab, "morph_table": new_morpheme_table, "special_tokens": self.special_tokens}
    
    def build_vocab(
        self, 
        morph_table_path, 
        morph_vocab_size
    ):
        morph_dict = self._build_morph_vocab(
            morph_table_path=morph_table_path, 
            morph_vocab_size=morph_vocab_size
        )
        morph_vocab = morph_dict["vocab"]
        morph_table = morph_dict["morph_table"]

        self.vocab = morph_vocab
        self.morph_table = morph_table
        print(f"Final Morpheme Table size: {len(morph_table)}")
        print(f"Final vocabulary size: {len(morph_vocab)}")

    def save_vocabulary(
        self, 
        save_directory: str, 
        push_to_hub: bool = False, 
        hf_repo_name: str = None, 
        filename_prefix: str = "", 
        morfessor_model_path: str = None,
        token: str = None
    ) -> None: 
        
        vocab_file = os.path.join(save_directory, filename_prefix + "vocab.json")
        
        with open(vocab_file, 'w') as f:
            obj = { 
                "vocab": self.vocab,
                "morph_table": self.morph_table,
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


    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path: str, 
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
            morfessor_model_path=morfessor_model_file,
            vocab_file=vocab_file,
            **kwargs
        )

        return tokenizer

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
        
    def save_pretrained(self, save_directory): 
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        vocab_file = os.path.join(save_directory, "vocab.json")
        morfessor_model_file = os.path.join(save_directory, "morfessor_model.bin")
        special_tokens_file = os.path.join(save_directory, "special_tokens.json")

        with open(vocab_file, 'w') as f:
            json.dump(self.vocab, f)

        io = morfessor.MorfessorIO()
        io.write_binary_model_file(morfessor_model_file, self.model)

        with open(special_tokens_file, 'w') as f:
            json.dump(self.special_tokens, f)