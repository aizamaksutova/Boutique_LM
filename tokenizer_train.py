import argparse
import glob
import json
import os
import random
from typing import List
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import requests
import sentencepiece as spm
import torch
import torch.distributed as dist
from tqdm import tqdm
from pathlib import Path

# from tokenizer import Tokenizer

DATA_DIR = "data"
TOK_DIR = 'vocab'

def train_tokenizer(vocab_size, num_samples=15):
    
    output_path_tok = os.path.join(TOK_DIR, f"tokenizer_{vocab_size}")
    all_data_dir = os.path.join(DATA_DIR, "")
    train_text = os.path.join(DATA_DIR, "token_text.txt")
    train_file = Path(train_text)

    filenames = sorted(glob.glob(os.path.join(all_data_dir, "*.json")))

    print("Writing the json data to our little txt file")
    if train_file.is_file() == False:
        with open(train_text, "w", encoding="utf-8") as f:
            for input_file in tqdm(filenames[:num_samples]):
                with open(input_file, "r") as file:
                    data_i = json.load(file)
                for sample in data_i:
                    story = sample['story']
                    story = story.strip()
                    f.write(story + '/n')
            
        print('Writing to file done - got the vocabulary for training the tokenizer')

    if train_file.is_file():
        print('File for tokenizer training exists')


    # Training the sentence piece model, here are the templates for it https://github.com/google/sentencepiece/tree/master/python
    
    spm.SentencePieceTrainer.train(
        input=train_text, 
        model_prefix=output_path_tok, 
        vocab_size=vocab_size, 
        character_coverage=1.0,
        model_type='bpe',
        
    )

    print('Finished training, the tokenizer is in file {}')

def process_batch(args, vocab_size):
    batch_id, batch = args
    model_tokenizer = os.path.join(TOK_DIR, f"tokenizer_{vocab_size}.model")
    all_data_dir = os.path.join(DATA_DIR, "")

    encoder = spm.SentencePieceProcessor(model_file=model_tokenizer)
    with open(batch, 'r') as file:
        data = json.load(file)
    
    encoded_tokens = []

    for sample in tqdm(data, position=batch_id):
        story = sample['story']
        story = story.strip()
        tokens = encoder.encode(story, out_type=int)
        encoded_tokens.extend(tokens)
    
    encoded_tokens = np.array(encoded_tokens, dtype=np.uint16)
    bin_directory = os.path.join(TOK_DIR)

    batch_basename = os.path.basename(batch)
    bin_basename = batch_basename.replace(".json", ".bin")
    tokenized_filename = os.path.join(bin_directory, bin_basename)

    with open(tokenized_filename, "wb") as f:
        f.write(encoded_tokens.tobytes())
    
    avg_seq_len = extended_tokens.size / ((extended_tokens == 1).sum())
    print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")
    
def pre_tokenize(vocab_size):
    data_dir = os.path.join(DATA_DIR, "")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    bin_dir = os.path.join(TOK_DIR)
    os.makedirs(bin_dir, exist_ok=True)

    # process all the shards in a process pool
    batch = partial(process_batch, vocab_size=vocab_size)
    with ProcessPoolExecutor() as executor:
        executor.map(batch, enumerate(shard_filenames))
    print("Done.")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=0, help="Tokenizer vocab size")
    parser.add_argument("--num_samples", type=int, default=10, help="Num samples for tokenizer training")
    args = parser.parse_args()

    
    train_tokenizer(vocab_size=args.vocab_size, num_samples=args.num_samples)
    pre_tokenize(vocab_size=args.vocab_size)







