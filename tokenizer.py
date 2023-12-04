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
TOKENIZED = 'tokenized_texts'

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
        model_type='bpe',
        normalization_rule_name = "nmt_nfkc",
        pad_id = 0,
        unk_id = 3,
    )

    print('Finished training, the tokenizer is in file {}')

def process_batch(args, vocab_size):

    batch_id, batch = args

    # model_tokenizer = os.path.join(TOK_DIR, f"tokenizer_{vocab_size}.model")

    with open(batch, 'r') as file:
        data = json.load(file)
    
    tokens_directory = os.path.join(TOKENIZED)
    batch_basename = os.path.basename(batch)
    batch_basename = batch_basename.replace(".json", ".txt")
    tokenized_filename = os.path.join(tokens_directory, batch_basename)

    training_data = ''
    for sample in tqdm(data, position=batch_id):
        story = sample['story'].replace('\n', ' ')
        training_data += story

    with open(tokenized_filename, 'w') as file:
    # Write the text to the file
        file.write(training_data)
    

    print(f"Saved {tokenized_filename}")
    

def pre_tokenize(vocab_size):
    data_dir = os.path.join(DATA_DIR, "")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    bin_dir = os.path.join(TOKENIZED)
    os.makedirs(bin_dir, exist_ok=True)

    # process all the shards in a process pool
    batch = partial(process_batch, vocab_size=vocab_size)

    with ProcessPoolExecutor() as executor:
        executor.map(batch, enumerate(shard_filenames))
    print("Done.")


def join_all_files(text_path, dir_path):
    directory = dir_path

    output_file_name = os.path.join(text_path, "df_small.txt")


    with open(output_file_name, 'w') as output_file:
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r') as file:
                    content = file.read()
                    output_file.write(content)

    print('Final dataset is saved to', output_file_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=0, help="Tokenizer vocab size")
    parser.add_argument("--num_samples", type=int, default=10, help="Num samples for tokenizer training")
    parser.add_argument("--batch_id", type=str, default='00', help="ID of the batch to be processed")
    parser.add_argument("--text_path", type=str, default='tokenized_texts/', help="Path of output text file")
    parser.add_argument("--dir_path", type=str, default='tokenized_texts/', help="Path of directory")
    args = parser.parse_args()

    
#     train_tokenizer(vocab_size=args.vocab_size, num_samples=args.num_samples)
#     pre_tokenize(vocab_size=args.vocab_size)
    join_all_files(text_path=args.text_path, dir_path=args.dir_path)







