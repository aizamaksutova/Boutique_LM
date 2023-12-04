import random
from typing import List, Union, Tuple

import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
import copy

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import re
import wandb
import torch.optim.lr_scheduler as scheduler
import os
from datetime import datetime
# import torch.cuda.amp.GradScaler

import sentencepiece as spm

CHK_DIR = 'saved'


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout: float = 0.1, max_len: int = 5000):
        """
        Inputs
            embed_dim - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, self.embed_dim)
        pos = torch.arange(max_len).reshape(-1, 1)
        denom = torch.pow(10000, (torch.arange(self.embed_dim) - (torch.arange(self.embed_dim) % 2)) / embed_dim)
        pe = pos / denom
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[-2], :]
        return self.dropout(x)


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_sequence_length):
        self.tokenizer = tokenizer
        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id
        self.pad_id = tokenizer.pad_id
        self.max_sequence_length = max_sequence_length

        with open(file_path, encoding="utf-8") as file:
            texts = list(map(lambda x: x.strip(), file.readlines()))

        self.sentences = texts

        self.indices = self.tokenizer.tokenize(self.sentences)

    def __len__(self):

        return len(self.indices)
    
    def text2ids(self, texts: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        """
        Encode a text or list of texts as tokenized indices
        :param texts: text or list of texts to tokenize
        :return: encoded indices
        """
        return self.tokenizer.tokenize(texts)

    def ids2text(self, ids: Union[torch.Tensor, List[int], List[List[int]]]) -> Union[str, List[str]]:
        """
        Decode indices as a text or list of tokens
        :param ids: 1D or 2D list (or torch.Tensor) of indices to decode
        :return: decoded texts
        """
        if torch.is_tensor(ids):
            assert len(ids.shape) <= 2, 'Expected tensor of shape (length, ) or (batch_size, length)'
            ids = ids.cpu().tolist()

        return self.tokenizer.token_to_character(ids)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Add specials to the index array and pad to maximal length
        :param item: text id
        :return: encoded text indices and its actual length (including BOS and EOS specials)
        """
        # These are placeholders, you may remove them.
        indices = [self.bos_id] + self.indices[idx][:self.max_sequence_length - 2] + [self.eos_id]
        padded_tokenized_sentence = self.pad_sequence(indices)
        return torch.tensor(padded_tokenized_sentence, dtype=torch.long)
    
    def pad_sequence(self, sequence):
        sequence += [self.tokenizer.pad_id] * (self.max_sequence_length - len(sequence))
        return sequence[:self.max_sequence_length]



class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout, batch_first=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=batch_first)
        self.attn_dropout = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, attention_mask, padding_mask):
        x_ = self.norm1(x)
        x_, _ = self.attention(x, x, x, attn_mask=attention_mask, key_padding_mask=padding_mask)
        x_ = x + x_
        x_ = self.norm2(x_)
        x_ = x_ + self.feed_forward(x_)
        return x_

class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.decoder = nn.ModuleList([copy.deepcopy(decoder_layer) for i in range(num_layers)])
    
    def forward(self, x, attention_mask, padding_mask):
        for layer in self.decoder:
            x = layer(x, attention_mask, padding_mask)
        return x


class LM(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, vocab_size, ff_dim, dropout, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.positional_encoding = PositionalEncoding(embed_dim=embed_dim, dropout=dropout)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = Decoder(
            DecoderLayer(embed_dim, num_heads, ff_dim, dropout, batch_first=True),
            num_layers=num_layers
        )
        self.classification = nn.Linear(embed_dim, vocab_size)
    
    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([torch.prod(torch.tensor(p.shape)) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)
    
    def forward(self, input_ids: Tensor, attention_mask: Tensor, padding_mask: Tensor):
        x = self.embedding(input_ids)
        x = self.positional_encoding(x)
        x = self.transformer(x, attention_mask, padding_mask)
        return self.classification(x)
    
    def get_next_token(self, prefix: Tensor, attention_mask: Tensor, padding_mask: Tensor):
        """ :returns: probabilities of next token """
        return self.forward(prefix, attention_mask, padding_mask)[:, -1, :]



class Tokenizer:

    def __init__(self, tokenizer_model=None):
        model_path = tokenizer_model

        self.sp_model = spm.SentencePieceProcessor(model_file=model_path)
        self.model_path = model_path

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        #print(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()


    def tokenize(self, s: Union[str, List[str]]) -> List[int]:
        t = self.sp_model.encode(s)
        return t
    
    def token_to_character(self, t: Union[torch.Tensor, List[int], List[List[int]]]) -> Union[str, List[str]]:
        return self.sp_model.decode(t)

    def size(self):
        return self.n_words



def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:

    def __init__(self, model, log_step, save_interval, sched_step, device, tokenizer: Tokenizer,optimizer=None, lr_scheduler=None):
        super().__init__()
        self.model = model
        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        else:
            self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id)
        self.log_step=log_step
        self.lr_scheduler = lr_scheduler
        self.save_interval = save_interval
        self.sched_step = sched_step
        self.device = device
        

    def generate_square_mask(self, sz, device):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_mask(self, x, pad_idx, device):
        if len(x.shape) == 2:
            tgt_seq_len = x.shape[1]
        else:
            tgt_seq_len = x.shape[0]
        tgt_mask = self.generate_square_mask(tgt_seq_len, device)
        tgt_padding_mask = (x == pad_idx)
        return tgt_mask, tgt_padding_mask


    def train(self, epochs, data_loader, wandb):

        accum_step = 0
        # scaler = GradScaler()

        for epoch in range(epochs):
            losses = []

            # Train the model on each batch
            for batch in data_loader:
                accum_step += 1
                self.model.train()

                batch_input = batch[:, :-1].to(self.device)


                batch_mask, batch_padding_mask = self.create_mask(batch_input, self.tokenizer.pad_id, self.device)

                # Compute the model output
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = self.model(batch_input, batch_mask, batch_padding_mask).to(self.device)
                    tgt_out = batch[:, 1:].to(self.device)
                    loss = self.loss_function(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
                losses.append(loss.item())

                # Backpropagate the loss.
                loss.backward()


                # Update the model parameters. This is done by taking a step in the direction of the gradient.


                losses.append(loss.item())

                prefix = self.generate(self.model, self.tokenizer, 3, self.tokenizer.pad_id)
                texts = dataset.ids2text(prefix)
            

                if accum_step % self.log_step == 0:

                    wandb.log({"train_loss": np.mean(losses),
                                "grad norm": self.get_grad_norm(),
                                "learning rate": self.lr_scheduler.get_last_lr()[0]
                                }, step=accum_step)

                    losses = []
                    prefix = self.generate(self.model, self.tokenizer, 3, self.tokenizer.pad_id)
                    generated_text = dataset.ids2text(prefix)[0]

                    wandb.log({"train_text": wandb.Html(generated_text)})
                if accum_step % self.save_interval == 0:
                    timestamp = str(datetime.now())
                    os.makedirs(CHK_DIR, exist_ok=True)
                    checkpoint_path = os.path.join(CHK_DIR, f"model_{timestamp}.pt")
                    state = {
                        "arch": type(self.model).__name__,
                        "epoch": epoch,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict() if self.scheduler is not None else ""
                    }
                    torch.save(state, checkpoint_path)
                
                
                self.optimizer.step()
               
                self.optimizer.zero_grad()

            # Print the loss

            print('Epoch:', epoch)


        return epoch
    
    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()
    @torch.no_grad()
    def generate(self, model, tokenizer, batch_size: int, pad_idx, prefix: Tensor=None, max_len=384):
        """
        Samples output sequence from probability distribution obtained by model.
        if Tensor of prefix is None then full it with [BOS] token

        :params
            model: predict next token for the whole batch of sequences
            tokenizer: tokenizer for the model and [BOS] token
            batch_size: number of sequence
            prefix: Tensor of tokens with shape: [batch_size, seq_len]
            max_len: max length of predicted sequence

        :return
            the Tensor of tokens of shape: [batch_size, max_len + 1]
        """
        model.eval()

        if prefix is None:
            prefix = torch.full((batch_size, 1), fill_value=tokenizer.bos_id).to(next(model.parameters()).device)
        
        count = max_len - prefix.shape[-1]
        for i in range(count):
            prefix = prefix.clone().detach()
            tgt_mask, tgt_padding_mask = self.create_mask(prefix, pad_idx, device='cuda')

            output_logits = torch.nn.functional.softmax(model.get_next_token(prefix, tgt_mask, tgt_padding_mask), dim=-1)
            
            prefix = torch.cat((prefix, torch.multinomial(output_logits, 1)), dim=-1)
        
        return prefix
    
    
    
    
if __name__ == "__main__":
    wandb.login()

    wandb.init(
                project='Boutique_LM'
            )
        # Create the tokenizer
    tokenizer_path = 'vocab/tokenizer_4096.model'
    tokenizer = Tokenizer(tokenizer_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_file = 'tokenized_texts/df_small.txt'
    embedding_dimension = 256
    feed_forward_dimension= 512
    max_sequence_length = 256 ##change
    number_of_tokens = tokenizer.size()
    batch_size=400
    print(number_of_tokens)

    dataset = TextDataset(train_file, tokenizer, max_sequence_length)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("Data loader complete")

    # Create the model
    model = LM(
        embed_dim=embedding_dimension,
        ff_dim=feed_forward_dimension,
        vocab_size=number_of_tokens,
        num_heads=8,
        num_layers=4,
        dropout=0.1
    ).to(get_device())




    # Train the model
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=1e-5)
    lr_scheduler = scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=2000, epochs=1000000)
    trainer = Trainer(model, 50, 500, 1000000, device, tokenizer, optimizer, lr_scheduler)
    epoch = trainer.train(epochs=1000000, data_loader=data_loader, wandb=wandb)
