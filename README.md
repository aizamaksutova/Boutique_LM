# Boutique_LM

## Data structure

Downloading data from HuggingFace

```
wget --quiet --show-progress "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
tar -xzvf TinyStories_all_data.tar.gz
jq '.' data00.json
```

Last command is done to see the structure of JSON. In one JSON we have multiple stories with such structure

```
{
    "story": ... ,
    "instruction": {
      "prompt:": ... ,
      "words": [
        ...,
        ... ,
        ...
      ],
      "features": [
        ...
      ]
    },
    "summary": ... ,
    "source": ...
  }
```

## Tokenizer

We need to implement and train a custom tokenizer for our task
To train the tokenizer model do the following:

```
python3 tokenizer_train.py --vocab_size=16000 --num_samples=15
```
You can customize the vocabulary size and num_samples to get more representative data. 
Your model and vocab will be saved in the vocab dir

