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

