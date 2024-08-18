import os
import json
import torch
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2Tokenizer
from tqdm import tqdm

def load_jsonl_files(directory):
    """
    Load and combine JSONL files from a specified directory.
    
    Args:
        directory (str): Path to the directory containing JSONL files.
    
    Returns:
        list: A list of documents where each document is a dictionary.
    """
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith("_processed.jsonl"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                for line in file:
                    documents.append(json.loads(line))
    return documents

def train_gpt2_mini():
    dataset_directory = './data/combined-datasets/'
    documents = load_jsonl_files(dataset_directory)
    os.makedirs("temp_corpus", exist_ok=True)
    with open("temp_corpus/corpus.txt", "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc['contents'] + "\n")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files="temp_corpus/corpus.txt", vocab_size=50257, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

    os.makedirs("custom_tokenizer", exist_ok=True)
    tokenizer.save_model("custom_tokenizer")
    tokenizer = GPT2Tokenizer.from_pretrained("custom_tokenizer")

    #custom conf
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=128,
        n_ctx=128,
        n_embd=64,
        n_layer=4,
        n_head=4,
    )

    model = GPT2LMHeadModel(config)

    def tokenize_function(doc):
        return tokenizer(doc['contents'], return_tensors='pt', max_length=128, truncation=True, padding='max_length')
    tokenized_inputs = []
    for doc in tqdm(documents, desc="Tokenizing documents"):
        tokenized_inputs.append(tokenize_function(doc))
    input_ids = torch.cat([item['input_ids'] for item in tokenized_inputs], dim=0)
    attention_mask = torch.cat([item['attention_mask'] for item in tokenized_inputs], dim=0)

    dataset = torch.utils.data.TensorDataset(input_ids, attention_mask)

    training_args = TrainingArguments(
        output_dir="./gpt2",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model("./gpt2")
    tokenizer.save_pretrained("./gpt2")

    print("Training complete. Model and tokenizer saved to ./gpt2")

if __name__ == "__main__":
    train_gpt2_mini()

