from datasets import list_datasets, load_dataset, list_metrics, load_metric

# Print all the available datasets
print(list_datasets())

# Load a dataset and print the first example in the training set
squad_dataset = load_dataset('squad')
print(squad_dataset['train'][0])

# List all the available metrics
print(list_metrics())

# Load a metric
squad_metric = load_metric('squad')

# Process the dataset - add a column with the length of the context texts
dataset_with_length = squad_dataset.map(lambda x: {"length": len(x["context"])})

# Process the dataset - tokenize the context texts (using a tokenizer from the 🤗 Transformers library)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

# tokenized_dataset = squad_dataset.map(lambda x: tokenizer(x['context']), batched=True)
tokenized_dataset = squad_dataset.map(lambda x: tokenizer.encode_plus(x['context'], max_length=1000), batched=True)