import json
from tqdm import tqdm
import logging
import os
import zstandard as zstd
from datasets import load_from_disk

logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def is_zstd_file(file_path):
    """
    Check if a file is a zstandard compressed file.

    :param file_path: The path to the file.
    :return: True if the file is zstandard compressed, False otherwise.
    """
    return file_path.endswith('.zst')

def read_zstd_file(file_path):
    """
    Decompress a zstandard compressed file and yield each line as a JSON object.
    
    :param file_path: The path to the zstandard compressed file.
    """
    try:
        with open(file_path, 'rb') as compressed:
            decompressor = zstd.ZstdDecompressor()
            with decompressor.stream_reader(compressed) as reader:
                file_content = reader.read()
                lines = file_content.decode('utf-8').splitlines()
                for line in lines:
                    json_data = json.loads(line)
                    yield json_data
    except zstd.ZstdError as e:
        logger.error(f"Failed to decompress {file_path}: {e}")
        raise

def process_split_dataset(dataset_split, subset_name, split_name, output_dir):
    """
    Process the dataset split and save each subset to a separate file.

    :param dataset_split: The dataset split to process.
    :param subset_name: The name of the subset for logging purposes.
    :param split_name: The name of the dataset split (train, validation, test).
    :param output_dir: The directory to save the processed subset.
    """
    logger.info(f"Processing {subset_name} {split_name} split.")

    split_output_dir = os.path.join(output_dir, split_name)
    if not os.path.exists(split_output_dir):
        os.makedirs(split_output_dir)

    subset_file_path = os.path.join(split_output_dir, f"{subset_name}.jsonl")
    
    num_items = 0
    with open(subset_file_path, "w") as fout:
        for item in tqdm(dataset_split, desc=f"Processing {subset_name} {split_name}"):
            # Assuming the item is a dictionary with a "text" or "output" field
            text = item.get("text", item.get("output", ""))
            fout.write(json.dumps({"text": text}) + "\n")
            num_items += 1
    
    logger.info(f"Processed {num_items} items from {subset_name} {split_name} split.")

def split_and_process_dataset(dataset, subset_name, output_dir):
    """
    Split the dataset into train, validation, and test, and process each.

    :param dataset: The dataset to process.
    :param subset_name: The name of the subset.
    :param output_dir: The directory to save the processed data.
    """
    logger.info(f"Splitting {subset_name} dataset.")

    # Splitting the dataset
    dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
    test_dataset = dataset_split['test']
    remaining = dataset_split['train'].train_test_split(test_size=0.11, seed=42)
    train_dataset = remaining['train']
    validation_dataset = remaining['test']

    # Process each split
    process_split_dataset(train_dataset, subset_name, "train", output_dir)
    process_split_dataset(validation_dataset, subset_name, "validation", output_dir)
    process_split_dataset(test_dataset, subset_name, "test", output_dir)

def extract_subset_data(subset_name, dataset_path):
    """
    Extracts and processes a dataset stored on disk.

    :param subset_name: Name of the subset to process.
    :param dataset_path: Path to the dataset directory.
    """
    logger.info(f"Loading {subset_name} dataset from {dataset_path}.")
    dataset = load_from_disk(dataset_path)
    output_dir = f"./data/processed/{subset_name}"
    split_and_process_dataset(dataset, subset_name, output_dir)

def extract_from_file(file_path, subset_name, output_dir):
    """
    Extracts data from a file, processing it based on its type (zstd or regular).

    :param file_path: Path to the file.
    :param subset_name: Name of the subset to process.
    :param output_dir: Path to the output directory.
    """
    if is_zstd_file(file_path):
        logger.info(f"Extracting and processing {subset_name} from zstd file {file_path}.")
        output_file_path = os.path.join(output_dir, f"{subset_name}.jsonl")
        
        num_items = 0
        with open(output_file_path, "w") as fout:
            for item in read_zstd_file(file_path):
                text = item.get("text", item.get("output", ""))
                fout.write(json.dumps({"text": text}) + "\n")
                num_items += 1

        logger.info(f"Processed {num_items} items from zstd file {file_path}.")
    else:
        logger.info(f"Processing regular dataset from {file_path}.")
        extract_subset_data(subset_name, file_path)

def main():
    datasets_info = {
        'mc4_it_clean': 'mc4_it_clean_tiny',
        'mc4_en_clean': 'c4_200m_output',
        'wikipedia_it': 'wikipedia_it_20220120',
        'wikipedia_en': 'wikipedia_en_20220120',
        'culturaX_it': 'culturaX_it',
        'culturaX_en': 'culturaX_en'
    }

    for subset_name, dataset_path in datasets_info.items():
        extract_from_file(dataset_path, subset_name, './data/processed')

    for subset_name, file_path in zstd_files_info.items():
        extract_from_file(file_path, subset_name, './data/processed')

if __name__ == '__main__':
    if not os.path.exists("./data/processed"):
        os.mkdir("./data/processed")
    main()

