import os
import shutil
import time
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from packing_dataset import JsonlDataset, CorpusDataset, AsyncDataset, BlendCorpusDataset, BlendAsyncDataset
from retrieval_packing import DefragmentConfig
from tqdm import tqdm
import logging
from retriv_bm25 import SparseRetriever
from retrieval_packing import bm25_defragment_retriv, bm25_defragment_retriv_simplified
from project_config import *
from preprocessing import *
import argparse


logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

tokenizer = CustomTokenizer()

def save_data_collator(examples):
    examples = [{"input_ids": example["input_ids"]} for example in examples]
    return examples


def load_dataset_with_retriever(
        file_path, chunk_size: int, defragment_config: DefragmentConfig, index_path, qlen
):
    retriever = SparseRetriever.load(index_path)
    doc_num = retriever.doc_count
    retriever.set_maximum_query_length(qlen)
    iter_order = list(range(doc_num))
    rng = np.random.RandomState(666)
    rng.shuffle(iter_order)
    jsonl_dataset = JsonlDataset(
        name="combined_datasets", jsonl_paths=[file_path], is_train_data=True,
        iter_in_order=False, has_tokenized=True, iter_order=iter_order
    )
    defragmentation_fn = bm25_defragment_retriv_simplified
    corpus_dataset = CorpusDataset(
        jsonl_dataset, tokenizer=tokenizer, chunk_size=chunk_size,
        is_eval_data=False, defragment_config=defragment_config, mask_chunk=False,
        defragmentation_fn=defragmentation_fn, retriever=retriever
    )

    return corpus_dataset


def create_one_file_dataset(
        subset_name, file_idx, fragments_buffer_size, chunk_size, qlen, data_name,
        multihop=True, shuffle_chains=False, save_batch_size=256, num_workers=1,
):
    index_path = f"{subset_name}_{file_idx}"
    jsonl_dir = os.path.join("./data/combined-datasets", subset_name)
    file_path = os.path.join(jsonl_dir, f"{subset_name}_chunk{file_idx}_processed.jsonl")

    defragment_config = DefragmentConfig(
        defragmentation_method="bm25", fragments_buffer_size=fragments_buffer_size,
        shuffle_chains=shuffle_chains, multihop=multihop
    )
    outer_data_dir = os.path.join("./data/offline_datasets", data_name)
    if os.path.exists(outer_data_dir):
        pass
    else:
        os.makedirs(outer_data_dir)

    subset_dir = os.path.join(outer_data_dir, subset_name)
    split_subset_dir = os.path.join(subset_dir, str(file_idx))
    if not os.path.exists(split_subset_dir):
        os.makedirs(split_subset_dir)

    get_batch_dir_and_path = AsyncDataset.get_batch_dir_and_path

    train_dataset = load_dataset_with_retriever(file_path, chunk_size, defragment_config, index_path, qlen)

    iter_batch_size = 4
    dataloader = DataLoader(
        train_dataset,
        batch_size=iter_batch_size,
        collate_fn=save_data_collator,
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=8 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )

    tqdm_bar = tqdm()

    batch_cache = []
    batch_idx = 0
    start_time = time.perf_counter()
    for iter_batch_data in dataloader:
        batch_cache.extend(iter_batch_data)
        tqdm_bar.update(len(iter_batch_data))
        if len(batch_cache) >= save_batch_size:
            batch_dir, batch_path = get_batch_dir_and_path(split_subset_dir, batch_idx)
            if not os.path.exists(batch_dir):
                os.mkdir(batch_dir)
            torch.save(batch_cache[:save_batch_size], batch_path)
            batch_cache = batch_cache[save_batch_size:]
            batch_idx += 1
            logger.info(f"yield {batch_idx}th batch, {time.perf_counter() - start_time:.5f}")
            start_time = time.perf_counter()
            tqdm_bar.set_description(f"save {batch_idx * save_batch_size * chunk_size / 1024 ** 3:.3f}B tokens")

    logger.info(f"saved {batch_idx * save_batch_size * chunk_size / 1024 ** 3:.3f}B tokens")


def estimate_total_batch_num(cur_all_split_dirs):
    cnt = 0
    for split_dir in cur_all_split_dirs:
        chunk_1024_ids = os.listdir(split_dir)
        chunk_1024_dirs = [os.path.join(split_dir, chunk_1024_id) for chunk_1024_id in chunk_1024_ids]
        for chunk_idx, chunk_dir in enumerate(chunk_1024_dirs):
            load_batch_ids = os.listdir(chunk_dir)
            cnt += len(load_batch_ids)
    return cnt


def combine_data(dataset_path):
    """
    original:
    [dataset_path]/[subset_name]/[split_id]/[chunk_1024_id]/[batch_id.pt]

    combine to:
    [dataset_path]/[subset_name]/[chunk_1024_id]/[batch_id.pt]
    with only one split, split_id = 1
    """

    get_batch_dir_and_path = AsyncDataset.get_batch_dir_and_path

    for subset in SUBSET_NAMES:

        cur_split_dir = os.path.join(dataset_path, subset)
        cur_all_split_dirs = os.listdir(cur_split_dir)
        cur_all_split_dirs = [os.path.join(cur_split_dir, split_dir) for split_dir in cur_all_split_dirs]

        # re-count batch-idx for combined subset
        cur_saved_batch_idx = 0

        bar = tqdm(total=estimate_total_batch_num(cur_all_split_dirs))
        for split_dir in cur_all_split_dirs:
            # get all chunk paths, each chunk contains 1024 batches at most
            chunk_1024_ids = os.listdir(split_dir)
            chunk_1024_dirs = [os.path.join(split_dir, chunk_1024_id) for chunk_1024_id in chunk_1024_ids]
            for chunk_idx, chunk_dir in enumerate(chunk_1024_dirs):
                # get all batch-paths in a chunk
                load_batch_ids = os.listdir(chunk_dir)
                load_batch_paths = [os.path.join(chunk_dir, batch_id) for batch_id in load_batch_ids]

                for load_batch_path in load_batch_paths:
                    save_batch_dir, save_batch_path = get_batch_dir_and_path(cur_split_dir, cur_saved_batch_idx)
                    if not os.path.exists(save_batch_dir):
                        os.mkdir(save_batch_dir)

                    shutil.copyfile(load_batch_path, save_batch_path)

                    cur_saved_batch_idx += 1

                    bar.update(1)
                    bar.set_description(f"{os.path.basename(split_dir)}-"
                                        f"{chunk_idx:05d}-"
                                        f"{os.path.basename(load_batch_path)}"
                                        f", saved {cur_saved_batch_idx} batches")
                shutil.rmtree(chunk_dir)
            shutil.rmtree(split_dir)


def blend_data(dataset_path, total_token_nums, chunk_size):
    corpus_datasets = []
    corpora_iter_order = []
    total_samples = total_token_nums // chunk_size
    for subset_idx, subset_name in enumerate(SUBSET_NAMES):
        subset_saved_data_path = os.path.join(dataset_path, subset_name)
        dir_nums = len(os.listdir(subset_saved_data_path))
        subset_dataset = AsyncDataset(subset_saved_data_path, name=subset_name, dir_nums=dir_nums)
        corpus_datasets.append(subset_dataset)
        cur_corpus_indices = [subset_idx] * round(SUBSET_WEIGHTS[subset_name] * total_samples + 1)
        corpora_iter_order.extend(cur_corpus_indices)
    rng = np.random.RandomState(666)
    rng.shuffle(corpora_iter_order)
    corpora_iter_order = corpora_iter_order[:total_samples]
    train_dataset = BlendAsyncDataset(corpus_datasets, corpora_iter_order, SUBSET_WEIGHTS)

    save_batch_size = 256
    target_batch_num = math.ceil(total_token_nums / (save_batch_size * chunk_size))
    get_batch_dir_and_path = AsyncDataset.get_batch_dir_and_path
    dataloader = DataLoader(
        train_dataset,
        batch_size=save_batch_size,
        collate_fn=save_data_collator,
        num_workers=12,
    )

    tqdm_bar = tqdm(total=target_batch_num)
    batch_idx = 0
    for iter_batch_idx, iter_batch_data in enumerate(dataloader):

        assert len(iter_batch_data) == save_batch_size

        batch_dir, batch_path = get_batch_dir_and_path(dataset_path, batch_idx)
        if not os.path.exists(batch_dir):
            os.mkdir(batch_dir)
        torch.save(iter_batch_data, batch_path)
        tqdm_bar.update(1)
        batch_idx += 1

        if batch_idx >= target_batch_num:
            break

    logger.info(f"saved {batch_idx * save_batch_size * chunk_size / 1024 ** 3:.3f}B tokens")


def save_bm25chunk_dataset(buffer_size, chunk_size, data_name, qlen, total_token_nums):
    for subset_name in SUBSET_NAMES:
        for file_idx in range(SUBSET_SPLIT_NUMS[subset_name]):
            create_one_file_dataset(subset_name, file_idx, buffer_size, chunk_size, qlen, data_name)
            logger.info(f"{subset_name} {file_idx} completed")
    combine_data(f"./data/offline_datasets/{data_name}")
    blend_data(f"./data/offline_datasets/{data_name}", total_token_nums, chunk_size)


def save_mixchunk_dataset(chunk_size, data_name, total_token_nums):
    data_dir = f"./data/offline_datasets/{data_name}"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    get_batch_dir_and_path = AsyncDataset.get_batch_dir_and_path

    all_files = []
    for subset_name in SUBSET_NAMES:
        for split_id in range(SUBSET_SPLIT_NUMS[subset_name]):
            all_files.append(f"./data/combined-datasets/{subset_name}/{subset_name}_chunk{split_id}_processed.jsonl")
    logger.info(f"detect {len(all_files)} jsonl files.")

    jsonl_dataset = JsonlDataset(
        name="combined_datasets", jsonl_paths=all_files, iter_order=None,
        is_train_data=True, iter_in_order=False, has_tokenized=True,
    )
    corpus_datasets = [CorpusDataset(
        jsonl_dataset, tokenizer=tokenizer, chunk_size=chunk_size,
        is_eval_data=False, defragment_config=None, mask_chunk=False
    )]
    train_dataset = BlendCorpusDataset(corpus_datasets, [0] * (total_token_nums // chunk_size))

    save_batch_size = 256
    iter_batch_size = 256
    dataloader = DataLoader(
        train_dataset,
        batch_size=iter_batch_size,
        collate_fn=save_data_collator,
        num_workers=8,
    )

    target_batch_num = math.ceil(total_token_nums / (save_batch_size * chunk_size))
    tqdm_bar = tqdm(total=target_batch_num)

    batch_cache = []
    batch_idx = 0

    for iter_batch_idx, iter_batch_data in enumerate(dataloader):

        batch_cache.extend(iter_batch_data)

        if len(batch_cache) >= save_batch_size:
            batch_dir, batch_path = get_batch_dir_and_path(data_dir, batch_idx)
            if not os.path.exists(batch_dir):
                os.mkdir(batch_dir)
            torch.save(batch_cache[:save_batch_size], batch_path)
            batch_cache = batch_cache[save_batch_size:]
            tqdm_bar.update(1)
            batch_idx += 1

            tqdm_bar.set_description(f"saved {batch_idx * save_batch_size * chunk_size / 1024 ** 3:.3f}B tokens")
            if batch_idx > target_batch_num:
                break

    logger.info(f"saved {batch_idx * save_batch_size * chunk_size / 1024 ** 3:.3f}B tokens")


def save_unichunk_dataset(chunk_size, data_name, total_token_nums):
    data_dir = f"./data/offline_datasets/{data_name}"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    get_batch_dir_and_path = AsyncDataset.get_batch_dir_and_path

    corpus_datasets = []
    for subset_name in SUBSET_NAMES:
        jsonl_paths = []
        for split_id in range(SUBSET_SPLIT_NUMS[subset_name]):
            jsonl_paths.append(f"./data/combined-datasets/{subset_name}/{subset_name}_chunk{split_id}_processed.jsonl")
        logger.info(f"{subset_name}, {len(jsonl_paths)} files.")
        jsonl_dataset = JsonlDataset(
            name=subset_name, jsonl_paths=jsonl_paths, iter_order=None,
            is_train_data=True, iter_in_order=False, has_tokenized=True
        )
        corpus_datasets.append(CorpusDataset(
            jsonl_dataset, tokenizer=tokenizer, chunk_size=chunk_size,
            is_eval_data=False, defragment_config=None, mask_chunk=False
        ))
    total_samples = total_token_nums // chunk_size
    corpora_iter_order = []
    for corpus_idx, corpus in enumerate(corpus_datasets):
        weight = SUBSET_WEIGHTS[corpus.jsonl_dataset.name]
        cur_corpus_indices = [corpus_idx] * round(weight * total_samples + 1)
        corpora_iter_order.extend(cur_corpus_indices)
    rng = np.random.RandomState(666)
    rng.shuffle(corpora_iter_order)
    corpora_iter_order = corpora_iter_order[:total_samples]
    train_dataset = BlendCorpusDataset(corpus_datasets, corpora_iter_order, corpus_weights=SUBSET_WEIGHTS)
    save_batch_size = 256
    iter_batch_size = 256
    dataloader = DataLoader(
        train_dataset,
        batch_size=iter_batch_size,
        collate_fn=save_data_collator,
        num_workers=8,
    )

    target_batch_num = math.ceil(total_token_nums / (save_batch_size * chunk_size))
    tqdm_bar = tqdm(total=target_batch_num)

    batch_cache = []
    batch_idx = 0

    for iter_batch_idx, iter_batch_data in enumerate(dataloader):

        batch_cache.extend(iter_batch_data)

        if len(batch_cache) >= save_batch_size:
            batch_dir, batch_path = get_batch_dir_and_path(data_dir, batch_idx)
            if not os.path.exists(batch_dir):
                os.mkdir(batch_dir)
            torch.save(batch_cache[:save_batch_size], batch_path)
            batch_cache = batch_cache[save_batch_size:]
            tqdm_bar.update(1)
            batch_idx += 1

            tqdm_bar.set_description(f"saved {batch_idx * save_batch_size * chunk_size / 1024 ** 3:.3f}B tokens")
            if batch_idx > target_batch_num:
                logger.info("finnish")
                break

    logger.info(f"saved {batch_idx * save_batch_size * chunk_size / 1024 ** 3:.3f}B tokens")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packing_strategy", type=str, choices=["mixchunk", "unichunk", "bm25chunk"])
    parser.add_argument("--chunk_size", type=int, default=8192)
    parser.add_argument("--buffer_size", type=int, default=4096)
    parser.add_argument("--max_query_len", type=int, default=300)
    parser.add_argument("--data_name", type=str, default=None)
    parser.add_argument("--total_token_nums", type=int, default=151 * 1024 ** 3)
    parser.add_argument("--retriv_base_path", type=str, default="./data/bm25index")
    parser.add_argument("--bm25chunk_onefile", action="store_true")
    parser.add_argument("--subset_name", type=str, default=None)
    parser.add_argument("--file_idx", type=int, default=None)
    parser.add_argument("--combine_data", action="store_true")
    args = parser.parse_args()
    if args.data_name is None:
        args.data_name = args.packing_strategy

    if os.path.exists(f"./data/offline_datasets/{args.data_name}"):
        raise FileExistsError(f"./data/offline_datasets/{args.data_name} is non-empty")

    if args.packing_strategy == "bm25chunk":
        if args.combine_data:
            combine_data(f"./data/offline_datasets/{args.data_name}")
            blend_data(f"./data/offline_datasets/{args.data_name}", args.total_token_nums, args.chunk_size)
        else:
            os.environ["RETRIV_BASE_PATH"] = args.retriv_base_path
            if args.bm25chunk_onefile:
                create_one_file_dataset(args.subset_name, args.file_idx, args.buffer_size,
                                        args.chunk_size, args.max_query_len, args.data_name)
            else:
                save_bm25chunk_dataset(args.buffer_size, args.chunk_size, args.data_name,
                                       args.max_query_len, args.total_token_nums)
    elif args.packing_strategy == "mixchunk":
        save_mixchunk_dataset(args.chunk_size, args.data_name, args.total_token_nums)
    else:
        save_unichunk_dataset(args.chunk_size, args.data_name, args.total_token_nums)


if __name__ == '__main__':
    main()

#main framework from https://github.com/yuzhaouoe/pretraining-data-packing/blob/main/save_offline_dataset.py
