import os
import polars as pl
from pathlib import Path
from logging import getLogger
import numpy as np
from sklearn.model_selection import train_test_split
from hginput.datatypes.metadata import MetaData


logger = getLogger(__name__)

SEED = 42


def create(tag: str, excluded: tuple[str, ...]):
    logger.info(f"start to merge raw datasets. excluded labels: {excluded}")
    parquet_files: list[str] = []
    raw_data_dir = "./hginput/model/data/raw"
    # create a directory if not exists
    dir = Path(f"./hginput/model/data/{tag}")
    if not dir.exists():
        dir.mkdir(parents=True)
    merged_data_path_test = Path(f"./hginput/model/data/{tag}/{tag}_test.parquet.zstd")
    merged_data_path_train = Path(f"./hginput/model/data/{tag}/{tag}_train.parquet.zstd")
    merged_data_path_metadata = Path(f"./hginput/model/data/{tag}/{tag}_metadata.json")

    for root, dirs, files in os.walk(raw_data_dir):
        logger.debug(f"root: {root}")
        if root.endswith(excluded):
            continue
        for file in files:
            if file.endswith(".parquet.zstd"):
                parquet_files.append(os.path.join(root, file))
    logger.info(f"number of target files: {len(parquet_files)}")

    # load LazyFrames
    lazy_dfs_dict: dict[str, list[pl.LazyFrame]] = {}  # key: label value: LazyFrame
    for file in parquet_files:
        df = pl.scan_parquet(file)
        label = file.split("/")[-2]
        if label not in lazy_dfs_dict.keys():
            lazy_dfs_dict[label] = [df]
        else:
            lazy_dfs_dict[label].append(df)
        logger.debug(f"scanned {file}")
    logger.info("scanned all files. saving...")

    # calculate the number of test records
    min_n_records = float("inf")
    n_records_dict: dict[str, int] = {}
    for label, lazy_dfs in lazy_dfs_dict.items():
        n_records_total = 0
        for lazy_df in lazy_dfs:
            n_records = lazy_df.collect().shape[0]
            n_records_total += n_records
        min_n_records = min(n_records_total, min_n_records)
        n_records_dict[label] = n_records_total
    test_data_rate = 0.2
    n_test_records_each_label = int(min_n_records * test_data_rate)
    logger.debug(
        f"the number of test records for each label is {n_test_records_each_label}"
    )
    logger.debug(f"total records: {sum(list(n_records_dict.values()))}")

    # split train and test
    train_dfs: list[pl.LazyFrame] = []
    test_dfs: list[pl.LazyFrame] = []
    for label, lazy_dfs in lazy_dfs_dict.items():
        lazy_df = pl.concat(lazy_dfs).with_row_count("index")
        row_indexes = np.arange(n_records_dict[label])
        train_row_indexes, test_row_indexes = train_test_split(
            row_indexes, test_size=n_test_records_each_label, random_state=SEED
        )
        logger.debug(
            f"the number of train records of {label}: {len(train_row_indexes)}"
        )
        logger.debug(f"the number of test records of {label}: {len(test_row_indexes)}")
        test_df = lazy_df.filter(pl.col("index").is_in(test_row_indexes)).drop("index")
        train_df = lazy_df.filter(pl.col("index").is_in(train_row_indexes)).drop(
            "index"
        )
        test_dfs.append(test_df)
        train_dfs.append(train_df)

    # preprocess
    tr_lazy = pl.concat(train_dfs)
    ts_lazy = pl.concat(test_dfs)
    # encode column "hand" to 0 or 1
    hand_encode_expr = (
        pl.when(pl.col("hand") == "Left")
        .then(0)
        .otherwise(1)
        .cast(pl.UInt8)
        .alias("hand")
    )
    tr_lazy = tr_lazy.with_columns(hand_encode_expr)
    ts_lazy = ts_lazy.with_columns(hand_encode_expr)
    # encode column "label" to 0, 1, 2, ...
    labels = [label for label in lazy_dfs_dict.keys()]
    label_map = {label: i for i, label in enumerate(labels)}
    label_encode_expr = (
        pl.col("label").map_dict(label_map).cast(pl.UInt8).alias("label")
    )
    tr_lazy = tr_lazy.with_columns(label_encode_expr)
    ts_lazy = ts_lazy.with_columns(label_encode_expr)
    # change dtypes
    columns_convert_to_f32 = tr_lazy.columns
    columns_convert_to_f32.remove("label")
    columns_convert_to_f32.remove("hand")
    tr_lazy = tr_lazy.with_columns(
        [pl.col(col).cast(pl.Float32) for col in columns_convert_to_f32]
    )
    ts_lazy = ts_lazy.with_columns(
        [pl.col(col).cast(pl.Float32) for col in columns_convert_to_f32]
    )

    # save train and test
    tr = tr_lazy.collect()
    logger.debug(f"the length of training dataset: {tr.shape[0]}")
    logger.debug(f"columns: {tr.columns}")
    tr.write_parquet(merged_data_path_train)
    ts = ts_lazy.collect()
    logger.debug(f"the length of test dataset: {ts.shape[0]}")
    ts.write_parquet(merged_data_path_test)

    # show the structure of the dataset
    logger.debug(tr.head())

    # save metadata
    metadata = MetaData(labels=labels)
    metadata.to_json()
    with open(merged_data_path_metadata, "w") as f:
        f.write(metadata.to_json())

    logger.info("succeeded to create datasets.")
