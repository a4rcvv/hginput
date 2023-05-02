import os
from typing import Optional
import polars as pl
from pathlib import Path
from logging import getLogger

logger = getLogger(__name__)


def merge(filename: str, excluded: tuple[str, ...]):
    logger.info(f"start to merge raw datasets. excluded labels: {excluded}")
    parquet_files: list[str] = []
    raw_data_dir = "./model/data/raw"
    merged_data_path = Path(f"./model/data/{filename}.parquet.zstd")

    for root, dirs, files in os.walk(raw_data_dir):
        logger.debug(f"root: {root}")
        if root.endswith(excluded):
            continue
        for file in files:
            if file.endswith(".parquet.zstd"):
                parquet_files.append(os.path.join(root, file))
    logger.info(f"number of target files: {len(parquet_files)}")

    dfs: list[pl.LazyFrame] = []
    for file in parquet_files:
        dfs.append(pl.scan_parquet(file))
        logger.debug(f"scanned {file}")

    logger.info("scanned all files. saving...")
    merged = pl.concat(dfs)
    df = merged.collect()
    df.write_parquet(merged_data_path)
    logger.info("succeeded to merge datasets.")
