import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import Dataset, random_split, DataLoader
import torch
import polars
from hginput.datatypes.metadata import MetaData


class GestureDataset(Dataset):
    def __init__(self, data_path: str, metadata_path: str) -> None:
        super().__init__()
        self.df = polars.read_parquet(data_path)
        with open(metadata_path, "r") as f:
            json = f.read()
            self.metadata: MetaData = MetaData.from_json(json)

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, index: int) -> tuple[polars.Series, polars.Series]:
        row = self.df[index]
        label_index = torch.tensor(row.get_column("label"))
        label_tensor: torch.Tensor = torch.functional.F.one_hot(
            label_index, self.metadata.n_labels
        )
        label_tensor = label_tensor.to(torch.float32)
        label_tensor = label_tensor.squeeze()
        feature_tensor = torch.tensor(
            row.drop("label").row(index=0), dtype=torch.float32
        )

        return feature_tensor, label_tensor


class GestureDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tag: str,
        batch_size: int = 32,
        validation_rate: float = 0.2,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.tag = tag
        self.train_path = f"./hginput/model/data/{tag}/{tag}_train.parquet.zstd"
        self.test_path = f"./hginput/model/data/{tag}/{tag}_test.parquet.zstd"
        self.metadata_path = f"./hginput/model/data/{tag}/{tag}_metadata.json"
        with open(self.metadata_path, "r") as f:
            self.metadata: MetaData = MetaData.from_json(f.read())
        self.batch_size = batch_size
        self.validation_rate = validation_rate
        self.tr_va_dataset: GestureDataset | None = None
        self.tr_dataset: GestureDataset | None = None
        self.va_dataset: GestureDataset | None = None
        self.ts_dataset: GestureDataset | None = None
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        if stage == "test" and self.ts_dataset is None:
            self.ts_dataset = GestureDataset(self.test_path, self.metadata_path)
        elif (stage == "validate" or stage == "fit") and (
            self.tr_dataset is None or self.va_dataset is None
        ):
            self.tr_va_dataset = GestureDataset(self.train_path, self.metadata_path)
            n_va = int(len(self.tr_va_dataset) * self.validation_rate)
            n_tr = len(self.tr_va_dataset) - n_va
            self.tr_dataset, self.va_dataset = random_split(
                self.tr_va_dataset, [n_tr, n_va]
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if self.tr_dataset is None:
            raise RuntimeError("training dataset is None")
        return DataLoader(
            self.tr_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.va_dataset is None:
            raise RuntimeError("validation dataset is None")
        return DataLoader(
            self.va_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        if self.ts_dataset is None:
            raise RuntimeError("test dataset is None")
        return DataLoader(
            self.ts_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
