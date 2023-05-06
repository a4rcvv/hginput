import dataclasses_json
from dataclasses import dataclass


@dataclass
class MetaData(dataclasses_json.DataClassJsonMixin):
    label_map: dict[str, int]
    n_labels: int
