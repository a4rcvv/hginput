from dataclasses import dataclass

import dataclasses_json


@dataclass
class MetaData(dataclasses_json.DataClassJsonMixin):
    labels: list[str]
