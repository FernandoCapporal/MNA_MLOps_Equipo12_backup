from pydantic import BaseModel
from typing import List, Dict


class PredictionInput(BaseModel):
    data: List[Dict[str, float]]
