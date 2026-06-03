from __future__ import annotations

import time
from typing import List

from pydantic import BaseModel


class Metric(BaseModel):
    name: str
    value: float
    unit: str


class WorkloadResult(BaseModel):
    workload: str
    duration: float
    metrics: List[Metric]
    timestamp: float = time.time()
