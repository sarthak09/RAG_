# data_ret/tracer.py

import time
import json
import logging
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger("rag.tracer")

class PipelineTimer:
    def __init__(self):
        self._times: dict[str, float] = {}

    @contextmanager
    def measure(self, stage: str):
        start = time.perf_counter()         
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._times[stage] = round(elapsed_ms, 2)

    def get(self, stage: str) -> float:
        return self._times.get(stage, 0.0)

    def total_ms(self) -> float:
        return round(sum(self._times.values()), 2)

class LatencyLogger:
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def save(self, trace: dict) -> None:
        filename = f"latency_trace_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
        filepath = self.log_dir / filename
        try:
            with open(filepath, "w") as f:
                json.dump(trace, f, indent=2)
            logger.info(
                f"Latency trace saved | strategy={trace.get('strategy')} "
                f"| total={trace.get('total_ms')}ms | file={filepath.name}"
            )
        except Exception as e:
            logger.error(f"Failed to save latency trace: {e}")

    def save_summary(self, traces: list[dict]) -> None:
        if not traces:
            return
        stages = ["query_encoding_ms", "retrieval_ms", "reranking_ms", "generation_ms", "total_ms"]
        summary = {"timestamp": datetime.now().isoformat(), "total_queries": len(traces), "stages": {}}
        for stage in stages:
            values = sorted([t.get(stage, 0.0) for t in traces])
            n = len(values)
            summary["stages"][stage] = {
                "p50":  round(values[int(n * 0.50)], 2),
                "p95":  round(values[int(n * 0.95)], 2),
                "p99":  round(values[int(n * 0.99)], 2),
                "mean": round(sum(values) / n, 2),
                "min":  round(values[0], 2),
                "max":  round(values[-1], 2),
            }
        filepath = self.log_dir / f"latency_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Latency summary saved → {filepath.name}")