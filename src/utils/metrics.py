"""Performance metrics collector for monitoring transcription quality."""

import time
from typing import Dict, List, Any, Optional
from collections import defaultdict
from loguru import logger


class MetricsCollector:
    """Collects and summarizes performance metrics.

    Tracks:
    - Transcription latency (ms)
    - Audio recording duration (ms)
    - Confidence scores
    - Error counts
    - Streaming chunk timings
    """

    def __init__(self, max_history: int = 200):
        self._metrics: Dict[str, List[float]] = defaultdict(list)
        self._max_history = max_history
        self._error_count: int = 0
        self._total_transcriptions: int = 0

    def record(self, metric_name: str, value: float):
        """Record a metric value.

        Args:
            metric_name: Name of the metric (e.g., 'transcription_ms')
            value: Metric value
        """
        self._metrics[metric_name].append(value)

        # Trim old values
        if len(self._metrics[metric_name]) > self._max_history:
            self._metrics[metric_name] = self._metrics[metric_name][-self._max_history:]

    def record_transcription(self, duration_ms: float, confidence: float):
        """Record a completed transcription with timing and confidence.

        Args:
            duration_ms: Transcription processing time in ms
            confidence: Transcription confidence (0-1)
        """
        self._total_transcriptions += 1
        self.record("transcription_ms", duration_ms)
        self.record("confidence", confidence)

    def record_error(self):
        """Record a transcription error."""
        self._error_count += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all metrics.

        Returns:
            Dictionary with metric summaries
        """
        import numpy as np

        summary = {
            "total_transcriptions": self._total_transcriptions,
            "total_errors": self._error_count,
        }

        for name, values in self._metrics.items():
            if values:
                arr = np.array(values)
                summary[name] = {
                    "count": len(arr),
                    "mean": float(np.mean(arr)),
                    "median": float(np.median(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "last": float(arr[-1]),
                }

        return summary

    def get_last(self, metric_name: str) -> Optional[float]:
        """Get the last recorded value for a metric."""
        values = self._metrics.get(metric_name, [])
        return values[-1] if values else None

    def reset(self):
        """Reset all metrics."""
        self._metrics.clear()
        self._error_count = 0
        self._total_transcriptions = 0


class Timer:
    """Simple context manager for timing code blocks.

    Usage:
        metrics = MetricsCollector()
        with Timer(metrics, "transcription_ms"):
            result = transcribe(audio)
    """

    def __init__(self, collector: MetricsCollector, metric_name: str):
        self._collector = collector
        self._metric_name = metric_name
        self._start: float = 0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        self._collector.record(self._metric_name, elapsed_ms)

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000
