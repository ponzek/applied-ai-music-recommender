"""Logger — Structured logging for the agent pipeline.

Records every action the agent takes with timestamps, inputs, outputs,
and durations. Outputs to console and saves to log files in logs/.
"""

import os
import time
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")


class AgentLogger:
    """Structured logger that records agent steps to console and file."""

    def __init__(self, run_name: str = None):
        """Initialize the logger.

        Args:
            run_name: Optional name for this run. Defaults to timestamp.
        """
        os.makedirs(LOGS_DIR, exist_ok=True)

        if run_name is None:
            run_name = time.strftime("%Y%m%d_%H%M%S")

        self.run_name = run_name
        self.log_file = os.path.join(LOGS_DIR, f"run_{run_name}.txt")
        self.steps: list = []
        self.start_time = time.time()

        self._write(f"{'=' * 70}")
        self._write(f"  AGENT RUN: {run_name}")
        self._write(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self._write(f"{'=' * 70}\n")

    def step(self, step_name: str, details: str = "") -> 'StepContext':
        """Log the start of a named step. Use as a context manager.

        Usage:
            with logger.step("Retrieve Knowledge") as s:
                # do work
                s.set_output("Found 3 entries")
        """
        return StepContext(self, step_name, details)

    def log(self, step_name: str, message: str, level: str = "INFO"):
        """Log a single message."""
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] [{level}] [{step_name}] {message}"
        self._write(entry)
        self.steps.append({
            "step": step_name,
            "message": message,
            "level": level,
            "timestamp": timestamp,
        })

    def error(self, step_name: str, message: str):
        """Log an error."""
        self.log(step_name, f"ERROR: {message}", level="ERROR")

    def warn(self, step_name: str, message: str):
        """Log a warning."""
        self.log(step_name, f"WARNING: {message}", level="WARN")

    def section(self, title: str):
        """Print a section divider."""
        self._write(f"\n{'-' * 50}")
        self._write(f"  {title}")
        self._write(f"{'-' * 50}")

    def finish(self):
        """Close the log with a summary."""
        elapsed = time.time() - self.start_time
        self._write(f"\n{'=' * 70}")
        self._write(f"  RUN COMPLETE")
        self._write(f"  Total steps: {len(self.steps)}")
        self._write(f"  Duration: {elapsed:.1f}s")
        self._write(f"  Log file: {self.log_file}")
        self._write(f"{'=' * 70}")

    def get_log_path(self) -> str:
        """Return the path to the log file."""
        return self.log_file

    def get_steps(self) -> list:
        """Return all logged steps."""
        return self.steps.copy()

    def _write(self, text: str):
        """Write to both console and file."""
        print(text)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(text + "\n")


class StepContext:
    """Context manager for logging a timed step."""

    def __init__(self, logger: AgentLogger, step_name: str, details: str):
        self.logger = logger
        self.step_name = step_name
        self.details = details
        self.output = ""
        self.start = 0.0

    def __enter__(self):
        self.start = time.time()
        msg = f"Starting: {self.details}" if self.details else "Starting..."
        self.logger.log(self.step_name, msg)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        if exc_type:
            self.logger.error(self.step_name, f"Failed after {elapsed:.1f}s: {exc_val}")
            return False  # Don't suppress the exception
        else:
            msg = f"Done ({elapsed:.1f}s)"
            if self.output:
                msg += f" — {self.output}"
            self.logger.log(self.step_name, msg)
        return False

    def set_output(self, output: str):
        """Set the output message for this step."""
        self.output = output
