"""Component-local status tracking for AutoML pipelines.

Each component publishes its own stage progress to an artifact. No workspace required.
The dashboard aggregates component statuses to show overall pipeline progress.

Usage:
    status = ComponentStatusTracker(artifact.path, "autogluon_models_training")
    status.record("load_data", "started")
    status.record("load_data", "completed", rows=1000)
    status.record("model_selection", "started")
    status.record("model_selection", "completed",
                 steps=["feature_eng", "training", "stacking"])
    status.save()
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

COMPONENT_STATUS_FILENAME = "component_status.json"

STATUS_STARTED = "started"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


class ComponentStatusTracker:
    """Track stage-level progress within a single component.

    Publishes component-local status to an artifact without requiring workspace.
    Each component independently tracks its stages and metadata.
    """

    def __init__(self, artifact_path: str, component_id: str) -> None:
        """Initialize the status tracker.

        Args:
            artifact_path: Path to the KFP artifact directory where status.json will be written.
            component_id: Unique component identifier (e.g., "autogluon_models_training").
        """
        self.artifact_path = Path(artifact_path)
        self.component_id = component_id
        self.stages: list[dict[str, Any]] = []
        self.started_at = self._utc_now_iso()
        self.metadata: dict[str, Any] = {}

    @staticmethod
    def _utc_now_iso() -> str:
        """Return current UTC timestamp in ISO format."""
        return datetime.now(UTC).isoformat().replace("+00:00", "Z")

    def record(self, stage_id: str, status: str, **metadata: Any) -> None:
        """Record or update a stage's status.

        If the stage already exists, it will be updated. Otherwise, a new stage is appended.

        Args:
            stage_id: Stage identifier (e.g., "load_data", "model_selection").
            status: Stage status ("started", "running", "completed", "failed").
            **metadata: Additional stage data (e.g., rows=1000, steps=["step1", "step2"]).

        Example:
            status.record("load_data", "completed", rows=1000, duration_seconds=5.2)
            status.record("model_selection", "completed",
                         steps=["feature_eng", "training", "stacking"],
                         models_trained=15)
        """
        # Find existing stage or create new one
        existing_idx = next((i for i, s in enumerate(self.stages) if s["id"] == stage_id), None)

        stage_data = {
            "id": stage_id,
            "status": status,
            "timestamp": self._utc_now_iso(),
            **metadata,
        }

        if existing_idx is not None:
            # Update existing stage
            self.stages[existing_idx] = stage_data
        else:
            # Append new stage
            self.stages.append(stage_data)

        logger.info(
            "COMPONENT_STATUS component=%s stage=%s status=%s %s",
            self.component_id,
            stage_id,
            status,
            " ".join(f"{k}={v}" for k, v in metadata.items()),
        )

    def set_metadata(self, **metadata: Any) -> None:
        """Set component-level metadata.

        Args:
            **metadata: Key-value pairs to store at component level.

        Example:
            status.set_metadata(total_training_time_seconds=3600, models_produced=5)
        """
        self.metadata.update(metadata)

    def save(self) -> None:
        """Write the final status to the artifact.

        Creates the artifact directory if needed and writes component_status.json
        with all recorded stages and metadata.
        """
        self.artifact_path.mkdir(parents=True, exist_ok=True)

        data = {
            "component_id": self.component_id,
            "started_at": self.started_at,
            "completed_at": self._utc_now_iso(),
            "stages": self.stages,
            "metadata": self.metadata,
        }

        output_file = self.artifact_path / COMPONENT_STATUS_FILENAME
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(
            "COMPONENT_STATUS component=%s saved status with %d stages to %s",
            self.component_id,
            len(self.stages),
            output_file,
        )


def load_component_status(artifact_path: str) -> dict[str, Any]:
    """Load component status from an artifact.

    Utility function for dashboards/monitoring to read component status.

    Args:
        artifact_path: Path to the artifact directory containing component_status.json.

    Returns:
        Dict containing component_id, started_at, completed_at, stages, and metadata.
        Returns empty dict if file doesn't exist.

    Example:
        status = load_component_status("/path/to/artifact")
        print(f"Component {status['component_id']} completed {len(status['stages'])} stages")
    """
    status_file = Path(artifact_path) / COMPONENT_STATUS_FILENAME
    if not status_file.exists():
        return {}

    with status_file.open("r", encoding="utf-8") as f:
        return json.load(f)
