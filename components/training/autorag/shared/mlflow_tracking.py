"""MLflow tracking helpers for the AutoRAG optimization component.

Reads the platform-native ``KFP_MLFLOW_CONFIG`` JSON blob that Kubeflow/RHOAI injects
into every pipeline step. No custom connection secret is required: the tracking URI,
parent run, experiment, and workspace all come from that blob, and authentication uses
the pod's mounted Kubernetes service-account token.

Results are logged incrementally. ``MlflowPatternEventHandler`` wraps the ``ai4rag``
event handler and opens a nested child run for each RAG pattern **as the optimizer
finishes evaluating it**, so the MLflow experiment fills in live during ``search()``
instead of being dumped in one batch at the end.

This module is delivered to the runtime image as a KFP embedded artifact, because the
``odh-autorag`` image does not ship the ``kfp_components`` package. It must therefore stay
self-contained: no imports of sibling modules, and no hard dependency on ``mlflow`` or
``ai4rag`` at import time.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

logger = logging.getLogger(__name__)

MlflowMode = Literal["disabled", "kfp"]

# Env var holding the platform-injected MLflow config (JSON). Set by the KFP MLflow
# integration on every step; absent when the platform integration is not enabled.
KFP_MLFLOW_CONFIG_ENV = "KFP_MLFLOW_CONFIG"

# Mounted service-account token used to authenticate to the MLflow server when
# ``authType`` is ``kubernetes``. MLflow sends it as an ``Authorization: Bearer`` header
# via ``MLFLOW_TRACKING_TOKEN`` (the runtime image's MLflow has no built-in kubernetes
# auth provider, so we populate the token ourselves).
SERVICE_ACCOUNT_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
KUBERNETES_AUTH_TYPE = "kubernetes"

# HTTP header RHOAI's multi-tenant MLflow requires to scope every request to a workspace
# (the project namespace, e.g. "ns-autorag"). Sent from the resolved config.
MLFLOW_WORKSPACE_HEADER = "x-mlflow-workspace"

MLFLOW_PARENT_RUN_ID_TAG = "mlflow.parentRunId"

RUN_TYPE_PIPELINE = "pipeline"
RUN_TYPE_PATTERN = "rag_pattern"

# MLflow rejects params longer than this; prompt templates and embedding params can be
# long, so values are truncated rather than dropped.
MAX_PARAM_VALUE_CHARS = 5000

# Free-text prompt fields from ``settings.generation``. Logged as a child-run artifact
# rather than params: they routinely exceed the param length limit and are more useful
# rendered as a file.
GENERATION_PROMPT_KEYS = ("context_template_text", "user_message_text", "system_message_text")

# MLflow permits alphanumerics, underscores, dashes, periods, spaces, and slashes in
# metric keys. Evaluator/metric names from ai4rag are plain identifiers, but sanitize
# defensively so a new metric name can never break a whole run's logging.
_UNSAFE_KEY_CHARS = re.compile(r"[^A-Za-z0-9_\-. /]")


@dataclass(frozen=True)
class MlflowConfig:
    """Resolved MLflow settings for the current pod, parsed from ``KFP_MLFLOW_CONFIG``."""

    mode: MlflowMode
    tracking_uri: str
    experiment_id: str = ""
    run_id: str = ""
    workspace: str = ""
    auth_type: str = ""
    timeout: str = ""


def resolve_mlflow_config() -> MlflowConfig | None:
    """Resolve MLflow config from the platform-injected ``KFP_MLFLOW_CONFIG`` blob.

    Returns ``None`` (tracking disabled) when the env var is absent, is not valid JSON,
    or lacks an ``endpoint``. The blob has the shape::

        {
            "endpoint": "...",
            "workspacesEnabled": true,
            "workspace": "ns-...",
            "parentRunId": "...",
            "experimentId": "2",
            "authType": "kubernetes",
            "timeout": "30s",
        }
    """
    raw = os.getenv(KFP_MLFLOW_CONFIG_ENV, "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("%s is set but is not valid JSON; MLflow logging disabled.", KFP_MLFLOW_CONFIG_ENV)
        return None
    if not isinstance(data, dict):
        logger.warning("%s is not a JSON object; MLflow logging disabled.", KFP_MLFLOW_CONFIG_ENV)
        return None

    tracking_uri = str(data.get("endpoint", "")).strip()
    if not tracking_uri:
        logger.warning("%s has no 'endpoint'; MLflow logging disabled.", KFP_MLFLOW_CONFIG_ENV)
        return None

    auth_type = str(data.get("authType", "")).strip()
    # Kubernetes auth attaches a ServiceAccount bearer token to every request. Refuse to send it
    # over a non-HTTPS endpoint, which would leak the token in cleartext (CWE-319). Disable
    # tracking rather than risk the credential.
    if auth_type == KUBERNETES_AUTH_TYPE and not tracking_uri.lower().startswith("https://"):
        logger.warning(
            "%s uses '%s' auth but 'endpoint' is not HTTPS; refusing to send bearer token over "
            "cleartext. MLflow logging disabled.",
            KFP_MLFLOW_CONFIG_ENV,
            KUBERNETES_AUTH_TYPE,
        )
        return None

    # Only scope requests to a workspace when the server has workspaces enabled.
    workspace = str(data.get("workspace", "")).strip() if data.get("workspacesEnabled") else ""
    return MlflowConfig(
        mode="kfp",
        tracking_uri=tracking_uri,
        experiment_id=str(data.get("experimentId", "")).strip(),
        run_id=str(data.get("parentRunId", "")).strip(),
        workspace=workspace,
        auth_type=auth_type,
        timeout=str(data.get("timeout", "")).strip(),
    )


def is_mlflow_enabled() -> bool:
    """Return True when the platform injected a usable MLflow tracking config."""
    return resolve_mlflow_config() is not None


def build_mlflow_run_url(tracking_uri: str, experiment_id: str, run_id: str) -> str:
    """Build a deep-link URL to the MLflow UI parent run view."""
    base = tracking_uri.rstrip("/")
    if not base or not experiment_id or not run_id:
        return ""
    return f"{base}/#/experiments/{experiment_id}/runs/{run_id}"


def build_mlflow_stage_map_block(
    *,
    tracking_uri: str | None = None,
    experiment_id: str | None = None,
    run_id: str | None = None,
    workspace: str | None = None,
) -> dict[str, Any]:
    """Build the ``mlflow`` object embedded in ``component_stage_map.json``.

    Consumed by the odh-dashboard AutoRAG BFF to deep-link the run into the MLflow UI.
    Values default to the resolved ``KFP_MLFLOW_CONFIG`` blob; explicit arguments override.
    """
    config = resolve_mlflow_config()
    uri = (tracking_uri if tracking_uri is not None else (config.tracking_uri if config else "")).strip()
    if not uri:
        return {"tracking_enabled": False}

    exp_id = (experiment_id if experiment_id is not None else (config.experiment_id if config else "")).strip()
    parent_run_id = (run_id if run_id is not None else (config.run_id if config else "")).strip()
    ws = (workspace if workspace is not None else (config.workspace if config else "")).strip()

    block: dict[str, Any] = {"tracking_enabled": True, "tracking_uri": uri}
    if exp_id:
        block["experiment_id"] = exp_id
    if parent_run_id:
        block["run_id"] = parent_run_id
    if ws:
        block["workspace"] = ws
    run_url = build_mlflow_run_url(uri, exp_id, parent_run_id)
    if run_url:
        block["run_url"] = run_url
    return block


def resolve_leaderboard_html_path(html_artifact_path: str | Path) -> Path | None:
    """Resolve the leaderboard HTML file from a KFP ``dsl.HTML`` artifact path.

    KFP may mount the artifact as a file path or as a directory containing the HTML.
    """
    path = Path(html_artifact_path)
    if path.is_file():
        return path
    if path.is_dir():
        for candidate in (path / "index.html", path / "leaderboard.html"):
            if candidate.is_file():
                return candidate
        html_files = sorted(path.glob("*.html"))
        if len(html_files) == 1:
            return html_files[0]
    return None


# ---------------------------------------------------------------------------
# Client configuration
# ---------------------------------------------------------------------------


def configure_mlflow_client(mlflow: Any, config: MlflowConfig) -> None:
    """Apply authentication, tracking URI, and workspace before MLflow API calls."""
    if config.auth_type == KUBERNETES_AUTH_TYPE:
        _apply_kubernetes_auth()
    timeout_seconds = _parse_timeout_seconds(config.timeout)
    if timeout_seconds is not None:
        # MLflow reads MLFLOW_HTTP_REQUEST_TIMEOUT as a whole-second integer.
        os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = str(timeout_seconds)
    mlflow.set_tracking_uri(config.tracking_uri)
    if config.workspace:
        _apply_workspace(mlflow, config.tracking_uri, config.workspace)


def _apply_kubernetes_auth() -> None:
    """Authenticate to MLflow with the pod's Kubernetes service-account token.

    MLflow reads ``MLFLOW_TRACKING_TOKEN`` and sends it as an ``Authorization: Bearer``
    header. Best-effort: a missing/unreadable token leaves auth unset and is surfaced by
    the eventual request failure rather than crashing here.
    """
    token_path = Path(SERVICE_ACCOUNT_TOKEN_PATH)
    try:
        token = token_path.read_text(encoding="utf-8").strip()
    except OSError:
        logger.warning(
            "authType=kubernetes but service-account token is not readable at %s; "
            "MLflow requests will be unauthenticated.",
            token_path,
        )
        return
    if not token:
        logger.warning("Service-account token at %s is empty; MLflow requests will be unauthenticated.", token_path)
        return
    os.environ["MLFLOW_TRACKING_TOKEN"] = token


def _apply_workspace(mlflow: Any, tracking_uri: str, workspace: str) -> None:
    """Route the workspace to the MLflow server on every request.

    RHOAI's multi-tenant MLflow rejects any call that does not identify a workspace
    (``INVALID_PARAMETER_VALUE: Workspace context is required``). Newer MLflow clients
    expose ``mlflow.set_workspace``; upstream/plain MLflow does not, so we attach the
    ``x-mlflow-workspace`` header on the underlying ``requests`` session instead.
    Best-effort: a failure here leaves tracking disabled but never fails the pipeline.
    """
    set_workspace = getattr(mlflow, "set_workspace", None)
    if callable(set_workspace):
        set_workspace(workspace)
        return
    try:
        _install_workspace_request_header(tracking_uri, workspace)
    except Exception:
        logger.exception("Failed to install MLflow workspace request header for %r.", workspace)


def _install_workspace_request_header(tracking_uri: str, workspace: str) -> None:
    """Monkeypatch ``requests`` so calls to the tracking host carry the workspace header.

    Scoped to the tracking host so it never affects artifact stores (e.g. S3) on other
    hosts. Idempotent: re-applying in the same process is a no-op.
    """
    from urllib.parse import urlsplit

    import requests

    host = urlsplit(tracking_uri).netloc
    if not host:
        return

    original_request = requests.Session.request
    if getattr(original_request, "_autorag_workspace_patch", False):
        return

    def request_with_workspace(self, method, url, *args, **kwargs):
        try:
            if urlsplit(url).netloc == host:
                headers = dict(kwargs.get("headers") or {})
                headers.setdefault(MLFLOW_WORKSPACE_HEADER, workspace)
                kwargs["headers"] = headers
        except Exception:
            logger.debug("Could not attach MLflow workspace header", exc_info=True)
        return original_request(self, method, url, *args, **kwargs)

    request_with_workspace._autorag_workspace_patch = True
    requests.Session.request = request_with_workspace


def _parse_timeout_seconds(timeout: str) -> int | None:
    """Parse a platform timeout like ``"30s"`` / ``"30"`` into whole seconds, or ``None``."""
    text = (timeout or "").strip().lower()
    if not text:
        return None
    if text.endswith("s"):
        text = text[:-1].strip()
    try:
        seconds = int(float(text))
    except ValueError:
        logger.debug("Could not parse MLflow timeout value %r; ignoring.", timeout)
        return None
    return seconds if seconds > 0 else None


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------


@contextmanager
def parent_mlflow_run(mlflow: Any, config: MlflowConfig, *, fallback_name: str = "") -> Iterator[Any]:
    """Open the parent run for nested logging.

    When the platform provides a ``parentRunId`` (the native RHOAI pipeline-submission
    path), that run is resumed so every AutoRAG job lands in the experiment the user
    picked. When it does not, a parent run is created instead, under an experiment named
    ``fallback_name`` (typically the KFP run name, which is unique per job), get-or-created
    when the platform also supplied no ``experimentId``.
    """
    configure_mlflow_client(mlflow, config)
    if config.run_id:
        with mlflow.start_run(run_id=config.run_id) as run:
            yield run
        return

    # No platform parent run: create our own. Bind an experiment first so the run does not
    # land in MLflow's Default experiment when the platform gave us no experiment id.
    if not config.experiment_id and fallback_name:
        try:
            mlflow.set_experiment(fallback_name)
        except Exception:
            logger.exception("Could not set/create MLflow experiment %r; using the default.", fallback_name)
    start_kwargs: dict[str, Any] = {}
    if config.experiment_id:
        start_kwargs["experiment_id"] = config.experiment_id
    if fallback_name:
        start_kwargs["run_name"] = fallback_name
    with mlflow.start_run(**start_kwargs) as run:
        yield run


@contextmanager
def _child_mlflow_run(
    mlflow: Any,
    *,
    experiment_id: str,
    parent_run_id: str,
    run_name: str,
    tags: dict[str, str],
) -> Iterator[Any]:
    """Open a nested child run, falling back to explicit parent linkage when needed."""
    try:
        started = mlflow.start_run(run_name=run_name, nested=True)
    except Exception as exc:
        logger.warning(
            "MLflow nested child run failed for %s (%s); trying MlflowClient.create_run.",
            run_name,
            exc,
        )
    else:
        # yield outside the except so caller-body exceptions propagate unchanged
        # instead of being caught here and triggering the create_run fallback.
        with started as run:
            yield run
        return

    # MlflowClient.create_run takes tags as a plain dict (it builds RunTag objects
    # internally); passing a list of RunTag raises AttributeError.
    run_tags = {MLFLOW_PARENT_RUN_ID_TAG: parent_run_id, **tags}

    client = mlflow.MlflowClient()
    created_run = client.create_run(experiment_id=experiment_id, run_name=run_name, tags=run_tags)
    try:
        # nested=True: the parent run is still active, so reopening this child by run_id
        # would otherwise be rejected by MLflow.
        with mlflow.start_run(run_id=created_run.info.run_id, nested=True) as run:
            yield run
    except Exception as exc:
        client.set_terminated(created_run.info.run_id, status="FAILED")
        raise exc


# ---------------------------------------------------------------------------
# Pattern payload mapping
# ---------------------------------------------------------------------------


def _safe_key(name: str) -> str:
    """Return ``name`` with characters MLflow rejects in metric/param keys removed."""
    return _UNSAFE_KEY_CHARS.sub("_", str(name)).strip() or "unnamed"


def _stringify_params(params: dict[str, Any]) -> dict[str, str]:
    """MLflow params must be strings, and are rejected above a length limit."""
    out: dict[str, str] = {}
    for key, value in params.items():
        if value is None:
            continue
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        if len(text) > MAX_PARAM_VALUE_CHARS:
            text = text[: MAX_PARAM_VALUE_CHARS - 3] + "..."
        out[_safe_key(key)] = text
    return out


def pattern_params(payload: dict[str, Any]) -> dict[str, str]:
    """Flatten a RAG pattern's ``settings`` into MLflow params.

    The free-text generation prompts are excluded; they are logged as a child-run
    artifact instead (see ``GENERATION_PROMPT_KEYS``).
    """
    settings = payload.get("settings") or {}
    chunking = settings.get("chunking") or {}
    embedding = settings.get("embedding") or {}
    retrieval = settings.get("retrieval") or {}
    generation = settings.get("generation") or {}
    vector_store = settings.get("vector_store_binding") or {}

    params: dict[str, Any] = {
        "pattern_name": payload.get("name"),
        "iteration": payload.get("iteration"),
        "max_combinations": payload.get("max_combinations"),
        "chunking.method": chunking.get("method"),
        "chunking.chunk_size": chunking.get("chunk_size"),
        "chunking.chunk_overlap": chunking.get("chunk_overlap"),
        "embedding.model_id": embedding.get("model_id"),
        "generation.model_id": generation.get("model_id"),
        "vector_store.provider_type": vector_store.get("provider_type"),
        "vector_store.collection_name": vector_store.get("collection_name"),
    }
    if embedding.get("embedding_params"):
        params["embedding.embedding_params"] = embedding["embedding_params"]
    # Retrieval keys are conditional: window_size only for the window method, ranker_*
    # only for hybrid search. Copy whatever ai4rag actually emitted.
    for key, value in retrieval.items():
        params[f"retrieval.{key}"] = value
    return _stringify_params(params)


def pattern_metrics(payload: dict[str, Any]) -> dict[str, float]:
    """Extract the per-pattern scores from ``evaluation.metrics`` as MLflow metrics.

    Each metric contributes ``{evaluator}_{name}`` plus its confidence-interval bounds
    when ai4rag computed them. ``duration_seconds`` is logged alongside so pattern cost
    is comparable in the same view as pattern quality.
    """
    metrics: dict[str, float] = {}
    evaluation = payload.get("evaluation") or {}
    for entry in evaluation.get("metrics") or []:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not name:
            continue
        base = _safe_key(f"{entry.get('evaluator', 'unknown')}_{name}")
        scores = entry.get("scores") or {}
        for suffix, score_key in (("", "mean"), ("_ci_low", "ci_low"), ("_ci_high", "ci_high")):
            value = scores.get(score_key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[f"{base}{suffix}"] = float(value)

    duration = payload.get("duration_seconds")
    if isinstance(duration, (int, float)) and not isinstance(duration, bool):
        metrics["duration_seconds"] = float(duration)
    return metrics


def optimization_metric_key(payload: dict[str, Any]) -> str:
    """Return the ``{evaluator}_{name}`` key of the metric the optimizer selected on."""
    evaluation = payload.get("evaluation") or {}
    for entry in evaluation.get("metrics") or []:
        if isinstance(entry, dict) and entry.get("optimization_metric"):
            return _safe_key(f"{entry.get('evaluator', 'unknown')}_{entry.get('name', '')}")
    return ""


def pattern_score(payload: dict[str, Any]) -> float | None:
    """Return the pattern's score on the optimization metric, or ``None``."""
    evaluation = payload.get("evaluation") or {}
    for entry in evaluation.get("metrics") or []:
        if isinstance(entry, dict) and entry.get("optimization_metric"):
            mean = (entry.get("scores") or {}).get("mean")
            if isinstance(mean, (int, float)) and not isinstance(mean, bool):
                return float(mean)
    return None


def generation_prompts(payload: dict[str, Any]) -> dict[str, str]:
    """Return the pattern's generation prompt templates, for artifact upload."""
    generation = (payload.get("settings") or {}).get("generation") or {}
    return {key: str(generation[key]) for key in GENERATION_PROMPT_KEYS if generation.get(key)}


# ---------------------------------------------------------------------------
# Version / image resolution
# ---------------------------------------------------------------------------


def _resolve_package_version(*packages: str) -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        for package in packages:
            try:
                return version(package)
            except PackageNotFoundError:
                continue
    except Exception:
        logger.debug("Could not resolve version for %s", packages, exc_info=True)
    return "unknown"


def _resolve_image() -> str:
    """Best-effort AutoRAG container image reference for the parent run.

    The odh-autorag image does not ship ``kfp_components``, so prefer the env var the
    pipeline build sets and fall back to the package constant for local runs.
    """
    image = os.getenv("RELATED_IMAGE_ODH_AUTORAG_IMAGE", "").strip()
    if image:
        return image
    try:
        from kfp_components.utils.consts import AUTORAG_IMAGE  # pyright: ignore[reportMissingImports]

        return str(AUTORAG_IMAGE)
    except Exception:
        logger.debug("Could not resolve AUTORAG_IMAGE for MLflow logging", exc_info=True)
        return ""


def _safe_log_artifact(mlflow: Any, file_path: Path, artifact_path: str) -> bool:
    """Upload a local file to MLflow, logging and continuing on failure."""
    if not file_path.is_file():
        return False
    try:
        mlflow.log_artifact(str(file_path), artifact_path=artifact_path)
        return True
    except Exception:
        logger.exception("Failed to upload MLflow artifact %s to %s", file_path, artifact_path)
        return False


def _write_temp_json(tmp_dir: Path, filename: str, payload: Any) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    output = tmp_dir / filename
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return output


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


class MlflowPatternLogger:
    """Incremental MLflow logger for the AutoRAG optimization component.

    Opens a nested child run per RAG pattern **as the optimizer finishes evaluating it**,
    so the experiment updates live during ``search()`` instead of being dumped in one
    batch at the end. It is:

    - **Null-safe**: when MLflow is disabled every method is a no-op, so callers need no
      ``if enabled`` guards.
    - **Best-effort**: each method swallows its own exceptions and logs them, so tracking
      problems never fail the surrounding optimization step.

    Typical usage (inside the optimization component)::

        with experiment_run_logger(optimization_metric=..., preset=...) as run_logger:
            run_logger.log_header(pipeline_name=..., kfp_run_id=..., ...)
            event_handler = MlflowPatternEventHandler(KFPEventHandler(), run_logger)
            experiment = AI4RAGExperiment(event_handler=event_handler, ...)
            experiment.search()
            run_logger.finalize(html_artifact_path=...)
        logged, tracking_info = run_logger.result()
    """

    def __init__(
        self,
        mlflow: Any,
        config: MlflowConfig | None,
        *,
        optimization_metric: str = "",
        log_pattern_artifacts: bool = True,
        log_evaluation_artifacts: bool = False,
    ) -> None:
        """Store MLflow handles and tracking config; disabled when either is missing."""
        self._mlflow = mlflow
        self._config = config
        self._optimization_metric = optimization_metric
        self._log_pattern_artifacts = log_pattern_artifacts
        self._log_evaluation_artifacts = log_evaluation_artifacts
        self.enabled = mlflow is not None and config is not None
        # MLflow was injected by the platform (KFP_MLFLOW_CONFIG present). Distinct from
        # ``enabled``, which also requires the mlflow package and an open parent run -- so
        # callers can tell "tracking is off" apart from "tracking is on but broken".
        self.configured = config is not None
        self.parent_run_id = ""
        self.experiment_id = config.experiment_id if config else ""
        self._child_run_ids: list[str] = []
        self._child_run_errors: list[str] = []
        self._header_logged = False
        self._finalize_logged = False
        self._tracking_errors: list[str] = []
        # Best pattern seen so far, tracked incrementally so finalize() does not depend on
        # the component having written its output artifacts yet.
        self._best_score: float | None = None
        self._best_pattern_name = ""
        self._pattern_count = 0
        self._total_pattern_seconds = 0.0

    # -- lifecycle -------------------------------------------------------

    def bind_parent_run(self, run: Any) -> None:
        """Record the parent run's identifiers once it has been opened."""
        info = getattr(run, "info", None)
        if info is None:
            return
        self.parent_run_id = str(getattr(info, "run_id", "") or "")
        experiment_id = str(getattr(info, "experiment_id", "") or "")
        if experiment_id:
            self.experiment_id = experiment_id

    def log_header(
        self,
        *,
        pipeline_name: str = "",
        kfp_run_id: str = "",
        kfp_run_name: str = "",
        preset: str = "",
        optimization_metric: str = "",
        max_rag_patterns: int = 0,
        active_evaluators: Any = (),
        embedding_models: Any = (),
        generation_models: Any = (),
        test_data_key: str = "",
        input_data_bucket_name: str = "",
        input_data_keys: Any = (),
    ) -> None:
        """Log the job-level params and tags onto the parent run.

        ``optimization_metric`` may be passed here rather than to the constructor: the
        component only resolves the evaluator-qualified metric after the search space has
        been rebuilt, which happens once the parent run is already open.
        """
        if optimization_metric:
            self._optimization_metric = optimization_metric
        if not self.enabled:
            return
        try:
            params = _stringify_params(
                {
                    "pipeline_name": pipeline_name,
                    "kfp_run_id": kfp_run_id,
                    "kfp_run_name": kfp_run_name,
                    "preset": preset,
                    "optimization_metric": self._optimization_metric,
                    "max_number_of_rag_patterns": max_rag_patterns,
                    "evaluators": ",".join(sorted(str(e) for e in active_evaluators)),
                    "embedding_model_candidates": list(embedding_models),
                    "generation_model_candidates": list(generation_models),
                    "test_data_key": test_data_key,
                    "input_data_bucket_name": input_data_bucket_name,
                    "input_data_keys": list(input_data_keys),
                    "ai4rag_version": _resolve_package_version("ai4rag"),
                    "kfp_version": _resolve_package_version("kfp", "kfp-server-api"),
                    "mlflow_version": _resolve_package_version("mlflow", "mlflow-skinny"),
                    "image": _resolve_image(),
                }
            )
            self._mlflow.log_params(params)
            self._mlflow.set_tags(
                {
                    "run_type": RUN_TYPE_PIPELINE,
                    "pipeline_name": pipeline_name,
                    "kfp_run_id": kfp_run_id,
                    "kfp_run_name": kfp_run_name,
                    "optimization_metric": self._optimization_metric,
                    "preset": preset,
                }
            )
            self._header_logged = True
        except Exception as exc:
            self.record_tracking_error(f"log_header: {exc}")
            logger.exception("MLflow parent header logging failed; continuing.")

    def log_pattern(self, payload: dict[str, Any], evaluation_results: Any = None) -> None:
        """Open and close a nested child run for one evaluated RAG pattern.

        Called from :class:`MlflowPatternEventHandler` the moment ai4rag finishes
        evaluating a pattern, so the experiment fills in live.
        """
        if not self.enabled:
            return

        name = str(payload.get("name") or f"pattern_{self._pattern_count}")
        # Track aggregates even if the child-run write fails, so finalize() still reports
        # the best pattern the optimizer actually found.
        self._pattern_count += 1
        score = pattern_score(payload)
        if score is not None and (self._best_score is None or score > self._best_score):
            self._best_score = score
            self._best_pattern_name = name
        duration = payload.get("duration_seconds")
        if isinstance(duration, (int, float)) and not isinstance(duration, bool):
            self._total_pattern_seconds += float(duration)

        try:
            tags = {
                "run_type": RUN_TYPE_PATTERN,
                "pattern_name": name,
                "optimization_metric": optimization_metric_key(payload) or self._optimization_metric,
            }
            with _child_mlflow_run(
                self._mlflow,
                experiment_id=self.experiment_id,
                parent_run_id=self.parent_run_id or (self._config.run_id if self._config else ""),
                run_name=name,
                tags=tags,
            ) as child:
                self._mlflow.set_tags(tags)
                self._mlflow.log_params(pattern_params(payload))
                metrics = pattern_metrics(payload)
                if metrics:
                    self._mlflow.log_metrics(metrics)
                self._log_pattern_artifacts_for(name, payload, evaluation_results)
                child_info = getattr(child, "info", None)
                child_run_id = str(getattr(child_info, "run_id", "") or "")
                if child_run_id:
                    self._child_run_ids.append(child_run_id)
        except Exception as exc:
            self._child_run_errors.append(f"{name}: {exc}")
            logger.exception("MLflow logging failed for pattern %s; continuing.", name)

    def _log_pattern_artifacts_for(self, name: str, payload: dict[str, Any], evaluation_results: Any) -> None:
        """Upload the pattern's JSON artifacts to its child run.

        ``evaluation_results`` holds verbatim questions, generated answers, and retrieved
        chunk text from the customer's documents, so it is uploaded only when the caller
        explicitly opts in via ``log_evaluation_artifacts``.
        """
        if not self._log_pattern_artifacts:
            return
        # The optimizer has not written its output directory yet at callback time, so
        # serialize the in-memory payload to a scratch dir instead of reading from disk.
        with tempfile.TemporaryDirectory(prefix="mlflow-autorag-") as tmp:
            tmp_dir = Path(tmp)
            artifact_path = f"patterns/{_safe_key(name)}"
            _safe_log_artifact(self._mlflow, _write_temp_json(tmp_dir, "pattern.json", payload), artifact_path)
            prompts = generation_prompts(payload)
            if prompts:
                _safe_log_artifact(
                    self._mlflow, _write_temp_json(tmp_dir, "generation_prompts.json", prompts), artifact_path
                )
            if self._log_evaluation_artifacts and evaluation_results:
                _safe_log_artifact(
                    self._mlflow,
                    _write_temp_json(tmp_dir, "evaluation_results.json", evaluation_results),
                    artifact_path,
                )

    def finalize(self, *, html_artifact_path: str | Path | None = None) -> None:
        """Log the job-level aggregates and the leaderboard onto the parent run."""
        if not self.enabled:
            return
        try:
            metrics: dict[str, float] = {"rag_pattern_count": float(self._pattern_count)}
            if self._best_score is not None:
                metrics["best_pattern_score"] = self._best_score
            if self._total_pattern_seconds:
                metrics["total_pattern_duration_seconds"] = self._total_pattern_seconds
            self._mlflow.log_metrics(metrics)
            if self._best_pattern_name:
                self._mlflow.log_param("best_pattern_name", self._best_pattern_name)
                self._mlflow.set_tag("best_pattern_name", self._best_pattern_name)
            if html_artifact_path is not None:
                leaderboard = resolve_leaderboard_html_path(html_artifact_path)
                if leaderboard is not None:
                    _safe_log_artifact(self._mlflow, leaderboard, "leaderboard")
            self._finalize_logged = True
        except Exception as exc:
            self.record_tracking_error(f"finalize: {exc}")
            logger.exception("MLflow finalize step failed; continuing.")

    def record_tracking_error(self, reason: str) -> None:
        """Record why MLflow tracking could not persist results (surfaced by ``result()``)."""
        self._tracking_errors.append(reason)

    def result(self) -> tuple[bool, dict[str, str]]:
        """Return ``(logged, tracking_info)`` for recording on the component status.

        ``logged`` is True only when something actually reached MLflow -- the parent header,
        the parent finalization, or at least one child run. Because every logging method
        swallows its own exceptions, a configured-but-unreachable tracking server would
        otherwise be reported as a successful log. When ``logged`` is False but tracking was
        configured, ``tracking_info["mlflow_tracking_error"]`` carries the reason.
        """
        if self._config is None:
            return False, {}
        persisted = bool(self._header_logged or self._finalize_logged or self._child_run_ids)
        run_id_for_url = self.parent_run_id or self._config.run_id
        tracking_info: dict[str, str] = {
            "mlflow_run_id": run_id_for_url,
            "mlflow_experiment_id": self.experiment_id,
            "tracking_mode": self._config.mode,
            "mlflow_child_run_ids": ",".join(self._child_run_ids),
            "mlflow_child_run_count": str(len(self._child_run_ids)),
        }
        run_url = build_mlflow_run_url(self._config.tracking_uri, self.experiment_id, run_id_for_url)
        if run_url:
            tracking_info["mlflow_run_url"] = run_url
        if self._child_run_errors:
            tracking_info["mlflow_child_run_errors"] = json.dumps(self._child_run_errors)
        if not persisted:
            reasons = self._tracking_errors + self._child_run_errors
            tracking_info["mlflow_tracking_error"] = "; ".join(reasons) or "no MLflow write succeeded"
        return persisted, tracking_info


# ---------------------------------------------------------------------------
# ai4rag event-handler callback
# ---------------------------------------------------------------------------


class MlflowPatternEventHandler:
    """ai4rag event handler that mirrors each evaluated pattern into MLflow.

    Wraps (rather than subclasses) the component's real handler so this module stays
    importable without ``ai4rag`` installed -- ``AI4RAGExperiment`` only duck-types the
    handler, it never isinstance-checks it. Every call is forwarded to the wrapped handler
    first, so the component's existing artifact generation is unaffected even if MLflow
    logging raises.
    """

    def __init__(self, inner: Any, run_logger: MlflowPatternLogger) -> None:
        """Wrap ``inner`` (the component's real handler) and mirror patterns to ``run_logger``."""
        self._inner = inner
        self._run_logger = run_logger

    # -- BaseEventHandler interface --------------------------------------

    def on_status_change(self, level: Any, message: str, step: str | None = None) -> None:
        """Forward an optimizer status update to the wrapped handler."""
        self._inner.on_status_change(level=level, message=message, step=step)

    def on_pattern_creation(self, payload: dict[str, Any], evaluation_results: Any, **kwargs: Any) -> None:
        """Forward an evaluated pattern to the wrapped handler, then log it to MLflow."""
        self._inner.on_pattern_creation(payload=payload, evaluation_results=evaluation_results, **kwargs)
        try:
            self._run_logger.log_pattern(payload, evaluation_results)
        except Exception:
            # log_pattern is already best-effort; this guard covers programming errors in
            # the mapping helpers so optimization never fails because of tracking.
            logger.exception("MLflow pattern callback failed; continuing optimization.")

    # -- passthrough for the wrapped handler's collected state -----------

    @property
    def patterns(self) -> list[dict[str, Any]]:
        """Patterns collected by the wrapped handler."""
        return self._inner.patterns

    @property
    def status_changes(self) -> list[dict]:
        """Status changes collected by the wrapped handler."""
        return self._inner.status_changes

    def __getattr__(self, item: str) -> Any:
        """Forward any other attribute access to the wrapped handler."""
        return getattr(self._inner, item)


@contextmanager
def experiment_run_logger(
    *,
    optimization_metric: str = "",
    run_name: str = "",
    log_pattern_artifacts: bool = True,
    log_evaluation_artifacts: bool = False,
) -> Iterator[MlflowPatternLogger]:
    """Yield an :class:`MlflowPatternLogger` bound to the parent run.

    Resolves ``KFP_MLFLOW_CONFIG`` and imports MLflow lazily. When the platform supplies a
    ``parentRunId`` that run is resumed; otherwise a parent run (and, if needed, an
    experiment) named ``run_name`` is created. When tracking is disabled or the parent run
    cannot be opened, a disabled (no-op) logger is yielded so the caller proceeds unchanged.
    The parent run is closed when the ``with`` block exits.
    """
    config = resolve_mlflow_config()
    mlflow_mod: Any = None
    if config is not None:
        try:
            import mlflow as mlflow_mod  # type: ignore[no-redef]
        except ImportError:
            logger.warning("mlflow package is not installed in the runtime image; skipping MLflow logging.")
            mlflow_mod = None

    run_logger = MlflowPatternLogger(
        mlflow_mod,
        config,
        optimization_metric=optimization_metric,
        log_pattern_artifacts=log_pattern_artifacts,
        log_evaluation_artifacts=log_evaluation_artifacts,
    )
    if not run_logger.enabled:
        if config is not None:
            # Configured but unusable -- report it as failed tracking, not as tracking off.
            run_logger.record_tracking_error("mlflow package is not installed in the runtime image")
        yield run_logger
        return

    parent_cm = parent_mlflow_run(mlflow_mod, config, fallback_name=run_name)
    try:
        parent_run = parent_cm.__enter__()
    except Exception as exc:
        logger.exception("Failed to open MLflow parent run; disabling tracking for this step.")
        run_logger.enabled = False
        run_logger.record_tracking_error(f"parent run could not be opened: {exc}")
        yield run_logger
        return

    run_logger.bind_parent_run(parent_run)
    try:
        yield run_logger
    finally:
        try:
            parent_cm.__exit__(None, None, None)
        except Exception:
            logger.exception("Error while closing MLflow parent run.")
