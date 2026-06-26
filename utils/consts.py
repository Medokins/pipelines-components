import os

DEFAULT_AUTOML_IMAGE = "quay.io/opendatahub/odh-automl:odh-stable"
DEFAULT_AUTORAG_IMAGE = "quay.io/opendatahub/odh-autorag:odh-stable"


def _normalize_image_ref(image: str) -> str:
    """Strip accidental URL schemes from image overrides (e.g. http://quay.io/...)."""
    return image.removeprefix("http://").removeprefix("https://").strip()


def _image_from_env(env_var: str, default: str = "") -> str:
    raw = os.getenv(env_var, default)
    return _normalize_image_ref(raw) if raw else ""


AUTOML_IMAGE = _image_from_env("RELATED_IMAGE_ODH_AUTOML_IMAGE", DEFAULT_AUTOML_IMAGE)
AUTORAG_IMAGE = _image_from_env("RELATED_IMAGE_ODH_AUTORAG_IMAGE", DEFAULT_AUTORAG_IMAGE)
RAY_RAG_BASE_IMAGE = _image_from_env("RELATED_IMAGE_RAG_BASE_RUNTIME")
