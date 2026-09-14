# Text Extraction ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Text Extraction component.

Thin wrapper that delegates to ``ai4rag.utils.data.text_extraction.extract_text``.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `documents_descriptor` | `dsl.Input[dsl.Artifact]` | `None` | Input artifact containing documents_descriptor.json with bucket, prefix, and documents list. Each document entry's ``key`` also names the extracted document, so the prefix is not passed on separately. |
| `extracted_text` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact directory where DoclingDocument JSON files will be written. |
| `component_status` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact containing stage-level progress tracking. |
| `embedded_artifact` | `dsl.EmbeddedInput[dsl.Dataset]` | `None` | Embedded ``autorag.shared`` helpers injected by KFP at runtime. |
| `error_tolerance` | `Optional[float]` | `None` | Fraction of documents (0.0-1.0) allowed to fail without raising an error. None (the default) means zero tolerance. |
| `max_extraction_workers` | `Optional[int]` | `None` | Number of parallel worker processes used for text extraction. Defaults to 4. Set to None to use all available CPU cores. |
| `preset` | `str` | `speed` | Pipeline quality tier. "speed" (default) disables Docling table structure parsing. "balanced" enables TableFormer table reconstruction. |
| `do_ocr` | `bool` | `False` | Run RapidOCR on pages Docling flags as needing it (scanned PDFs, images). Off by default: born-digital documents already carry a text layer, so OCR only adds runtime cost. Requires the RapidOCR models to be present under ``$DOCLING_ARTIFACTS_PATH/RapidOcr/``; ai4rag raises ``FileNotFoundError`` when they are missing. |
| `ocr_lang` | `Optional[str]` | `None` | RapidOCR language selection, e.g. "english" or "chinese". There is no auto-detection. Latin-script languages resolve to the English model bundle; only Chinese switches bundles. None uses ai4rag's default ("english"). Ignored when ``do_ocr`` is False. |
| `ocr_det_model_path` | `Optional[str]` | `None` | Optional path to a custom RapidOCR text-detection ONNX model, overriding the bundle selected by ``ocr_lang``. |
| `ocr_cls_model_path` | `Optional[str]` | `None` | Optional path to a custom RapidOCR angle-classification ONNX model. |
| `ocr_rec_model_path` | `Optional[str]` | `None` | Optional path to a custom RapidOCR text-recognition ONNX model. |
| `ocr_rec_keys_path` | `Optional[str]` | `None` | Optional path to the character-keys dictionary matching a custom recognition model. Required when that model uses a non-default character set. |

## Usage Examples 🧪

```python
"""Example pipelines demonstrating usage of text_extraction."""

from kfp import dsl
from kfp_components.components.data_processing.autorag.text_extraction import text_extraction


@dsl.pipeline(name="text-extraction-example")
def example_pipeline():
    """Example pipeline using text_extraction."""
    documents_descriptor = dsl.importer(
        artifact_uri="gs://placeholder/documents_descriptor",
        artifact_class=dsl.Artifact,
    )
    text_extraction(documents_descriptor=documents_descriptor.output)

```

## Metadata 🗂️

- **Name**: text_extraction
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
- **Tags**:
  - data-processing
  - autorag
  - text-extraction
- **Last Verified**: 2026-09-15 00:00:00+00:00
- **Owners**:
  - No Parent Owners: Yes
  - Approvers:
    - LukaszCmielowski
    - DorotaDR
    - Mateusz-Switala
    - filip-komarzyniec
    - jakub-walaszczyk
  - Reviewers:
    - filip-komarzyniec
    - jakub-walaszczyk
    - MichalSteczko
