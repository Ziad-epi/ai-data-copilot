from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from app.schemas.datasets import DatasetMetadata


@dataclass
class RagDocument:
    text: str
    metadata: dict[str, Any]


def build_summary_doc(metadata: DatasetMetadata, columns: list[str]) -> RagDocument:
    timestamp = datetime.now(timezone.utc).isoformat()
    col_list = ", ".join(columns)
    parts = [
        f"dataset_id={metadata.dataset_id}",
        f"filename={metadata.filename}",
        f"rows={metadata.nb_rows}",
        f"columns={metadata.nb_columns}",
        f"columns_list={col_list}",
    ]

    numeric_bits: list[str] = []
    for col, stats in metadata.numeric_columns_summary.items():
        if col not in columns:
            continue
        numeric_bits.append(
            f"{col}: min={stats.get('min')} max={stats.get('max')} mean={stats.get('mean')}"
        )
    if numeric_bits:
        parts.append("numeric_summary=" + " | ".join(numeric_bits))

    top_value_bits: list[str] = []
    for col, values in metadata.top_values.items():
        if col not in columns:
            continue
        formatted = ", ".join(f"{item['value']}({item['count']})" for item in values)
        top_value_bits.append(f"{col}: {formatted}")
    if top_value_bits:
        parts.append("top_values=" + " | ".join(top_value_bits))

    if metadata.warnings:
        parts.append("warnings=" + " | ".join(metadata.warnings))

    text = " | ".join(parts)
    return RagDocument(
        text=text,
        metadata={
            "dataset_id": metadata.dataset_id,
            "doc_type": "summary",
            "row_start": None,
            "row_end": None,
            "columns_included": columns,
            "created_at": timestamp,
        },
    )


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return "null"
    return str(value)


def iter_row_docs(
    file_path: Path,
    metadata: DatasetMetadata,
    columns: list[str],
    max_rows: int,
    rows_per_doc: int,
) -> Iterator[RagDocument]:
    timestamp = datetime.now(timezone.utc).isoformat()
    delimiter = metadata.delimiter or ","
    encoding = metadata.encoding or "utf-8"
    row_index = 1

    reader = pd.read_csv(
        file_path,
        sep=delimiter,
        encoding=encoding,
        header=0,
        names=metadata.columns,
        usecols=columns,
        skip_blank_lines=True,
        skipinitialspace=True,
        nrows=max_rows,
        chunksize=rows_per_doc,
    )

    for chunk in reader:
        if chunk.empty:
            continue
        row_start = row_index
        lines: list[str] = []
        for _, row in chunk.iterrows():
            parts = [f"{col}={_format_value(row[col])}" for col in columns]
            lines.append(f"row_index={row_index} | " + " | ".join(parts))
            row_index += 1
        row_end = row_index - 1
        text = "\n".join(lines)
        yield RagDocument(
            text=text,
            metadata={
                "dataset_id": metadata.dataset_id,
                "doc_type": "rows",
                "row_start": row_start,
                "row_end": row_end,
                "columns_included": columns,
                "created_at": timestamp,
            },
        )
