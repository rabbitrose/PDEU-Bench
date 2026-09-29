"""Global offline SandboxDB lookup with argument-sensitive retrieval."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import threading
import time
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SANDBOX_DIR = PROJECT_ROOT / "SandboxDB"
DATABASE_DIR = SANDBOX_DIR / "database"
SOURCE_INFO_DIR = SANDBOX_DIR / "source_info"
VECTOR_INDEX_DIR = SANDBOX_DIR / "vector_index"

# Sandbox semantic retrieval settings. Corpus embeddings are built explicitly
# by scripts/build_sandbox_e5_index.py; evaluation only loads those vectors.
E5_MODEL_ID = "intfloat/multilingual-e5-small"
E5_MODEL_DIR = SANDBOX_DIR / "models" / "multilingual-e5-small"
E5_MAX_SEQUENCE_LENGTH = 512
E5_GPU_DOCUMENT_BATCH_SIZE = 128
E5_CPU_DOCUMENT_BATCH_SIZE = 32
E5_MIN_COSINE_SIMILARITY = 0.80
E5_QUERY_CACHE_SIZE = 4096
VECTOR_INDEX_SCHEMA_VERSION = "SandboxE5VectorIndex.v1"
VECTOR_INDEX_BUILD_COMMAND = "uv run python scripts/build_sandbox_e5_index.py"

CANDIDATE_LIST_KEYS = {
    "items",
    "contacts",
    "places",
    "products",
    "restaurants",
    "recipes",
    "meal_options",
    "lodging_options",
    "flight_options",
    "flights",
    "forecast",
    "route_options",
    "transportation_options",
    "providers",
    "courses",
    "resources",
    "events",
    "venues",
    "sources",
    "tools",
    "options",
    "vendors",
    "experts",
    "messages",
    "documents",
    "assets",
    "transactions",
    "accounts",
    "services",
    "guidelines",
    "requirements",
    "actionable_findings",
}

SOURCE_RESULT_KEYS = {
    "calendar.search_events": "events",
    "contacts.search": "contacts",
    "education.search_courses": "courses",
    "education.build_study_plan": "resources",
    "shopping.check_inventory": "products",
    "shopping.search_products": "products",
    "food.search_restaurants": "restaurants",
    "maps.search_places": "places",
    "travel.search_flights": "flights",
    "travel.search_lodging": "lodging_options",
    "weather.get_forecast": "forecast",
    "entertainment.search_events": "events",
}

CATEGORY_TO_THEME = {
    "calendar": "calendar",
    "email": "email",
    "health": "health",
    "entertainment": "music_media",
    "media": "music_media",
    "maps": "navigation",
    "travel": "navigation",
    "weather": "weather",
    "research": "web_browsing",
    "shopping": "shopping",
    "procurement": "shopping",
    "communications": "communication",
    "contacts": "communication",
    "messaging": "communication",
    "documents": "documents",
    "education": "education",
    "events": "events",
    "finance": "finance",
    "food": "food",
    "smart_home": "home",
    "legal": "legal",
    "project": "project",
    "security": "security",
    "technology": "technology",
    "translation": "translation",
    "civic": "civic",
    "privacy": "privacy",
    "notes": "documents",
    "reminders": "calendar",
}

THEME_TO_SOURCE_DOMAINS = {
    "shopping": ["shopping"],
    "music_media": ["music_media"],
    "navigation": ["navigation", "food", "travel_flights", "weather"],
    "food": ["food", "navigation"],
    "weather": ["weather"],
    "events": ["navigation", "food", "weather"],
    "web_browsing": ["shopping", "music_media", "navigation", "food", "travel_flights", "weather"],
    "research": ["shopping", "music_media", "navigation", "food", "travel_flights", "weather"],
    "documents": ["shopping", "music_media", "navigation", "food", "travel_flights", "weather"],
    "education": [],
    "finance": ["shopping", "travel_flights", "navigation"],
    "technology": ["shopping"],
}

def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def stable_sequence_hash(values: Any) -> str:
    """Hash an ordered sequence without materializing one large JSON value."""
    digest = hashlib.sha256()
    for value in values:
        payload = canonical(value).encode("utf-8")
        digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
        digest.update(payload)
    return digest.hexdigest()


def flatten_text(value: Any, *, max_chars: int = 12000) -> str:
    parts: list[str] = []

    def visit(item: Any) -> None:
        if len(" ".join(parts)) > max_chars:
            return
        if isinstance(item, dict):
            for key, subvalue in item.items():
                parts.append(str(key))
                visit(subvalue)
        elif isinstance(item, list):
            for subvalue in item:
                visit(subvalue)
        elif item is not None:
            parts.append(str(item))

    visit(value)
    text = " ".join(parts)
    return text[:max_chars]


SEMANTIC_QUERY_ARGUMENTS = (
    "query",
    "topic",
    "learning_goal",
    "interests",
    "media_type",
    "mood",
    "title",
)


def semantic_query_text(arguments: dict[str, Any]) -> str:
    intent_values = [
        arguments[name]
        for name in SEMANTIC_QUERY_ARGUMENTS
        if name in arguments and not _is_empty(arguments[name])
    ]
    if intent_values:
        return flatten_text(intent_values, max_chars=4000)
    return flatten_text(arguments, max_chars=4000)


UPPER_BOUND_ARGUMENTS = {
    "budget_limit",
    "max_nightly_rate",
    "max_price",
    "max_total",
}
LOWER_BOUND_ARGUMENTS = {
    "min_rating",
    "minimum_rating",
    "minimum_review_count",
    "rating_threshold",
}
EXACT_ARGUMENTS = {
    "cabin_class",
    "category",
    "check_in",
    "check_out",
    "date_range",
    "departure_date",
    "destination",
    "destination_airport",
    "format",
    "language",
    "level",
    "location",
    "media_type",
    "mood",
    "near",
    "origin",
    "origin_airport",
    "pickup_or_delivery",
    "privacy_mode",
    "return_date",
    "start_date",
    "timezone",
    "units",
}
REQUIRED_SET_ARGUMENTS = {
    "accessibility_needs",
    "amenities",
    "dietary_constraints",
    "must_have_features",
    "product_ids",
}
EXCLUDED_SET_ARGUMENTS = {
    "avoid_airlines",
    "avoid_features",
}
CAPACITY_ARGUMENTS = {
    "group_size",
    "guests",
    "party_size",
}

# Rules are intentionally API-specific. A parameter is only a hard constraint
# when the offline entity table has a field that can verify it.
SOURCE_HARD_FILTER_RULES: dict[str, dict[str, dict[str, Any]]] = {
    "shopping.check_inventory": {
        "product_ids": {"op": "one_of", "attributes": ["asin"]},
    },
    "shopping.search_products": {
        "max_price": {"op": "max", "attributes": ["price_number"]},
        "minimum_rating": {"op": "min", "attributes": ["rating"]},
        "minimum_review_count": {"op": "min", "attributes": ["review_count"]},
        "must_have_features": {"op": "contains_all", "record_text": True},
        "avoid_features": {"op": "contains_none", "record_text": True},
    },
    "food.search_restaurants": {
        "location": {"op": "text", "attributes": ["city"]},
        "budget_limit": {"op": "max", "attributes": ["average_cost"]},
        "rating_threshold": {"op": "min", "attributes": ["rating"]},
        "cuisine_preferences": {"op": "contains_any", "attributes": ["cuisines"]},
    },
    "maps.search_places": {
        "near": {"op": "text", "attributes": ["city", "address"]},
        "min_rating": {"op": "min", "attributes": ["rating", "review_rate_number"]},
        "avoid_features": {"op": "contains_none", "record_text": True},
    },
    "travel.search_flights": {
        "origin_airport": {"op": "text", "attributes": ["origin_city"]},
        "destination_airport": {"op": "text", "attributes": ["destination_city"]},
        "departure_date": {"op": "text", "attributes": ["flight_date"]},
        "max_price": {"op": "max", "attributes": ["price"]},
        "avoid_airlines": {"op": "contains_none", "record_text": True},
    },
    "travel.search_lodging": {
        "destination": {"op": "text", "attributes": ["city"]},
        "guests": {"op": "capacity", "attributes": ["maximum_occupancy"]},
        "max_nightly_rate": {"op": "max", "attributes": ["price"]},
        "amenities": {"op": "contains_all", "record_text": True},
    },
    "weather.get_forecast": {
        "location": {"op": "text", "attributes": ["city", "region", "country"]},
        "start_date": {"op": "text", "attributes": ["date"]},
    },
    "entertainment.search_events": {
        "location": {"op": "text", "attributes": ["city", "address"]},
        "max_price": {"op": "max", "attributes": ["price", "average_cost"]},
        "group_size": {"op": "capacity", "attributes": ["maximum_occupancy"]},
    },
}


def _is_empty(value: Any) -> bool:
    return value in (None, "", [], {})


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        if match:
            return float(match.group(0))
    return None


def _text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return flatten_text(value).casefold().strip()
    return str(value or "").casefold().strip()


def _values(value: Any) -> list[str]:
    items = value if isinstance(value, list) else [value]
    return [_text(item) for item in items if not _is_empty(item)]


def _text_matches(expected: Any, actual: Any) -> bool:
    expected_text = _text(expected)
    actual_text = _text(actual)
    if not expected_text or not actual_text:
        return False
    return expected_text == actual_text or expected_text in actual_text or actual_text in expected_text


def hard_constraints_compatible(
    requested_arguments: dict[str, Any],
    record_arguments: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Conservatively reject cached calls that cannot satisfy request bounds."""
    failures: list[str] = []
    for name, requested in requested_arguments.items():
        if _is_empty(requested):
            continue
        if name not in (
            UPPER_BOUND_ARGUMENTS
            | LOWER_BOUND_ARGUMENTS
            | EXACT_ARGUMENTS
            | REQUIRED_SET_ARGUMENTS
            | EXCLUDED_SET_ARGUMENTS
            | CAPACITY_ARGUMENTS
        ):
            continue
        if name not in record_arguments or _is_empty(record_arguments.get(name)):
            failures.append(f"{name}:missing")
            continue
        recorded = record_arguments[name]
        if name in UPPER_BOUND_ARGUMENTS:
            requested_number = _number(requested)
            recorded_number = _number(recorded)
            if requested_number is None or recorded_number is None or recorded_number > requested_number:
                failures.append(f"{name}:upper_bound")
        elif name in LOWER_BOUND_ARGUMENTS:
            requested_number = _number(requested)
            recorded_number = _number(recorded)
            if requested_number is None or recorded_number is None or recorded_number < requested_number:
                failures.append(f"{name}:lower_bound")
        elif name in EXACT_ARGUMENTS:
            if isinstance(requested, (dict, list)) or isinstance(recorded, (dict, list)):
                matches = canonical(requested) == canonical(recorded)
            else:
                matches = _text_matches(requested, recorded)
            if not matches:
                failures.append(f"{name}:mismatch")
        elif name in REQUIRED_SET_ARGUMENTS:
            requested_values = set(_values(requested))
            recorded_values = set(_values(recorded))
            if not requested_values.issubset(recorded_values):
                failures.append(f"{name}:required_values")
        elif name in EXCLUDED_SET_ARGUMENTS:
            requested_values = set(_values(requested))
            recorded_values = set(_values(recorded))
            if not requested_values.issubset(recorded_values):
                failures.append(f"{name}:excluded_values")
        elif name in CAPACITY_ARGUMENTS:
            requested_number = _number(requested)
            recorded_number = _number(recorded)
            if requested_number is None or recorded_number is None or recorded_number < requested_number:
                failures.append(f"{name}:capacity")
    return not failures, failures


def _record_attribute(record: dict[str, Any], names: list[str]) -> Any:
    attributes = record.get("attributes") if isinstance(record.get("attributes"), dict) else {}
    for name in names:
        value = attributes.get(name)
        if not _is_empty(value):
            return value
        value = record.get(name)
        if not _is_empty(value):
            return value
    return None


def _source_record_matches(
    record: dict[str, Any],
    value: Any,
    rule: dict[str, Any],
) -> bool:
    actual = (
        flatten_text(record, max_chars=6000)
        if rule.get("record_text")
        else _record_attribute(record, list(rule.get("attributes") or []))
    )
    op = rule.get("op")
    if op == "max":
        expected_number, actual_number = _number(value), _number(actual)
        return expected_number is not None and actual_number is not None and actual_number <= expected_number
    if op == "min":
        expected_number, actual_number = _number(value), _number(actual)
        return expected_number is not None and actual_number is not None and actual_number >= expected_number
    if op == "capacity":
        expected_number, actual_number = _number(value), _number(actual)
        return expected_number is not None and actual_number is not None and actual_number >= expected_number
    if op == "text":
        return _text_matches(value, actual)
    if op == "one_of":
        return _text(actual) in set(_values(value))
    actual_text = _text(actual)
    expected_values = _values(value)
    if op == "contains_all":
        return all(item in actual_text for item in expected_values)
    if op == "contains_any":
        return not expected_values or any(item in actual_text for item in expected_values)
    if op == "contains_none":
        return all(item not in actual_text for item in expected_values)
    return True


def hard_filter_source_records(
    api_name: str,
    arguments: dict[str, Any],
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rules = SOURCE_HARD_FILTER_RULES.get(api_name, {})
    active_rules = {
        name: rule
        for name, rule in rules.items()
        if name in arguments and not _is_empty(arguments[name])
    }
    rejected: Counter[str] = Counter()
    filtered: list[dict[str, Any]] = []
    for record in records:
        failed_names = [
            name
            for name, rule in active_rules.items()
            if not _source_record_matches(record, arguments[name], rule)
        ]
        if failed_names:
            rejected.update(failed_names)
        else:
            filtered.append(record)
    return filtered, {
        "strategy": "structured_hard_constraints_before_multilingual_e5",
        "input_record_count": len(records),
        "output_record_count": len(filtered),
        "applied_constraints": sorted(active_rules),
        "rejected_by_constraint": dict(sorted(rejected.items())),
    }


def _candidate_as_source_record(candidate: Any, index: int) -> dict[str, Any]:
    if not isinstance(candidate, dict):
        return {
            "title": str(candidate),
            "text": str(candidate),
            "attributes": {},
            "_candidate_index": index,
        }
    attributes = copy.deepcopy(candidate.get("attributes")) if isinstance(candidate.get("attributes"), dict) else {}
    aliases = {
        "price_number": ("price_number", "price", "estimated_total", "total_price", "cost"),
        "average_cost": ("average_cost", "estimated_total", "price", "cost"),
        "rating": ("rating", "aggregate_rating", "review_score"),
        "review_count": ("review_count", "reviews"),
        "city": ("city", "location", "destination"),
        "address": ("address", "location"),
        "maximum_occupancy": ("maximum_occupancy", "capacity", "max_guests"),
        "origin_city": ("origin_city", "origin", "origin_airport"),
        "destination_city": ("destination_city", "destination", "destination_airport"),
        "flight_date": ("flight_date", "departure_date", "date"),
        "date": ("date", "start_date"),
    }
    for target, sources in aliases.items():
        if not _is_empty(attributes.get(target)):
            continue
        for source in sources:
            value = candidate.get(source)
            if not _is_empty(value):
                attributes[target] = value
                break
    return {
        "title": candidate.get("title") or candidate.get("name") or candidate.get("id"),
        "text": flatten_text(candidate, max_chars=6000),
        "attributes": attributes,
        "_candidate_index": index,
    }


def response_shape(value: Any) -> str:
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    if value is None:
        return "null"
    return type(value).__name__


class MultilingualE5Encoder:
    """Process-local multilingual E5 model with a bounded query cache."""

    def __init__(self, model_dir: Path = E5_MODEL_DIR) -> None:
        required_file = model_dir / "model.safetensors"
        if not required_file.exists():
            raise RuntimeError(
                f"离线检索模型不存在: {required_file}。"
                f"请先下载 {E5_MODEL_ID} 到 {model_dir}。"
            )
        from sentence_transformers import SentenceTransformer

        self.model_dir = model_dir
        self.model_id = E5_MODEL_ID
        requested_device = os.environ.get("PPEVAL_SANDBOX_E5_DEVICE", "").strip() or None
        self.model = SentenceTransformer(
            str(model_dir),
            device=requested_device,
            local_files_only=True,
        )
        self.model.max_seq_length = E5_MAX_SEQUENCE_LENGTH
        self.device = str(self.model.device)
        self.dimension = int(self.model.get_embedding_dimension())
        self.document_batch_size = (
            E5_GPU_DOCUMENT_BATCH_SIZE
            if self.device.startswith("cuda")
            else E5_CPU_DOCUMENT_BATCH_SIZE
        )
        self._model_lock = threading.Lock()
        self._query_cache_lock = threading.Lock()
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        passages = [f"passage: {text}" for text in texts]
        with self._model_lock:
            embeddings = self.model.encode(
                passages,
                batch_size=self.document_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        cache_key = stable_hash(text)
        with self._query_cache_lock:
            cached = self._query_cache.get(cache_key)
            if cached is not None:
                self._query_cache.move_to_end(cache_key)
                return cached
        with self._model_lock:
            embedding = self.model.encode(
                [f"query: {text}"],
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )[0]
        vector = np.asarray(embedding, dtype=np.float32)
        with self._query_cache_lock:
            self._query_cache[cache_key] = vector
            self._query_cache.move_to_end(cache_key)
            while len(self._query_cache) > E5_QUERY_CACHE_SIZE:
                self._query_cache.popitem(last=False)
        return vector


class ApiRecordIndex:
    """Hard-filtered multilingual E5 index over one API's records."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        api_def: dict[str, Any] | None,
        encoder: MultilingualE5Encoder,
        embeddings: np.ndarray,
    ) -> None:
        self.records = records
        self.api_def = api_def or {}
        self.api_name = str(self.api_def.get("api_name") or (records[0].get("api_name") if records else ""))
        self.encoder = encoder
        self.documents = [self._record_text(record) for record in records]
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        if self.embeddings.shape != (len(records), encoder.dimension):
            raise ValueError(
                f"API record embedding shape mismatch: api={self.api_name} "
                f"records={len(records)} embeddings={self.embeddings.shape}"
            )

    @staticmethod
    def _record_text(record: dict[str, Any]) -> str:
        function_call = record.get("function_call") if isinstance(record.get("function_call"), dict) else {}
        arguments = function_call.get("arguments") if isinstance(function_call.get("arguments"), dict) else {}
        response = record.get("mock_response") if isinstance(record.get("mock_response"), dict) else {}
        body = response.get("response_body") if isinstance(response.get("response_body"), dict) else {}
        compact_body = {
            key: body.get(key)
            for key in ("summary", "task_brief", "ranking_basis", "preference_alignment", "items")
            if key in body
        }
        return flatten_text(
            {
                "api_name": record.get("api_name"),
                "task_title": record.get("task_title"),
                "daily_life_domain": record.get("daily_life_domain"),
                "arguments": arguments,
                "response": compact_body,
            }
        )

    def best(
        self,
        arguments: dict[str, Any],
        *,
        user_id: str,
        task_id: str,
    ) -> dict[str, Any] | None:
        if not self.records:
            return None
        argument_hash = stable_hash(arguments)
        rejected_by_constraint: Counter[str] = Counter()
        compatible_indices: list[int] = []
        hard_constraint_names = (
            UPPER_BOUND_ARGUMENTS
            | LOWER_BOUND_ARGUMENTS
            | EXACT_ARGUMENTS
            | REQUIRED_SET_ARGUMENTS
            | EXCLUDED_SET_ARGUMENTS
            | CAPACITY_ARGUMENTS
        )
        applied_constraints = sorted(
            name
            for name, value in arguments.items()
            if name in hard_constraint_names and not _is_empty(value)
        )
        for idx, record in enumerate(self.records):
            record_arguments = (
                record.get("function_call", {}).get("arguments", {})
                if isinstance(record.get("function_call"), dict)
                else {}
            )
            compatible, failures = hard_constraints_compatible(arguments, record_arguments)
            if not compatible:
                rejected_by_constraint.update(failures)
                continue
            compatible_indices.append(idx)
        if not compatible_indices:
            return None
        query_text = semantic_query_text(arguments)
        query_embedding = self.encoder.encode_query(query_text)
        candidate_embeddings = self.embeddings[np.asarray(compatible_indices, dtype=np.int64)]
        scores = candidate_embeddings @ query_embedding
        best_local_index = int(np.argmax(scores))
        best_score = float(scores[best_local_index])
        if best_score < E5_MIN_COSINE_SIMILARITY:
            return None
        best_record = self.records[compatible_indices[best_local_index]]
        record_arguments = (
            best_record.get("function_call", {}).get("arguments", {})
            if isinstance(best_record.get("function_call"), dict)
            else {}
        )
        best_meta = {
            "strategy": "hard_filter_then_multilingual_e5",
            "model": self.encoder.model_id,
            "multilingual_e5_score": round(best_score, 6),
            "minimum_similarity": E5_MIN_COSINE_SIMILARITY,
            "exact_argument_match": stable_hash(record_arguments) == argument_hash,
            "selected_record_id": best_record.get("record_id"),
            "selected_user_id": best_record.get("user_id"),
            "selected_task_id": best_record.get("task_id"),
            "selected_call_id": best_record.get("call_id"),
            "selected_cache_keys": best_record.get("cache_keys", {}),
            "hard_constraint_filter": {
                "strategy": "structured_hard_constraints_before_multilingual_e5",
                "input_record_count": len(self.records),
                "output_record_count": len(compatible_indices),
                "applied_constraints": applied_constraints,
                "rejected_by_constraint": dict(sorted(rejected_by_constraint.items())),
            },
        }
        selected = copy.deepcopy(best_record)
        selected["_sandbox_retrieval"] = best_meta
        return selected


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def corpus_metadata(
    records: list[dict[str, Any]],
    documents: list[str],
    *,
    id_field: str,
) -> dict[str, Any]:
    if len(records) != len(documents):
        raise ValueError(
            f"语料记录与文档数量不一致: records={len(records)} documents={len(documents)}"
        )
    return {
        "record_count": len(records),
        "record_ids_hash": stable_sequence_hash(record.get(id_field) for record in records),
        "documents_hash": stable_sequence_hash(documents),
    }


def validate_corpus_metadata(
    *,
    label: str,
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    mismatches = {
        field: {"expected": expected.get(field), "actual": actual.get(field)}
        for field in ("record_count", "record_ids_hash", "documents_hash")
        if actual.get(field) != expected.get(field)
    }
    if mismatches:
        raise RuntimeError(
            f"Sandbox E5 索引与当前语料不一致: {label} {canonical(mismatches)}。"
            f"请重新运行 `{VECTOR_INDEX_BUILD_COMMAND}`。"
        )


def load_api_index_corpus(
    database_dir: Path = DATABASE_DIR,
) -> tuple[
    dict[str, dict[str, list[dict[str, Any]]]],
    list[tuple[str, list[dict[str, Any]], list[str]]],
]:
    """Load API records in the single canonical order used to build vectors."""
    theme_records_by_api: dict[str, dict[str, list[dict[str, Any]]]] = {}
    api_records_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    api_record_dir = database_dir / "api_records"
    for path in sorted(api_record_dir.glob("*.jsonl")):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in load_jsonl_records(path):
            api_name = record.get("api_name")
            if isinstance(api_name, str):
                grouped[api_name].append(record)
                api_records_by_name[api_name].append(record)
        theme_records_by_api[path.stem] = grouped

    entries: list[tuple[str, list[dict[str, Any]], list[str]]] = []
    for api_name in sorted(api_records_by_name):
        records = api_records_by_name[api_name]
        documents = [ApiRecordIndex._record_text(record) for record in records]
        entries.append((api_name, records, documents))
    return theme_records_by_api, entries


class SourceInfoIndex:
    """Hard-filtered multilingual E5 index over source-info documents."""

    def __init__(
        self,
        source_info_dir: Path = SOURCE_INFO_DIR,
        vector_index_dir: Path = VECTOR_INDEX_DIR,
    ) -> None:
        self.source_info_dir = source_info_dir
        self.vector_index_dir = vector_index_dir
        self.manifest_path = source_info_dir / "manifest.json"
        self.available = self.manifest_path.exists()
        self._records_by_domain: dict[str, list[dict[str, Any]]] = {}
        self._embeddings_by_domain: dict[str, np.ndarray] = {}
        self._positions_by_domain: dict[str, dict[str, int]] = {}
        self.encoder: MultilingualE5Encoder | None = None
        self.prepared = False

    def prepare(
        self,
        encoder: MultilingualE5Encoder,
        vector_manifest: dict[str, Any],
        *,
        logger: Any | None = None,
    ) -> int:
        self.encoder = encoder
        self._records_by_domain = {}
        self._embeddings_by_domain = {}
        self._positions_by_domain = {}
        records_dir = self.source_info_dir / "records"
        source_manifest = vector_manifest.get("source_info")
        if not isinstance(source_manifest, dict):
            raise RuntimeError("Sandbox E5 manifest 缺少 source_info 配置。")
        domain_manifest = source_manifest.get("domains")
        if not isinstance(domain_manifest, dict):
            raise RuntimeError("Sandbox E5 manifest 缺少 source_info.domains 配置。")
        actual_domains = [path.stem for path in sorted(records_dir.glob("*.jsonl"))]
        expected_domains = sorted(str(domain) for domain in domain_manifest)
        if actual_domains != expected_domains:
            raise RuntimeError(
                "Sandbox E5 source_info domain 集合或顺序已变化: "
                f"expected={expected_domains} actual={actual_domains}。"
                f"请重新运行 `{VECTOR_INDEX_BUILD_COMMAND}`。"
            )
        total = 0
        for domain in actual_domains:
            records = self._load_domain(domain)
            documents = [self._record_text(record) for record in records]
            expected = domain_manifest[domain]
            if not isinstance(expected, dict):
                raise RuntimeError(f"Sandbox E5 manifest domain 配置无效: {domain}")
            validate_corpus_metadata(
                label=f"source_info/{domain}",
                actual=corpus_metadata(records, documents, id_field="doc_id"),
                expected=expected,
            )
            vector_file = expected.get("vector_file")
            if not isinstance(vector_file, str) or not vector_file:
                raise RuntimeError(f"Sandbox E5 manifest 缺少向量文件: source_info/{domain}")
            vector_path = self.vector_index_dir / vector_file
            if not vector_path.exists():
                raise RuntimeError(
                    f"Sandbox E5 向量文件不存在: {vector_path}。"
                    f"请运行 `{VECTOR_INDEX_BUILD_COMMAND}`。"
                )
            embeddings = np.load(vector_path, mmap_mode="r", allow_pickle=False)
            expected_shape = (len(records), encoder.dimension)
            declared_shape = expected.get("shape")
            declared_dtype = expected.get("dtype")
            if (
                declared_shape != list(expected_shape)
                or declared_dtype != "float32"
                or embeddings.shape != expected_shape
                or embeddings.dtype != np.float32
            ):
                raise RuntimeError(
                    f"Sandbox E5 向量格式不匹配: {vector_path} "
                    f"declared_shape={declared_shape} declared_dtype={declared_dtype} "
                    f"expected_shape={expected_shape} actual_shape={embeddings.shape} "
                    f"expected_dtype=float32 actual_dtype={embeddings.dtype}。"
                    f"请重新运行 `{VECTOR_INDEX_BUILD_COMMAND}`。"
                )
            if logger is not None:
                logger.info(
                    "加载预编码 E5 source_info 向量：domain=%s records=%s file=%s",
                    domain,
                    len(records),
                    vector_path,
                )
            self._records_by_domain[domain] = records
            self._embeddings_by_domain[domain] = embeddings
            self._positions_by_domain[domain] = {
                str(record.get("doc_id")): index
                for index, record in enumerate(records)
            }
            total += len(records)
        if source_manifest.get("record_count") != total:
            raise RuntimeError(
                "Sandbox E5 source_info 总记录数不一致: "
                f"expected={source_manifest.get('record_count')} actual={total}。"
                f"请重新运行 `{VECTOR_INDEX_BUILD_COMMAND}`。"
            )
        self.prepared = True
        return total

    def search(
        self,
        *,
        theme: str,
        api_name: str,
        arguments: dict[str, Any],
        user_context: dict[str, Any] | None = None,
        limit: int = 5,
    ) -> dict[str, Any] | None:
        if not self.available:
            return None
        if not self.prepared or self.encoder is None:
            raise RuntimeError("source_info E5 向量尚未准备；评测开始前必须调用 prepare_vector_indexes()。")
        domains = self._domains_for_theme(theme)
        records = self._records_for_domains(domains)
        if not records:
            return None
        records, hard_filter_audit = hard_filter_source_records(api_name, arguments, records)
        if not records:
            return {
                "strategy": "hard_filter_then_multilingual_e5",
                "source_info_dir": str(self.source_info_dir.relative_to(PROJECT_ROOT)),
                "domains": domains,
                "result_status": "no_results",
                "model": self.encoder.model_id,
                "minimum_similarity": E5_MIN_COSINE_SIMILARITY,
                "hit_count": 0,
                "hard_constraint_filter": hard_filter_audit,
                "hits": [],
            }
        query_text = semantic_query_text(arguments)
        if not query_text.strip():
            return None
        query_embedding = self.encoder.encode_query(query_text)
        scored = self._score_records(records, query_embedding)
        hits = []
        for score, record in scored[:limit]:
            hit = copy.deepcopy(record)
            hit["multilingual_e5_score"] = round(score, 6)
            hits.append(hit)
        if not hits:
            return {
                "strategy": "hard_filter_then_multilingual_e5",
                "source_info_dir": str(self.source_info_dir.relative_to(PROJECT_ROOT)),
                "domains": domains,
                "result_status": "no_results",
                "model": self.encoder.model_id,
                "minimum_similarity": E5_MIN_COSINE_SIMILARITY,
                "hit_count": 0,
                "hard_constraint_filter": hard_filter_audit,
                "hits": [],
            }
        return {
            "strategy": "hard_filter_then_multilingual_e5",
            "source_info_dir": str(self.source_info_dir.relative_to(PROJECT_ROOT)),
            "domains": domains,
            "result_status": "matched",
            "model": self.encoder.model_id,
            "minimum_similarity": E5_MIN_COSINE_SIMILARITY,
            "hit_count": len(hits),
            "hard_constraint_filter": hard_filter_audit,
            "hits": hits,
        }

    def _domains_for_theme(self, theme: str) -> list[str]:
        if theme in THEME_TO_SOURCE_DOMAINS:
            return list(THEME_TO_SOURCE_DOMAINS[theme])
        return []

    def _records_for_domains(self, domains: list[str]) -> list[dict[str, Any]]:
        all_records: list[dict[str, Any]] = []
        for domain in domains:
            records = self._records_by_domain.get(domain)
            if records is None:
                records = self._load_domain(domain)
                self._records_by_domain[domain] = records
            all_records.extend(records)
        return all_records

    def _load_domain(self, domain: str) -> list[dict[str, Any]]:
        path = self.source_info_dir / "records" / f"{domain}.jsonl"
        return load_jsonl_records(path)

    def _score_records(
        self,
        records: list[dict[str, Any]],
        query_embedding: np.ndarray,
    ) -> list[tuple[float, dict[str, Any]]]:
        scored: list[tuple[float, dict[str, Any]]] = []
        records_by_domain: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for record in records:
            domain = str(record.get("domain") or "")
            position = self._positions_by_domain.get(domain, {}).get(str(record.get("doc_id")))
            if position is not None:
                records_by_domain[domain].append((position, record))
        for domain, positioned_records in records_by_domain.items():
            positions = np.asarray([item[0] for item in positioned_records], dtype=np.int64)
            embeddings = self._embeddings_by_domain[domain][positions]
            scores = embeddings @ query_embedding
            for score, (_, record) in zip(scores.tolist(), positioned_records, strict=True):
                if float(score) >= E5_MIN_COSINE_SIMILARITY:
                    scored.append((float(score), record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

    @staticmethod
    def _record_text(record: dict[str, Any]) -> str:
        return flatten_text(
            {
                "domain": record.get("domain"),
                "title": record.get("title"),
                "text": record.get("text"),
                "attributes": record.get("attributes"),
            },
            max_chars=4000,
        )

class GlobalSandboxDatabase:
    """ETAPP-style local database and StableToolBench-style deterministic cache lookup."""

    def __init__(
        self,
        api_definitions: dict[str, dict[str, Any]],
        database_dir: Path = DATABASE_DIR,
        vector_index_dir: Path = VECTOR_INDEX_DIR,
    ) -> None:
        self.api_definitions = api_definitions
        self.database_dir = database_dir
        self.vector_index_dir = vector_index_dir
        self.vector_manifest_path = vector_index_dir / "manifest.json"
        self.manifest_path = database_dir / "manifest.json"
        self.available = self.manifest_path.exists()
        self._theme_records_by_api: dict[str, dict[str, list[dict[str, Any]]]] = {}
        self._api_indexes: dict[str, ApiRecordIndex] = {}
        self.source_info = SourceInfoIndex(vector_index_dir=vector_index_dir)
        self.encoder: MultilingualE5Encoder | None = None
        self.vector_indexes_prepared = False
        self.vector_index_stats: dict[str, Any] = {}
        self._prepare_lock = threading.Lock()

    def prepare_vector_indexes(self, *, logger: Any | None = None) -> dict[str, Any]:
        with self._prepare_lock:
            if self.vector_indexes_prepared:
                return copy.deepcopy(self.vector_index_stats)
            started = time.perf_counter()
            if not self.vector_manifest_path.exists():
                raise RuntimeError(
                    f"Sandbox E5 预编码索引不存在: {self.vector_manifest_path}。"
                    f"请先运行 `{VECTOR_INDEX_BUILD_COMMAND}`；正式评测不会自动编码语料。"
                )
            with self.vector_manifest_path.open("r", encoding="utf-8") as f:
                vector_manifest = json.load(f)
            if vector_manifest.get("schema_version") != VECTOR_INDEX_SCHEMA_VERSION:
                raise RuntimeError(
                    "Sandbox E5 索引版本不兼容: "
                    f"expected={VECTOR_INDEX_SCHEMA_VERSION} "
                    f"actual={vector_manifest.get('schema_version')}。"
                    f"请重新运行 `{VECTOR_INDEX_BUILD_COMMAND}`。"
                )
            if logger is not None:
                logger.info("开始加载 Sandbox multilingual-e5-small 预编码向量；评测期不编码文档")
            encoder = MultilingualE5Encoder()
            model_manifest = vector_manifest.get("model")
            if not isinstance(model_manifest, dict):
                raise RuntimeError("Sandbox E5 manifest 缺少 model 配置。")
            expected_model = {
                "model_id": E5_MODEL_ID,
                "embedding_dimension": encoder.dimension,
                "max_sequence_length": E5_MAX_SEQUENCE_LENGTH,
                "document_prefix": "passage: ",
                "query_prefix": "query: ",
            }
            model_mismatches = {
                field: {"expected": value, "actual": model_manifest.get(field)}
                for field, value in expected_model.items()
                if model_manifest.get(field) != value
            }
            if model_mismatches:
                raise RuntimeError(
                    f"Sandbox E5 编码模型配置不匹配: {canonical(model_mismatches)}。"
                    f"请重新运行 `{VECTOR_INDEX_BUILD_COMMAND}`。"
                )

            source_record_count = self.source_info.prepare(
                encoder,
                vector_manifest,
                logger=logger,
            )

            self._theme_records_by_api, api_entries = load_api_index_corpus(self.database_dir)
            api_manifest = vector_manifest.get("api_records")
            if not isinstance(api_manifest, dict):
                raise RuntimeError("Sandbox E5 manifest 缺少 api_records 配置。")
            expected_entries = api_manifest.get("entries")
            if not isinstance(expected_entries, list):
                raise RuntimeError("Sandbox E5 manifest 缺少 api_records.entries 配置。")
            actual_names = [entry[0] for entry in api_entries]
            expected_names = [entry.get("api_name") for entry in expected_entries if isinstance(entry, dict)]
            if len(expected_names) != len(expected_entries) or actual_names != expected_names:
                raise RuntimeError(
                    "Sandbox E5 API 集合或顺序已变化: "
                    f"expected={expected_names} actual={actual_names}。"
                    f"请重新运行 `{VECTOR_INDEX_BUILD_COMMAND}`。"
                )

            all_api_records: list[dict[str, Any]] = []
            all_api_documents: list[str] = []
            cursor = 0
            for (api_name, records, documents), expected in zip(
                api_entries,
                expected_entries,
                strict=True,
            ):
                validate_corpus_metadata(
                    label=f"api_records/{api_name}",
                    actual=corpus_metadata(records, documents, id_field="record_id"),
                    expected=expected,
                )
                expected_start = expected.get("start")
                expected_stop = expected.get("stop")
                if expected_start != cursor or expected_stop != cursor + len(records):
                    raise RuntimeError(
                        f"Sandbox E5 API 向量切片不匹配: api={api_name} "
                        f"expected=({expected_start}, {expected_stop}) "
                        f"actual=({cursor}, {cursor + len(records)})。"
                        f"请重新运行 `{VECTOR_INDEX_BUILD_COMMAND}`。"
                    )
                all_api_records.extend(records)
                all_api_documents.extend(documents)
                cursor += len(records)
            validate_corpus_metadata(
                label="api_records/all",
                actual=corpus_metadata(
                    all_api_records,
                    all_api_documents,
                    id_field="record_id",
                ),
                expected=api_manifest,
            )

            vector_file = api_manifest.get("vector_file")
            if not isinstance(vector_file, str) or not vector_file:
                raise RuntimeError("Sandbox E5 manifest 缺少 api_records.vector_file 配置。")
            api_vector_path = self.vector_index_dir / vector_file
            if not api_vector_path.exists():
                raise RuntimeError(
                    f"Sandbox E5 向量文件不存在: {api_vector_path}。"
                    f"请运行 `{VECTOR_INDEX_BUILD_COMMAND}`。"
                )
            all_api_embeddings = np.load(api_vector_path, mmap_mode="r", allow_pickle=False)
            expected_shape = (len(all_api_documents), encoder.dimension)
            declared_shape = api_manifest.get("shape")
            declared_dtype = api_manifest.get("dtype")
            if (
                declared_shape != list(expected_shape)
                or declared_dtype != "float32"
                or all_api_embeddings.shape != expected_shape
                or all_api_embeddings.dtype != np.float32
            ):
                raise RuntimeError(
                    f"Sandbox E5 向量格式不匹配: {api_vector_path} "
                    f"declared_shape={declared_shape} declared_dtype={declared_dtype} "
                    f"expected_shape={expected_shape} actual_shape={all_api_embeddings.shape} "
                    f"expected_dtype=float32 actual_dtype={all_api_embeddings.dtype}。"
                    f"请重新运行 `{VECTOR_INDEX_BUILD_COMMAND}`。"
                )
            self._api_indexes = {}
            cursor = 0
            for api_name, records, documents in api_entries:
                next_cursor = cursor + len(documents)
                self._api_indexes[api_name] = ApiRecordIndex(
                    records,
                    self.api_definitions.get(api_name),
                    encoder,
                    all_api_embeddings[cursor:next_cursor],
                )
                cursor = next_cursor

            elapsed = time.perf_counter() - started
            self.encoder = encoder
            self.vector_indexes_prepared = True
            self.vector_index_stats = {
                "strategy": "load_preencoded_vectors_then_hard_filter_and_multilingual_e5",
                "model": encoder.model_id,
                "model_dir": str(encoder.model_dir.relative_to(PROJECT_ROOT)),
                "vector_index_dir": str(self.vector_index_dir.relative_to(PROJECT_ROOT)),
                "device": encoder.device,
                "embedding_dimension": encoder.dimension,
                "minimum_similarity": E5_MIN_COSINE_SIMILARITY,
                "api_record_count": len(all_api_documents),
                "source_info_record_count": source_record_count,
                "elapsed_seconds": round(elapsed, 3),
                "vectors_persisted_between_evals": True,
                "documents_encoded_during_eval_startup": 0,
            }
            if logger is not None:
                logger.info(
                    "Sandbox E5 预编码向量加载完成：api_records=%s source_info=%s device=%s elapsed=%.2fs",
                    len(all_api_documents),
                    source_record_count,
                    encoder.device,
                    elapsed,
                )
            return copy.deepcopy(self.vector_index_stats)

    def _ensure_vector_indexes(self) -> None:
        if not self.vector_indexes_prepared:
            self.prepare_vector_indexes()

    def select_response(
        self,
        api_name: str,
        arguments: dict[str, Any],
        *,
        user_id: str,
        task_id: str,
    ) -> dict[str, Any] | None:
        if not self.available:
            return None
        self._ensure_vector_indexes()
        records = self._records_for_api(api_name)
        if not records:
            return None
        index = self._api_indexes.get(api_name)
        if index is None:
            return None
        return index.best(arguments, user_id=user_id, task_id=task_id)

    def condition_response_body(
        self,
        response_body: Any,
        arguments: dict[str, Any],
        *,
        api_name: str | None = None,
    ) -> Any:
        if not isinstance(response_body, dict):
            return response_body
        body = copy.deepcopy(response_body)
        argument_hash = stable_hash(arguments)
        body["selected_inputs"] = copy.deepcopy(arguments)
        body["argument_conditioning"] = {
            "enabled": True,
            "argument_hash": argument_hash,
            "retrieval_model": E5_MODEL_ID,
        }
        filter_audits: dict[str, Any] = {}
        applied_filter = False
        filtered_candidate_count = 0
        for key, value in list(body.items()):
            if key not in CANDIDATE_LIST_KEYS or not isinstance(value, list):
                continue
            if api_name:
                candidate_records = [_candidate_as_source_record(item, index) for index, item in enumerate(value)]
                filtered_records, audit = hard_filter_source_records(api_name, arguments, candidate_records)
                filter_audits[key] = audit
                if audit["applied_constraints"]:
                    applied_filter = True
                    allowed_indexes = {int(record["_candidate_index"]) for record in filtered_records}
                    value = [item for index, item in enumerate(value) if index in allowed_indexes]
                    body[key] = value
            filtered_candidate_count += len(value)
            body[key] = value
        if filter_audits:
            body["hard_constraint_filter"] = filter_audits
        if applied_filter:
            body["result_status"] = "matched" if filtered_candidate_count else "no_results"
            if not filtered_candidate_count:
                missing = body.get("missing_information")
                if not isinstance(missing, list):
                    missing = []
                missing.append("没有候选同时满足 tool_call.arguments 中可验证的硬约束。")
                body["missing_information"] = missing
        return body

    def retrieve_source_info(
        self,
        *,
        api_name: str,
        arguments: dict[str, Any],
        user_context: dict[str, Any] | None = None,
        limit: int = 5,
    ) -> dict[str, Any] | None:
        self._ensure_vector_indexes()
        theme = self._theme_for_api(api_name)
        return self.source_info.search(
            theme=theme,
            api_name=api_name,
            arguments=arguments,
            user_context=user_context,
            limit=limit,
        )

    def _records_for_api(self, api_name: str) -> list[dict[str, Any]]:
        theme = self._theme_for_api(api_name)
        theme_records = self._theme_records_by_api.get(theme)
        if theme_records is None:
            theme_records = self._load_theme_records(theme)
            self._theme_records_by_api[theme] = theme_records
        return theme_records.get(api_name, [])

    def _load_theme_records(self, theme: str) -> dict[str, list[dict[str, Any]]]:
        path = self.database_dir / "api_records" / f"{theme}.jsonl"
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in load_jsonl_records(path):
            api_name = record.get("api_name")
            if isinstance(api_name, str):
                grouped[api_name].append(record)
        return grouped

    def _theme_for_api(self, api_name: str) -> str:
        api_def = self.api_definitions.get(api_name, {})
        category = api_def.get("category") or api_name.split(".", 1)[0]
        return CATEGORY_TO_THEME.get(str(category), str(category))


def materialize_sandbox_response(
    *,
    template: dict[str, Any],
    api_name: str,
    arguments: dict[str, Any],
    task_id: str,
    action: Any,
    transport: dict[str, Any],
    validation: dict[str, Any],
    conditioning_database: GlobalSandboxDatabase,
    retrieval_metadata: dict[str, Any] | None = None,
    user_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = copy.deepcopy(template)
    response["sandbox_status"] = "success"
    response["api_name"] = api_name
    response["action"] = action
    response["task_context"] = task_id
    response["transport"] = transport
    response["request_echo"] = {
        "function": api_name,
        "arguments": copy.deepcopy(arguments),
        "idempotency_key": transport["request_id"].replace("req_", "idm_", 1),
    }
    response["validation"] = validation
    response.setdefault("warnings", [])
    response["warnings"].extend(validation.get("warnings", []))
    response.setdefault("errors", [])
    response.setdefault("feedback", {})
    response["feedback"].setdefault("planner_relevance", "usable_for_next_step")
    response["feedback"].setdefault("requires_user_confirmation", False)
    if "response_body" in response:
        response["response_body"] = conditioning_database.condition_response_body(
            response["response_body"],
            arguments,
            api_name=api_name,
        )
        source_info = conditioning_database.retrieve_source_info(
            api_name=api_name,
            arguments=arguments,
            user_context=user_context,
        )
        if source_info:
            response["source_info_retrieval"] = {
                key: value for key, value in source_info.items() if key != "hits"
            }
            if isinstance(response["response_body"], dict):
                response["response_body"]["source_info"] = {
                    "description": (
                        "multilingual-e5-small 检索的真实世界离线源信息；"
                        "这些结果用于约束 Sandbox 候选，不包含用户私有数据。"
                    ),
                    "hits": source_info["hits"],
                }
                source_filter = source_info.get("hard_constraint_filter") or {}
                if source_info.get("result_status") == "matched":
                    # Only entities that passed the action's hard constraints may
                    # remain as candidates. Old synthetic mock candidates are not
                    # mixed with verified source-info entities.
                    for key, value in list(response["response_body"].items()):
                        if key in CANDIDATE_LIST_KEYS and isinstance(value, list):
                            response["response_body"][key] = []
                    result_key = SOURCE_RESULT_KEYS.get(api_name, "items")
                    response["response_body"][result_key] = copy.deepcopy(source_info["hits"])
                    body_filter = response["response_body"].get("hard_constraint_filter")
                    if isinstance(body_filter, dict):
                        body_filter[result_key] = copy.deepcopy(source_filter)
                    response["response_body"]["result_status"] = "matched"
                    missing = response["response_body"].get("missing_information")
                    if isinstance(missing, list):
                        response["response_body"]["missing_information"] = [
                            item
                            for item in missing
                            if item
                            not in {
                                "没有候选同时满足 tool_call.arguments 中可验证的硬约束。",
                                "离线实体库在硬过滤和 multilingual-e5-small 检索后没有匹配结果。",
                            }
                        ]
                elif source_info.get("result_status") == "no_results":
                    for key, value in list(response["response_body"].items()):
                        if key in CANDIDATE_LIST_KEYS and isinstance(value, list):
                            response["response_body"][key] = []
                    response["response_body"]["result_status"] = "no_results"
                    missing = response["response_body"].get("missing_information")
                    if not isinstance(missing, list):
                        missing = []
                    missing.append("离线实体库在硬过滤和 multilingual-e5-small 检索后没有匹配结果。")
                    response["response_body"]["missing_information"] = missing
    if retrieval_metadata:
        response["sandbox_retrieval"] = retrieval_metadata
        effects_key = "preference_effects" if "preference_effects" in response else "context_effects"
        response.setdefault(effects_key, [])
        retrieval_strategy = retrieval_metadata.get("strategy")
        if retrieval_strategy == "workflow_call_example_fallback":
            retrieval_effect = "全局缓存无兼容记录后，使用通过参数硬约束复核的 workflow call example。"
        elif retrieval_strategy == "generic_fallback_after_no_compatible_cache_record":
            retrieval_effect = "全局缓存和 workflow example 均无兼容记录，使用当前参数生成回退结构并继续执行实体硬过滤与 multilingual-e5-small 检索。"
        else:
            retrieval_effect = "先依据当前 tool_call.arguments 硬过滤全局 SandboxDB/database，再以 multilingual-e5-small 余弦相似度排序。"
        response[effects_key].append(retrieval_effect)
    if response.get("source_info_retrieval"):
        effects_key = "preference_effects" if "preference_effects" in response else "context_effects"
        response.setdefault(effects_key, [])
        response[effects_key].append(
            "response_body.source_info 来自 SandboxDB/source_info；候选先执行结构化硬过滤，再按 tool_call.arguments 的 multilingual-e5-small 相似度排序。"
        )
    if isinstance(response.get("response_body"), dict):
        if response["response_body"].get("result_status") == "no_results":
            response["feedback"]["planner_relevance"] = "insufficient"
            response["feedback"]["missing_information"] = copy.deepcopy(
                response["response_body"].get("missing_information", [])
            )
        response["raw_response_shape"] = {
            key: response_shape(value)
            for key, value in response["response_body"].items()
        }
    return response
