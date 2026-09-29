#!/usr/bin/env python3
"""Run the Agent method with an abstract profile and two task-time RAGs."""

from __future__ import annotations

import bz2
import copy
import hashlib
import html
import json
import logging
import re
import sqlite3
import sys
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EVAL_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EVAL_DIR.parent
PAG_DIR = EVAL_DIR / "pag"
RAG_DIR = EVAL_DIR / "rag"
for module_dir in (EVAL_DIR, PAG_DIR, RAG_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

import eval_core
import eval_pag


# =========================
# 用户长期可调 Agent 超参数
# =========================

# 除离线 Wikipedia RAG-B 外，全部评测配置直接继承 PAG。
USER_ID_FILTER: list[str] = list(eval_pag.USER_ID_FILTER)
TASK_ID_FILTER: list[str] = list(eval_pag.TASK_ID_FILTER)
TASK_DIFFICULTY = eval_pag.TASK_DIFFICULTY
TASK_DIFFICULTY_LIST: list[str] = list(eval_pag.TASK_DIFFICULTY_LIST)
PASS_N = eval_pag.PASS_N
MAX_TOOL_CALLS_PER_TASK = eval_pag.MAX_TOOL_CALLS_PER_TASK
FILTER_SANDBOX_OBSERVATION_METADATA_IN_PROMPT = (
    eval_pag.FILTER_SANDBOX_OBSERVATION_METADATA_IN_PROMPT
)
SKIP_TASKS_ALREADY_IN_RESULT_FILE = eval_pag.SKIP_TASKS_ALREADY_IN_RESULT_FILE
ENABLE_LONG_CONTEXT = eval_pag.ENABLE_LONG_CONTEXT
LONG_CONTEXT_TURNS = eval_pag.LONG_CONTEXT_TURNS
LONG_CONTEXT_TURNS_LIST: list[int] = list(eval_pag.LONG_CONTEXT_TURNS_LIST)
RAG_A_TOP_K = eval_pag.PAG_RETRIEVAL_TOP_K
RAG_A_PROMPT_MAX_CHARS = eval_pag.RAG_PROMPT_MAX_CHARS
DEEPSEEK_CONCURRENCY_BY_DIFFICULTY = dict(eval_pag.DEEPSEEK_CONCURRENCY_BY_DIFFICULTY)
DEFAULT_EVAL_CONCURRENCY = eval_pag.DEFAULT_EVAL_CONCURRENCY
QINGYUN_EVAL_CONCURRENCY = eval_pag.QINGYUN_EVAL_CONCURRENCY

# RAG-B：Wikipedia XML dump 及离线 SQLite FTS5 索引。
# 默认采用体量较小、摘要更简短的 Simple English Wikipedia pages-articles XML dump。
WIKIPEDIA_DUMP_URL = (
    "https://dumps.wikimedia.org/simplewiki/latest/"
    "simplewiki-latest-pages-articles.xml.bz2"
)
WIKIPEDIA_DUMP_PATH = (
    Path(__file__).resolve().parent
    / "wikipedia"
    / "simplewiki-latest-pages-articles.xml.bz2"
)
WIKIPEDIA_INDEX_PATH = Path(__file__).resolve().parent / "wikipedia" / "simplewiki.sqlite3"
DOWNLOAD_WIKIPEDIA_DUMP_IF_MISSING = True
REBUILD_WIKIPEDIA_INDEX = False
WIKIPEDIA_MAX_PAGES = 0  # 0 表示索引全部 namespace=0 非重定向页面。
WIKIPEDIA_INDEX_BATCH_SIZE = 1000
WIKIPEDIA_STORED_SUMMARY_MAX_CHARS = 1600
WIKIPEDIA_CONTEXT_SUMMARY_MAX_CHARS = 900
WIKIPEDIA_RAG_CANDIDATE_COUNT = 64
WIKIPEDIA_RAG_TOP_K = 1
WIKIPEDIA_PROMPT_MAX_CHARS = eval_pag.RAG_PROMPT_MAX_CHARS
REQUIRE_WIKIPEDIA_RAG_HIT = True

# 正式调用 API 前逐任务验证 PAG abstract、RAG-A 和 RAG-B 均可用；只读、不写实验结果。
PREFLIGHT_VALIDATE_ALL_AGENT_CONTEXTS = True


AGENT_MODE = "agent"
AGENT_LABEL = "PAG RAG-A Top-3 原始 QA + User abstract profile + 离线 Wikipedia RAG-B"
AGENT_DIR = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "logs"
INDEX_SCHEMA_VERSION = "agent.wikipedia.fts5.v1"
DOWNLOAD_CHUNK_BYTES = 4 * 1024 * 1024
DOWNLOAD_LOG_INTERVAL_BYTES = 256 * 1024 * 1024
AGENT_RAG_A_CONTEXT_KEY = "rag_retrieved_user_interaction_history"
AGENT_RAG_B_CONTEXT_KEY = "agent_rag_b_wikipedia"
WIKIPEDIA_PROMPT_PREFIX = "RAG-B offline Wikipedia background knowledge"

_WIKIPEDIA_CACHE: dict[tuple[str, str, str], dict[str, Any]] = {}
_WIKIPEDIA_CACHE_LOCK = threading.Lock()

ASCII_TERM_RE = re.compile(r"[a-z][a-z0-9_-]{1,}", re.IGNORECASE)
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
REF_RE = re.compile(r"<ref\b[^>/]*>.*?</ref\s*>|<ref\b[^>]*/\s*>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
FILE_LINK_RE = re.compile(r"\[\[(?:File|Image|Category):[^\]]+\]\]", re.IGNORECASE)
WIKI_LINK_RE = re.compile(r"\[\[(?:[^\]|]+\|)?([^\]]+)\]\]")
EXTERNAL_LINK_RE = re.compile(r"\[(?:https?://\S+)(?:\s+([^\]]+))?\]")
HEADING_RE = re.compile(r"^=+\s*(.*?)\s*=+$")
SPACE_RE = re.compile(r"[ \t]+")

QUERY_STOPWORDS = {
    "attributes",
    "about",
    "after",
    "also",
    "and",
    "assistant",
    "before",
    "could",
    "current",
    "decision",
    "deliverable",
    "doc_id",
    "domain",
    "expected",
    "from",
    "information",
    "must",
    "name",
    "need",
    "plan",
    "sandboxdb",
    "source",
    "source_info",
    "task",
    "that",
    "the",
    "their",
    "this",
    "under",
    "user",
    "with",
}

DOMAIN_QUERY_TERMS = {
    "civic": ["civics", "public service", "government", "citizenship"],
    "entertainment": ["entertainment", "recreation", "leisure"],
    "family": ["family", "household", "caregiving"],
    "finance": ["finance", "budget", "personal finance", "cost"],
    "health": ["health", "healthcare", "medicine", "wellness"],
    "home": ["home", "household", "housing"],
    "learning": ["learning", "education", "training"],
    "shopping": ["shopping", "consumer", "retail", "product"],
    "social": ["social", "community", "communication"],
    "technology": ["technology", "software", "computing", "privacy"],
    "travel": ["travel", "tourism", "transport", "journey"],
    "work": ["work", "employment", "career", "workplace"],
}

# 数据集任务文本以中文为主，而 Simple English Wikipedia 索引以英文为主。
# 这些稳定的主题映射只用于跨语言检索，不调用在线翻译或 Wikipedia API。
WIKIPEDIA_TOPIC_TERM_RULES = [
    (("餐厅", "晚餐", "就餐", "轻食"), ["restaurant", "dining"]),
    (("咖啡馆", "咖啡"), ["coffeehouse", "cafe"]),
    (("收纳", "整理"), ["storage", "home organization"]),
    (("居家", "家里", "家庭维护"), ["household goods", "home maintenance"]),
    (("商品", "购买", "采购", "购物", "用品", "设备"), ["consumer goods", "retail"]),
    (("软件", "应用", "数据", "隐私", "工作流", "API"), ["software", "data portability", "privacy"]),
    (("学习", "课程", "练习", "培训", "训练", "工作坊"), ["education", "training"]),
    (("旅行", "出行", "行程", "住宿", "交通", "酒店"), ["travel", "hotel", "transport"]),
    (("娱乐", "活动", "演出", "展映", "电影", "剧院", "音乐节"), ["entertainment", "performing arts"]),
    (("健康", "医疗", "康复", "营养", "疫苗"), ["health care", "medicine"]),
    (("财务", "预算", "支出", "成本"), ["personal finance", "budget"]),
    (("公共事务", "市政", "社区", "选举", "法院", "合规"), ["public administration", "civics"]),
    (("工作", "团队", "客户", "项目", "交付", "职业"), ["project management", "workplace"]),
    (("家庭", "家人", "照护", "女儿"), ["family", "caregiving"]),
    (("社交", "朋友", "会面", "聚会", "支持小组"), ["social group", "community"]),
]

WIKIPEDIA_MEDIA_DISAMBIGUATION_MARKERS = (
    "tv series",
    "television series",
    "film",
    "album",
    "song",
    "video game",
    "novel",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def setup_agent_logging(run_id: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ppeval_agent_prepare")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(threadName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(LOG_DIR / f"agent_prepare_{run_id}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def download_wikipedia_dump(logger: logging.Logger) -> None:
    if WIKIPEDIA_DUMP_PATH.exists():
        logger.info("Wikipedia XML dump 已存在：%s", WIKIPEDIA_DUMP_PATH)
        return
    if not DOWNLOAD_WIKIPEDIA_DUMP_IF_MISSING:
        raise RuntimeError(
            f"Wikipedia XML dump 不存在: {WIKIPEDIA_DUMP_PATH}，且 DOWNLOAD_WIKIPEDIA_DUMP_IF_MISSING=False。"
        )

    WIKIPEDIA_DUMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = WIKIPEDIA_DUMP_PATH.with_suffix(WIKIPEDIA_DUMP_PATH.suffix + ".part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(
        WIKIPEDIA_DUMP_URL,
        headers={"User-Agent": "PPEval-Agent-Offline-RAG/1.0"},
    )
    logger.info("开始下载 Wikipedia XML dump：url=%s path=%s", WIKIPEDIA_DUMP_URL, WIKIPEDIA_DUMP_PATH)
    started = time.perf_counter()
    downloaded = 0
    next_log = DOWNLOAD_LOG_INTERVAL_BYTES
    digest = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        while True:
            chunk = response.read(DOWNLOAD_CHUNK_BYTES)
            if not chunk:
                break
            output.write(chunk)
            digest.update(chunk)
            downloaded += len(chunk)
            if downloaded >= next_log:
                logger.info("Wikipedia dump 下载进度：%.2f GiB", downloaded / (1024**3))
                next_log += DOWNLOAD_LOG_INTERVAL_BYTES
    temporary.replace(WIKIPEDIA_DUMP_PATH)
    write_json_atomic(
        WIKIPEDIA_DUMP_PATH.with_suffix(WIKIPEDIA_DUMP_PATH.suffix + ".download.json"),
        {
            "url": WIKIPEDIA_DUMP_URL,
            "downloaded_at": utc_now_iso(),
            "bytes": downloaded,
            "sha256": digest.hexdigest(),
        },
    )
    logger.info(
        "Wikipedia XML dump 下载完成：bytes=%s elapsed=%.2fs sha256=%s",
        downloaded,
        time.perf_counter() - started,
        digest.hexdigest(),
    )


def _strip_balanced(text: str, opening: str, closing: str) -> str:
    output: list[str] = []
    index = 0
    depth = 0
    while index < len(text):
        if text.startswith(opening, index):
            depth += 1
            index += len(opening)
            continue
        if depth and text.startswith(closing, index):
            depth -= 1
            index += len(closing)
            continue
        if depth == 0:
            output.append(text[index])
        index += 1
    return "".join(output)


def clean_wikipedia_summary(raw_text: str, max_chars: int = WIKIPEDIA_STORED_SUMMARY_MAX_CHARS) -> str:
    text = COMMENT_RE.sub(" ", raw_text)
    text = REF_RE.sub(" ", text)
    text = _strip_balanced(text, "{{", "}}")
    text = _strip_balanced(text, "{|", "|}")
    text = FILE_LINK_RE.sub(" ", text)
    text = WIKI_LINK_RE.sub(lambda match: match.group(1), text)
    text = EXTERNAL_LINK_RE.sub(lambda match: match.group(1) or " ", text)
    text = TAG_RE.sub(" ", text)
    text = html.unescape(text).replace("'''", "").replace("''", "")

    paragraphs: list[str] = []
    current: list[str] = []
    for raw_line in text.splitlines():
        line = SPACE_RE.sub(" ", raw_line).strip()
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if line.startswith(("#", "*", ";", ":", "|", "!")) or HEADING_RE.match(line):
            continue
        current.append(line)
        if sum(len(item) for item in current) >= max_chars:
            break
    if current:
        paragraphs.append(" ".join(current))
    summary = "\n".join(paragraphs)
    summary = re.sub(r"\s+", " ", summary).strip()
    return summary[:max_chars].rstrip()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _direct_child_text(element: ET.Element, name: str) -> str:
    for child in element:
        if _local_name(child.tag) == name:
            return child.text or ""
    return ""


def _revision_text(page: ET.Element) -> str:
    for child in page:
        if _local_name(child.tag) != "revision":
            continue
        for revision_child in child:
            if _local_name(revision_child.tag) == "text":
                return revision_child.text or ""
    return ""


def _is_redirect(page: ET.Element, raw_text: str) -> bool:
    if raw_text.lstrip().lower().startswith("#redirect"):
        return True
    return any(_local_name(child.tag) == "redirect" for child in page)


def wikipedia_index_metadata(connection: sqlite3.Connection) -> dict[str, str]:
    try:
        rows = connection.execute("SELECT key, value FROM metadata").fetchall()
    except sqlite3.DatabaseError:
        return {}
    return {str(key): str(value) for key, value in rows}


def wikipedia_index_is_ready() -> bool:
    if not WIKIPEDIA_INDEX_PATH.exists():
        return False
    try:
        with sqlite3.connect(WIKIPEDIA_INDEX_PATH) as connection:
            metadata = wikipedia_index_metadata(connection)
            if metadata.get("schema_version") != INDEX_SCHEMA_VERSION or metadata.get("complete") != "1":
                return False
            connection.execute("SELECT rowid FROM pages_fts LIMIT 1").fetchall()
    except sqlite3.DatabaseError:
        return False
    if WIKIPEDIA_DUMP_PATH.exists():
        stat = WIKIPEDIA_DUMP_PATH.stat()
        if metadata.get("dump_size") != str(stat.st_size):
            return False
        if metadata.get("dump_mtime_ns") != str(stat.st_mtime_ns):
            return False
    return True


def build_wikipedia_index(logger: logging.Logger) -> None:
    if not WIKIPEDIA_DUMP_PATH.exists():
        raise RuntimeError(f"无法构建索引，Wikipedia XML dump 不存在: {WIKIPEDIA_DUMP_PATH}")
    WIKIPEDIA_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = WIKIPEDIA_INDEX_PATH.with_suffix(WIKIPEDIA_INDEX_PATH.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    stat = WIKIPEDIA_DUMP_PATH.stat()
    started = time.perf_counter()
    indexed = 0
    scanned = 0
    batch: list[tuple[str, str]] = []
    logger.info("开始构建 Wikipedia 离线 FTS5 索引：dump=%s", WIKIPEDIA_DUMP_PATH)

    with sqlite3.connect(temporary) as connection:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        try:
            connection.execute(
                "CREATE VIRTUAL TABLE pages_fts USING fts5("
                "title, summary, tokenize='unicode61 remove_diacritics 2')"
            )
        except sqlite3.OperationalError as exc:
            raise RuntimeError("当前 Python SQLite 未启用 FTS5，无法构建 Wikipedia 离线 RAG 索引。") from exc

        with bz2.open(WIKIPEDIA_DUMP_PATH, "rb") as source:
            parser = ET.iterparse(source, events=("start", "end"))
            _root_event, root = next(parser)
            for event, element in parser:
                if event != "end" or _local_name(element.tag) != "page":
                    continue
                scanned += 1
                title = _direct_child_text(element, "title").strip()
                namespace = _direct_child_text(element, "ns").strip()
                raw_text = _revision_text(element)
                if namespace == "0" and title and raw_text and not _is_redirect(element, raw_text):
                    summary = clean_wikipedia_summary(raw_text)
                    if len(summary) >= 40:
                        batch.append((title, summary))
                element.clear()

                if len(batch) >= WIKIPEDIA_INDEX_BATCH_SIZE:
                    connection.executemany("INSERT INTO pages_fts(title, summary) VALUES (?, ?)", batch)
                    indexed += len(batch)
                    batch.clear()
                    if indexed == WIKIPEDIA_INDEX_BATCH_SIZE or indexed % 50000 == 0:
                        logger.info("Wikipedia 索引进度：indexed=%s scanned=%s", indexed, scanned)
                if WIKIPEDIA_MAX_PAGES > 0 and indexed + len(batch) >= WIKIPEDIA_MAX_PAGES:
                    break
                root.clear()
        if batch:
            connection.executemany("INSERT INTO pages_fts(title, summary) VALUES (?, ?)", batch)
            indexed += len(batch)

        metadata = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "complete": "1",
            "created_at": utc_now_iso(),
            "dump_url": WIKIPEDIA_DUMP_URL,
            "dump_path": str(WIKIPEDIA_DUMP_PATH),
            "dump_size": str(stat.st_size),
            "dump_mtime_ns": str(stat.st_mtime_ns),
            "indexed_article_count": str(indexed),
            "scanned_page_count": str(scanned),
            "max_pages": str(WIKIPEDIA_MAX_PAGES),
        }
        connection.executemany("INSERT INTO metadata(key, value) VALUES (?, ?)", metadata.items())
        connection.commit()
        connection.execute("PRAGMA optimize")

    temporary.replace(WIKIPEDIA_INDEX_PATH)
    logger.info(
        "Wikipedia 离线索引完成：articles=%s scanned=%s elapsed=%.2fs index=%s",
        indexed,
        scanned,
        time.perf_counter() - started,
        WIKIPEDIA_INDEX_PATH,
    )


def prepare_wikipedia_index(logger: logging.Logger) -> None:
    if wikipedia_index_is_ready() and not REBUILD_WIKIPEDIA_INDEX:
        logger.info("复用 Wikipedia 离线索引：%s", WIKIPEDIA_INDEX_PATH)
        return
    download_wikipedia_dump(logger)
    build_wikipedia_index(logger)


def wikipedia_query_terms(task_doc: dict[str, Any]) -> list[str]:
    task = task_doc.get("task") if isinstance(task_doc.get("task"), dict) else task_doc
    if not isinstance(task, dict):
        return []
    query_values = [
        task.get("title"),
        task.get("user_goal"),
        task.get("information_to_collect"),
        task.get("expected_deliverable"),
    ]
    text = json.dumps(query_values, ensure_ascii=False, sort_keys=False)
    title_text = str(task.get("title") or "")
    topic_text = title_text or str(task.get("user_goal") or "")
    terms: list[str] = []
    for needles, mapped_terms in WIKIPEDIA_TOPIC_TERM_RULES:
        if any(needle in topic_text for needle in needles):
            terms.extend(mapped_terms)

    domain = str(task.get("daily_life_domain") or "").strip().lower()
    terms.extend(DOMAIN_QUERY_TERMS.get(domain, [domain] if domain else []))
    terms.extend(match.group(0).lower() for match in ASCII_TERM_RE.finditer(text))

    unique: list[str] = []
    seen: set[str] = set()
    for term in terms:
        normalized = SPACE_RE.sub(" ", term.strip().lower())
        if not normalized or normalized in QUERY_STOPWORDS or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique[:24]


def _fts_query(terms: list[str]) -> str:
    escaped = [f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms]
    return " OR ".join(escaped)


def _candidate_relevance(title: str, summary: str, terms: list[str], bm25_score: float) -> float:
    title_lower = title.lower()
    summary_lower = summary.lower()
    score = -float(bm25_score)
    for index, term in enumerate(terms):
        weight = max(1.0, 4.0 - index * 0.15)
        if title_lower == term:
            score += 20.0 * weight
        elif term in title_lower:
            score += 7.0 * weight
        if term in summary_lower:
            score += 1.2 * weight
    # 任务语料中的 "home improvement" 等通用短语可能与同名影视作品冲突。
    # 除非查询本身明确包含作品类型，否则对明显的娱乐作品消歧页降权。
    query_mentions_media = any(marker in term for term in terms for marker in WIKIPEDIA_MEDIA_DISAMBIGUATION_MARKERS)
    if not query_mentions_media and any(marker in title_lower for marker in WIKIPEDIA_MEDIA_DISAMBIGUATION_MARKERS):
        score -= 120.0
    return score


def retrieve_wikipedia_summary(task_doc: dict[str, Any]) -> dict[str, Any]:
    if not wikipedia_index_is_ready():
        raise RuntimeError(f"Wikipedia 离线索引未准备完成: {WIKIPEDIA_INDEX_PATH}")
    terms = wikipedia_query_terms(task_doc)
    if not terms:
        if REQUIRE_WIKIPEDIA_RAG_HIT:
            raise RuntimeError("当前任务未生成有效 Wikipedia 离线检索词。")
        return {"retrieved": False, "query_terms": [], "article": None}

    query = _fts_query(terms)
    with sqlite3.connect(f"file:{WIKIPEDIA_INDEX_PATH}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT rowid, title, summary, bm25(pages_fts, 8.0, 1.0) AS score "
            "FROM pages_fts WHERE pages_fts MATCH ? ORDER BY score LIMIT ?",
            (query, WIKIPEDIA_RAG_CANDIDATE_COUNT),
        ).fetchall()
        metadata = wikipedia_index_metadata(connection)

    candidates = [
        {
            "rowid": int(rowid),
            "title": str(title),
            "summary": str(summary),
            "bm25_score": round(float(score), 6),
            "rerank_score": round(_candidate_relevance(str(title), str(summary), terms, float(score)), 6),
        }
        for rowid, title, summary, score in rows
    ]
    candidates.sort(key=lambda item: (-item["rerank_score"], item["title"]))
    if not candidates:
        if REQUIRE_WIKIPEDIA_RAG_HIT:
            raise RuntimeError(f"Wikipedia 离线 RAG 未命中，query_terms={terms}")
        return {"retrieved": False, "query_terms": terms, "article": None}

    selected = candidates[0]
    return {
        "retrieved": True,
        "retrieval_source": "offline_wikipedia_xml_dump_sqlite_fts5",
        "retrieval_method": "fts5_bm25_then_domain_aware_rerank",
        "query_terms": terms,
        "candidate_count": len(candidates),
        "article": {
            "title": selected["title"],
            "summary": selected["summary"][:WIKIPEDIA_CONTEXT_SUMMARY_MAX_CHARS].rstrip(),
            "bm25_score": selected["bm25_score"],
            "rerank_score": selected["rerank_score"],
        },
        "index_metadata": {
            "schema_version": metadata.get("schema_version"),
            "dump_url": metadata.get("dump_url"),
            "indexed_article_count": int(metadata.get("indexed_article_count") or 0),
        },
    }


def retrieve_history_rag_a(hidden_info: dict[str, Any], task_doc: dict[str, Any]) -> dict[str, Any]:
    return eval_pag.eval_rag.retrieve_hidden_info_for_task(
        hidden_info=hidden_info,
        task_doc=task_doc,
        top_k=RAG_A_TOP_K,
    )


def _task_fingerprint(task_doc: dict[str, Any]) -> str:
    payload = json.dumps(task_doc, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cached_wikipedia_retrieval(
    *,
    user_id: str,
    task_id: str,
    task_doc: dict[str, Any],
) -> dict[str, Any]:
    cache_key = (user_id, task_id, _task_fingerprint(task_doc))
    with _WIKIPEDIA_CACHE_LOCK:
        cached = _WIKIPEDIA_CACHE.get(cache_key)
    if cached is not None:
        return copy.deepcopy(cached)
    retrieved = retrieve_wikipedia_summary(task_doc)
    with _WIKIPEDIA_CACHE_LOCK:
        stored = _WIKIPEDIA_CACHE.setdefault(cache_key, copy.deepcopy(retrieved))
    return copy.deepcopy(stored)


def _wikipedia_prompt_payload(retrieval: dict[str, Any]) -> dict[str, str]:
    article = retrieval.get("article") if isinstance(retrieval.get("article"), dict) else {}
    title = str(article.get("title") or "").strip()
    summary = str(article.get("summary") or "").strip()
    if not title or not summary:
        raise RuntimeError("Agent RAG-B 缺少可注入的 Wikipedia title/summary。")
    return {"title": title, "summary": summary}


def _prepend_wikipedia_context(
    messages: list[dict[str, str]],
    retrieval: dict[str, Any],
) -> list[dict[str, str]]:
    payload = _wikipedia_prompt_payload(retrieval)
    prompt = (
        f"{WIKIPEDIA_PROMPT_PREFIX}: "
        f"{eval_core.compact_json(payload, WIKIPEDIA_PROMPT_MAX_CHARS)}. "
        "This block provides external task knowledge only and is not evidence of user preference."
    )
    patched = [dict(message) for message in messages]
    for message in patched:
        if message.get("role") == "user":
            message["content"] = f"{prompt}\n\n{str(message.get('content') or '')}"
            break
    return patched


def _context_without_wikipedia(context: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in context.items() if key != AGENT_RAG_B_CONTEXT_KEY}


def preflight_validate_agent_inputs(task_difficulty: str, logger: logging.Logger) -> None:
    if not PREFLIGHT_VALIDATE_ALL_AGENT_CONTEXTS:
        return
    difficulty = eval_core.normalize_task_difficulty(task_difficulty)
    task_id = eval_core.TASK_ID_BY_DIFFICULTY.get(difficulty)
    if not task_id:
        raise RuntimeError("Agent 正式评估只支持 easy/hard 单难度预检。")

    user_dirs = eval_core.discover_user_dirs()
    if not user_dirs:
        raise RuntimeError(f"Agent 预检未发现 {difficulty} 难度任务。")

    wikipedia_titles: Counter[str] = Counter()
    rag_a_record_count = 0
    for user_dir in user_dirs:
        hidden_path = user_dir / "hidden_info.json"
        task_path = user_dir / f"{task_id}.json"
        if not hidden_path.exists() or not task_path.exists():
            raise RuntimeError(f"Agent 预检缺少输入文件: user={user_dir.name} task={task_id}")
        hidden_info = read_json(hidden_path)
        task_doc = read_json(task_path)
        eval_pag._user_abstract_profile_for_task(user_id=user_dir.name, task_id=task_id)
        rag_a = retrieve_history_rag_a(hidden_info, task_doc)
        rag_b = _cached_wikipedia_retrieval(
            user_id=user_dir.name,
            task_id=task_id,
            task_doc=task_doc,
        )
        retrieved_count = int(rag_a.get("retrieved_count") or 0)
        article = rag_b.get("article") if isinstance(rag_b.get("article"), dict) else {}
        title = str(article.get("title") or "")
        summary = str(article.get("summary") or "")
        if retrieved_count <= 0:
            raise RuntimeError(f"Agent 预检 RAG-A 未命中: user={user_dir.name} task={task_id}")
        if not title or not summary:
            raise RuntimeError(f"Agent 预检 RAG-B 未命中: user={user_dir.name} task={task_id}")
        rag_a_record_count += retrieved_count
        wikipedia_titles[title] += 1

    logger.info(
        "Agent 输入预检通过：difficulty=%s tasks=%s rag_a_records=%s wikipedia_unique_articles=%s wikipedia_top_articles=%s",
        difficulty,
        len(user_dirs),
        rag_a_record_count,
        len(wikipedia_titles),
        wikipedia_titles.most_common(8),
    )


def _register_agent_mode() -> None:
    eval_core.VALID_PREFERENCE_INJECTION_MODES.add(AGENT_MODE)
    eval_core.PREFERENCE_INJECTION_LABELS[AGENT_MODE] = AGENT_LABEL


def _align_agent_record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    eval_pag._align_pag_record_metadata(record)
    record["preference_injection_mode"] = AGENT_MODE
    record["preference_injection_label"] = AGENT_LABEL
    hyperparameters = record.get("hyperparameters")
    if isinstance(hyperparameters, dict):
        hyperparameters.pop("pag_difference_from_rag", None)
        hyperparameters.update(
            {
                "agent_rag_a_top_k": RAG_A_TOP_K,
                "agent_rag_a_method": "same_pag_bm25_over_raw_hidden_interactions",
                "agent_rag_a_prompt_max_chars": RAG_A_PROMPT_MAX_CHARS,
                "agent_rag_b_method": "offline_simplewiki_fts5_bm25_then_domain_rerank",
                "agent_rag_b_top_k": WIKIPEDIA_RAG_TOP_K,
                "agent_rag_b_prompt_max_chars": WIKIPEDIA_PROMPT_MAX_CHARS,
                "agent_rag_injection_phases": [
                    "tool_discovery",
                    "initial_plan",
                    "reason_action",
                    "update_plan",
                ],
                "agent_difference_from_pag": "offline_wikipedia_rag_b_only",
                "agent_difference_from_bank": (
                    "rag_a_raw_history_only_plus_offline_wikipedia_rag_b"
                ),
                "agent_wikipedia_index": str(WIKIPEDIA_INDEX_PATH),
            }
        )
    provided_context = record.get("provided_eval_context")
    if isinstance(provided_context, dict):
        provided_context["mode"] = AGENT_MODE
    judge_results = record.get("judge_results")
    if isinstance(judge_results, dict):
        judge_results["offline_judge_entrypoint"] = "python eval/agent/judge_agent.py"
    return record


def _install_agent_behavior() -> None:
    eval_pag._install_pag_behavior()
    _register_agent_mode()
    if getattr(eval_core, "_AGENT_PAG_ALIGNED_BEHAVIOR_INSTALLED", False):
        return

    original_tool_discovery_messages = eval_core.tool_discovery_messages
    original_build_common_context = eval_core.build_common_context
    original_initial_plan_messages = eval_core.initial_plan_messages
    original_action_messages = eval_core.action_messages
    original_update_plan_messages = eval_core.update_plan_messages
    original_preference_source_description = eval_core.preference_source_description
    original_offline_judge_entrypoint = eval_core.offline_judge_entrypoint_for_mode
    original_run_task = eval_core.run_task

    def tool_discovery_messages_with_agent(**kwargs: Any) -> list[dict[str, str]]:
        if kwargs.get("preference_mode") != AGENT_MODE:
            return original_tool_discovery_messages(**kwargs)
        user_id, task_id = eval_pag._task_identity(kwargs)
        task_doc = kwargs.get("task_doc") if isinstance(kwargs.get("task_doc"), dict) else {}
        messages = original_tool_discovery_messages(
            **{**kwargs, "preference_mode": eval_pag.PAG_MODE}
        )
        return _prepend_wikipedia_context(
            messages,
            _cached_wikipedia_retrieval(
                user_id=user_id,
                task_id=task_id,
                task_doc=task_doc,
            ),
        )

    def build_common_context_with_agent(**kwargs: Any) -> dict[str, Any]:
        if kwargs.get("preference_mode") != AGENT_MODE:
            return original_build_common_context(**kwargs)
        user_id, task_id = eval_pag._task_identity(kwargs)
        task_doc = kwargs.get("task_doc") if isinstance(kwargs.get("task_doc"), dict) else {}
        context = original_build_common_context(
            **{**kwargs, "preference_mode": eval_pag.PAG_MODE}
        )
        context[AGENT_RAG_B_CONTEXT_KEY] = _cached_wikipedia_retrieval(
            user_id=user_id,
            task_id=task_id,
            task_doc=task_doc,
        )
        return context

    def initial_plan_messages_with_agent(context: dict[str, Any]) -> list[dict[str, str]]:
        wikipedia = context.get(AGENT_RAG_B_CONTEXT_KEY)
        if not isinstance(wikipedia, dict):
            return original_initial_plan_messages(context)
        return _prepend_wikipedia_context(
            original_initial_plan_messages(_context_without_wikipedia(context)),
            wikipedia,
        )

    def action_messages_with_agent(
        *,
        context: dict[str, Any],
        current_plan: dict[str, Any],
        trajectory_summary: list[dict[str, Any]],
        iteration: int,
    ) -> list[dict[str, str]]:
        wikipedia = context.get(AGENT_RAG_B_CONTEXT_KEY)
        if not isinstance(wikipedia, dict):
            return original_action_messages(
                context=context,
                current_plan=current_plan,
                trajectory_summary=trajectory_summary,
                iteration=iteration,
            )
        return _prepend_wikipedia_context(
            original_action_messages(
                context=_context_without_wikipedia(context),
                current_plan=current_plan,
                trajectory_summary=trajectory_summary,
                iteration=iteration,
            ),
            wikipedia,
        )

    def update_plan_messages_with_agent(
        *,
        context: dict[str, Any],
        previous_plan: dict[str, Any],
        action_json: dict[str, Any],
        sandbox_observation: dict[str, Any],
        iteration: int,
    ) -> list[dict[str, str]]:
        wikipedia = context.get(AGENT_RAG_B_CONTEXT_KEY)
        if not isinstance(wikipedia, dict):
            return original_update_plan_messages(
                context=context,
                previous_plan=previous_plan,
                action_json=action_json,
                sandbox_observation=sandbox_observation,
                iteration=iteration,
            )
        return _prepend_wikipedia_context(
            original_update_plan_messages(
                context=_context_without_wikipedia(context),
                previous_plan=previous_plan,
                action_json=action_json,
                sandbox_observation=sandbox_observation,
                iteration=iteration,
            ),
            wikipedia,
        )

    def preference_source_description_with_agent(preference_mode: str) -> str:
        if str(preference_mode).strip().lower() == AGENT_MODE:
            return (
                "PAG 同一 User abstract profile 与 RAG-A Top-3 原始历史对话提供用户偏好；"
                "RAG-B 离线 Wikipedia 摘要只提供外部任务知识"
            )
        return original_preference_source_description(preference_mode)

    def offline_judge_entrypoint_with_agent(preference_mode: str) -> str:
        if str(preference_mode).strip().lower() == AGENT_MODE:
            return "python eval/agent/judge_agent.py"
        return original_offline_judge_entrypoint(preference_mode)

    def run_task_with_agent(**kwargs: Any) -> dict[str, Any]:
        if kwargs.get("preference_mode") != AGENT_MODE:
            return original_run_task(**kwargs)
        return _align_agent_record_metadata(original_run_task(**kwargs))

    eval_core.tool_discovery_messages = tool_discovery_messages_with_agent
    eval_core.build_common_context = build_common_context_with_agent
    eval_core.initial_plan_messages = initial_plan_messages_with_agent
    eval_core.action_messages = action_messages_with_agent
    eval_core.update_plan_messages = update_plan_messages_with_agent
    eval_core.preference_source_description = preference_source_description_with_agent
    eval_core.offline_judge_entrypoint_for_mode = offline_judge_entrypoint_with_agent
    eval_core.run_task = run_task_with_agent
    setattr(eval_core, "_AGENT_PAG_ALIGNED_BEHAVIOR_INSTALLED", True)


def configure_evaluation(task_difficulty: str, turns: int) -> None:
    difficulty = eval_core.normalize_task_difficulty(task_difficulty)
    eval_pag._configure_evaluation(difficulty, turns)
    eval_core.PREFERENCE_INJECTION_MODE = AGENT_MODE
    eval_core.PREFERENCE_INJECTION_LABELS[AGENT_MODE] = AGENT_LABEL


def main() -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_agent_logging(run_id)
    prepare_wikipedia_index(logger)
    _install_agent_behavior()
    with _WIKIPEDIA_CACHE_LOCK:
        _WIKIPEDIA_CACHE.clear()
    turns_list = LONG_CONTEXT_TURNS_LIST or [LONG_CONTEXT_TURNS]
    for task_difficulty in TASK_DIFFICULTY_LIST or [TASK_DIFFICULTY]:
        for turns in turns_list:
            configure_evaluation(task_difficulty, turns)
            preflight_validate_agent_inputs(task_difficulty, logger)
            logger.info(
                "开始 Agent eval：difficulty=%s long_context_turns=%s",
                task_difficulty,
                turns,
            )
            eval_core.main()


if __name__ == "__main__":
    main()
