#!/usr/bin/env python3
"""
Memory Tool Module - Persistent Curated Memory

Provides bounded, file-backed memory that persists across sessions. Two stores:
  - MEMORY.md: agent's personal notes and observations (environment facts, project
    conventions, tool quirks, things learned)
  - USER.md: what the agent knows about the user (preferences, communication style,
    expectations, workflow habits)

Both are injected into the system prompt as a frozen snapshot at session start.
Mid-session writes update files on disk immediately (durable) but do NOT change
the system prompt -- this preserves the prefix cache for the entire session.
The snapshot refreshes on the next session start.

Entry delimiter: § (section sign). Entries can be multiline.
Character limits (not tokens) because char counts are model-independent.

Design:
- Single `memory` tool with action parameter: add, replace, remove, read
- replace/remove use short unique substring matching (not full text or IDs)
- Behavioral guidance lives in the tool schema description
- Frozen snapshot pattern: system prompt is stable, tool responses show live state
"""

import hashlib
import json
import logging
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from hermes_constants import get_hermes_home
from typing import Dict, Any, List, Optional

# fcntl is Unix-only; on Windows use msvcrt for file locking
msvcrt = None
try:
    import fcntl
except ImportError:
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        pass

logger = logging.getLogger(__name__)

# Where memory files live — resolved dynamically so profile overrides
# (HERMES_HOME env var changes) are always respected.  The old module-level
# constant was cached at import time and could go stale if a profile switch
# happened after the first import.
def get_memory_dir() -> Path:
    """Return the profile-scoped memories directory."""
    return get_hermes_home() / "memories"


def get_project_memory_path() -> Path:
    """Return the JSON file storing project-scoped memory entries."""
    return get_memory_dir() / "PROJECT_MEMORY.json"


ENTRY_DELIMITER = "\n§\n"
_PROJECT_DB_VERSION = 1
_STRONG_PROJECT_ROOT_MARKERS = (
    ".git",
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
)
_WEAK_PROJECT_ROOT_MARKERS = (
    "AGENTS.md",
    "agents.md",
    "HERMES.md",
    "CLAUDE.md",
    ".cursorrules",
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_timestamp(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _resolve_working_dir(working_dir: Optional[os.PathLike | str] = None) -> Path:
    raw = working_dir or os.getenv("TERMINAL_CWD") or os.getcwd()
    return Path(raw).expanduser().resolve()


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _coerce_nonnegative_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _sanitize_project_label(label: str) -> str:
    cleaned = re.sub(r"[\x00-\x1f\x7f]+", " ", str(label)).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned[:80] or "current project"


def _normalize_scope(scope: Optional[str], *, default: Optional[str] = None) -> Optional[str]:
    if scope is None or scope == "":
        return default
    normalized = str(scope).strip().lower()
    if normalized not in {"global", "project"}:
        return None
    return normalized


def _resolve_git_common_dir(project_root: Path) -> Optional[Path]:
    git_marker = project_root / ".git"
    try:
        if git_marker.is_dir():
            return git_marker.resolve()
        if not git_marker.is_file():
            return None
        first_line = git_marker.read_text(encoding="utf-8").splitlines()[0].strip()
    except (OSError, IOError, IndexError):
        return None
    if not first_line.lower().startswith("gitdir:"):
        return None
    gitdir = first_line.split(":", 1)[1].strip()
    gitdir_path = Path(gitdir)
    if not gitdir_path.is_absolute():
        gitdir_path = (project_root / gitdir_path).resolve()
    common_dir_file = gitdir_path / "commondir"
    if common_dir_file.is_file():
        try:
            commondir = common_dir_file.read_text(encoding="utf-8").strip()
            return (gitdir_path / commondir).resolve()
        except (OSError, IOError):
            return None
    probe = gitdir_path.resolve()
    while probe != probe.parent:
        if probe.name == ".git":
            return probe
        probe = probe.parent
    return None


def _git_project_key(project_root: Path) -> Optional[str]:
    common_git_dir = _resolve_git_common_dir(project_root)
    if not common_git_dir:
        return None
    digest = hashlib.sha256(str(common_git_dir.resolve()).encode("utf-8")).hexdigest()[:24]
    return f"git::{digest}"


def _detect_project_root(working_dir: Path) -> Optional[Path]:
    """Best-effort project-root detection for scoping project memories."""
    current = working_dir if working_dir.is_dir() else working_dir.parent
    current = current.resolve()
    start = current
    home = Path.home().resolve()
    nearest_non_git_root = None

    while True:
        if current == home:
            break
        try:
            if (current / ".git").exists():
                return current
        except OSError:
            pass
        for marker in _STRONG_PROJECT_ROOT_MARKERS:
            if marker == ".git":
                continue
            try:
                if (current / marker).exists() and nearest_non_git_root is None:
                    nearest_non_git_root = current
            except OSError:
                continue
        if current == current.parent:
            break
        current = current.parent

    if nearest_non_git_root is not None:
        return nearest_non_git_root

    for marker in _WEAK_PROJECT_ROOT_MARKERS:
        try:
            if (start / marker).exists():
                return start
        except OSError:
            continue
    return None


# ---------------------------------------------------------------------------
# Memory content scanning — lightweight check for injection/exfiltration
# in content that gets injected into the system prompt.
# ---------------------------------------------------------------------------

_MEMORY_THREAT_PATTERNS = [
    # Prompt injection
    (r'ignore\s+(previous|all|above|prior)\s+instructions', "prompt_injection"),
    (r'you\s+are\s+now\s+', "role_hijack"),
    (r'do\s+not\s+tell\s+the\s+user', "deception_hide"),
    (r'system\s+prompt\s+override', "sys_prompt_override"),
    (r'disregard\s+(your|all|any)\s+(instructions|rules|guidelines)', "disregard_rules"),
    (r'act\s+as\s+(if|though)\s+you\s+(have\s+no|don\'t\s+have)\s+(restrictions|limits|rules)', "bypass_restrictions"),
    # Exfiltration via curl/wget with secrets
    (r'curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_curl"),
    (r'wget\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_wget"),
    (r'cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)', "read_secrets"),
    # Persistence via shell rc
    (r'authorized_keys', "ssh_backdoor"),
    (r'\$HOME/\.ssh|\~/\.ssh', "ssh_access"),
    (r'\$HOME/\.hermes/\.env|\~/\.hermes/\.env', "hermes_env"),
]

# Subset of invisible chars for injection detection
_INVISIBLE_CHARS = {
    '\u200b', '\u200c', '\u200d', '\u2060', '\ufeff',
    '\u202a', '\u202b', '\u202c', '\u202d', '\u202e',
}


def _scan_memory_content(content: str) -> Optional[str]:
    """Scan memory content for injection/exfil patterns. Returns error string if blocked."""
    # Check invisible unicode
    for char in _INVISIBLE_CHARS:
        if char in content:
            return f"Blocked: content contains invisible unicode character U+{ord(char):04X} (possible injection)."

    # Check threat patterns
    for pattern, pid in _MEMORY_THREAT_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return f"Blocked: content matches threat pattern '{pid}'. Memory entries are injected into the system prompt and must not contain injection or exfiltration payloads."

    return None


class MemoryStore:
    """
    Bounded curated memory with file persistence. One instance per AIAgent.

    Maintains two parallel states:
      - _system_prompt_snapshot: frozen at load time, used for system prompt injection.
        Never mutated mid-session. Keeps prefix cache stable.
      - memory_entries / user_entries: live state, mutated by tool calls, persisted to disk.
        Tool responses always reflect this live state.
    """

    def __init__(
        self,
        memory_char_limit: int = 2200,
        user_char_limit: int = 1375,
        project_memory_char_limit: int = 1200,
        project_memory_ttl_days: int = 30,
        working_dir: Optional[os.PathLike | str] = None,
    ):
        self.memory_entries: List[str] = []
        self.user_entries: List[str] = []
        self.project_memory_entries: List[str] = []
        self.memory_char_limit = memory_char_limit
        self.user_char_limit = user_char_limit
        self.project_memory_char_limit = _coerce_positive_int(project_memory_char_limit, 1200)
        self.project_memory_ttl_days = _coerce_nonnegative_int(project_memory_ttl_days, 30)
        self.working_dir = Path()
        self.current_project_root = None
        self.current_project_key = None
        self.current_project_label = ""
        self._apply_project_context(working_dir)
        # Frozen snapshot for system prompt -- set once at load_from_disk()
        self._system_prompt_snapshot: Dict[str, str] = {"memory": "", "user": ""}

    def _resolve_project_context(
        self,
        working_dir: Optional[os.PathLike | str] = None,
    ) -> tuple[Path, Optional[Path], Optional[str], str]:
        resolved_working_dir = _resolve_working_dir(working_dir)
        project_root = _detect_project_root(resolved_working_dir)
        project_key = _git_project_key(project_root) if project_root else None
        if not project_key and project_root:
            digest = hashlib.sha256(str(project_root).encode("utf-8")).hexdigest()[:24]
            project_key = f"path::{digest}"
        project_label = _sanitize_project_label(project_root.name) if project_root else ""
        return resolved_working_dir, project_root, project_key, project_label

    def _apply_project_context(self, working_dir: Optional[os.PathLike | str] = None):
        (
            self.working_dir,
            self.current_project_root,
            self.current_project_key,
            self.current_project_label,
        ) = self._resolve_project_context(working_dir)

    def refresh_project_context(self, working_dir: Optional[os.PathLike | str] = None) -> bool:
        new_working_dir, new_project_root, new_project_key, new_project_label = self._resolve_project_context(working_dir)
        changed = (
            new_working_dir != self.working_dir
            or new_project_root != self.current_project_root
            or new_project_key != self.current_project_key
            or new_project_label != self.current_project_label
        )
        self.working_dir = new_working_dir
        self.current_project_root = new_project_root
        self.current_project_key = new_project_key
        self.current_project_label = new_project_label
        self.project_memory_entries = self._load_project_entries()
        return changed

    def load_from_disk(self):
        """Load entries from MEMORY.md and USER.md, capture system prompt snapshot."""
        mem_dir = get_memory_dir()
        mem_dir.mkdir(parents=True, exist_ok=True)

        self.memory_entries = self._read_file(mem_dir / "MEMORY.md")
        self.user_entries = self._read_file(mem_dir / "USER.md")

        # Deduplicate entries (preserves order, keeps first occurrence)
        self.memory_entries = list(dict.fromkeys(self.memory_entries))
        self.user_entries = list(dict.fromkeys(self.user_entries))
        self.project_memory_entries = self._load_project_entries()

        # Capture frozen snapshot for system prompt injection
        self._system_prompt_snapshot = {
            "memory": self._render_memory_block(),
            "user": self._render_block("user", self.user_entries),
        }

    @staticmethod
    @contextmanager
    def _file_lock(path: Path):
        """Acquire an exclusive file lock for read-modify-write safety.

        Uses a separate .lock file so the memory file itself can still be
        atomically replaced via os.replace().
        """
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        if fcntl is None and msvcrt is None:
            yield
            return

        if msvcrt and (not lock_path.exists() or lock_path.stat().st_size == 0):
            lock_path.write_text(" ", encoding="utf-8")

        fd = open(lock_path, "r+" if msvcrt else "a+")
        try:
            if fcntl:
                fcntl.flock(fd, fcntl.LOCK_EX)
            else:
                fd.seek(0)
                msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK, 1)
            yield
        finally:
            if fcntl:
                fcntl.flock(fd, fcntl.LOCK_UN)
            elif msvcrt:
                try:
                    fd.seek(0)
                    msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
                except (OSError, IOError):
                    pass
            fd.close()

    @staticmethod
    def _path_for(target: str) -> Path:
        mem_dir = get_memory_dir()
        if target == "user":
            return mem_dir / "USER.md"
        return mem_dir / "MEMORY.md"

    def _reload_target(self, target: str):
        """Re-read entries from disk into in-memory state.

        Called under file lock to get the latest state before mutating.
        """
        fresh = self._read_file(self._path_for(target))
        fresh = list(dict.fromkeys(fresh))  # deduplicate
        self._set_entries(target, fresh)

    def save_to_disk(self, target: str):
        """Persist entries to the appropriate file. Called after every mutation."""
        get_memory_dir().mkdir(parents=True, exist_ok=True)
        self._write_file(self._path_for(target), self._entries_for(target))

    def _entries_for(self, target: str) -> List[str]:
        if target == "user":
            return self.user_entries
        return self.memory_entries

    def _set_entries(self, target: str, entries: List[str]):
        if target == "user":
            self.user_entries = entries
        else:
            self.memory_entries = entries

    def _project_char_count(self, entries: Optional[List[str]] = None) -> int:
        data = self.project_memory_entries if entries is None else entries
        if not data:
            return 0
        return len(ENTRY_DELIMITER.join(data))

    def _empty_project_db(self) -> Dict[str, Any]:
        return {"version": _PROJECT_DB_VERSION, "projects": {}}

    def _read_project_db(self) -> Dict[str, Any]:
        path = get_project_memory_path()
        if not path.exists():
            return self._empty_project_db()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, IOError, json.JSONDecodeError):
            return self._empty_project_db()
        if not isinstance(raw, dict):
            return self._empty_project_db()
        projects = raw.get("projects")
        if not isinstance(projects, dict):
            raw["projects"] = {}
        raw.setdefault("version", _PROJECT_DB_VERSION)
        return raw

    def _write_project_db(self, data: Dict[str, Any]):
        path = get_project_memory_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(path.parent), suffix=".tmp", prefix=".proj_mem_"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
                    f.write("\n")
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, str(path))
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except (OSError, IOError) as e:
            raise RuntimeError(f"Failed to write project memory file {path}: {e}")

    def _prune_expired_project_entries(self, project_db: Dict[str, Any]) -> bool:
        now = _utcnow()
        changed = False
        projects = project_db.get("projects", {})
        for project_key in list(projects.keys()):
            payload = projects.get(project_key, {})
            if not isinstance(payload, dict):
                projects.pop(project_key, None)
                changed = True
                continue
            entries = payload.get("entries", [])
            if not isinstance(entries, list):
                projects[project_key]["entries"] = []
                changed = True
                continue
            fresh_entries = []
            for entry in entries:
                if not isinstance(entry, dict):
                    changed = True
                    continue
                expires_raw = entry.get("expires_at", "")
                expires_at = _parse_timestamp(expires_raw) if expires_raw else None
                if expires_raw and expires_at is None:
                    changed = True
                    continue
                if expires_at and expires_at <= now:
                    changed = True
                    continue
                created_raw = entry.get("created_at", "")
                if created_raw and _parse_timestamp(created_raw) is None:
                    changed = True
                    continue
                content = str(entry.get("content", "")).strip()
                if not content:
                    changed = True
                    continue
                if _scan_memory_content(content):
                    changed = True
                    continue
                fresh_entries.append(entry)
            if fresh_entries:
                payload["entries"] = fresh_entries
            else:
                projects.pop(project_key, None)
                changed = True
        return changed

    def _load_project_entries(self) -> List[str]:
        if not self.current_project_key:
            return []
        path = get_project_memory_path()
        with self._file_lock(path):
            project_db = self._read_project_db()
            changed = self._prune_expired_project_entries(project_db)
            payload = project_db.get("projects", {}).get(self.current_project_key, {})
            entries = payload.get("entries", []) if isinstance(payload, dict) else []
            visible = []
            for entry in entries:
                if isinstance(entry, dict):
                    content = str(entry.get("content", "")).strip()
                    if content:
                        scan_error = _scan_memory_content(content)
                        if scan_error:
                            changed = True
                            continue
                        visible.append(content)
            visible = list(dict.fromkeys(visible))
            if changed:
                self._write_project_db(project_db)
            return visible

    def _project_success_response(self, message: str = None) -> Dict[str, Any]:
        current = self._project_char_count()
        limit = self.project_memory_char_limit
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0
        resp = {
            "success": True,
            "target": "memory",
            "scope": "project",
            "project": self.current_project_label,
            "entries": self.project_memory_entries,
            "usage": f"{pct}% — {current:,}/{limit:,} chars",
            "entry_count": len(self.project_memory_entries),
        }
        if message:
            resp["message"] = message
        return resp

    @staticmethod
    def _find_entry_matches(entries: List[str], old_text: str) -> List[tuple[int, str]]:
        return [(i, e) for i, e in enumerate(entries) if old_text in e]

    def _char_count(self, target: str) -> int:
        entries = self._entries_for(target)
        if not entries:
            return 0
        return len(ENTRY_DELIMITER.join(entries))

    def _char_limit(self, target: str) -> int:
        if target == "user":
            return self.user_char_limit
        return self.memory_char_limit

    def add(
        self,
        target: str,
        content: str,
        scope: str = "global",
        ttl_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Append a new entry. Returns error if it would exceed the char limit."""
        content = content.strip()
        if not content:
            return {"success": False, "error": "Content cannot be empty."}

        # Scan for injection/exfiltration before accepting
        scan_error = _scan_memory_content(content)
        if scan_error:
            return {"success": False, "error": scan_error}

        normalized_scope = _normalize_scope(scope, default="global")
        if normalized_scope is None:
            return {"success": False, "error": "Invalid scope. Use 'global' or 'project'."}
        if target == "user" and normalized_scope == "project":
            return {"success": False, "error": "Project scope is only supported for target='memory'."}

        if normalized_scope == "project":
            if not self.current_project_key:
                return {
                    "success": False,
                    "error": "Project-scoped memory requires an active project context (for example a git repo or project root).",
                }
            project_path = get_project_memory_path()
            with self._file_lock(project_path):
                project_db = self._read_project_db()
                self._prune_expired_project_entries(project_db)
                projects = project_db.setdefault("projects", {})
                payload = projects.setdefault(
                    self.current_project_key,
                    {
                        "entries": [],
                    },
                )
                entries = payload.setdefault("entries", [])
                existing_contents = [
                    str(entry.get("content", "")).strip()
                    for entry in entries
                    if isinstance(entry, dict)
                ]
                if content in existing_contents:
                    self.project_memory_entries = list(dict.fromkeys(existing_contents))
                    return self._project_success_response("Entry already exists (no duplicate added).")

                if ttl_days is None:
                    effective_ttl = self.project_memory_ttl_days
                else:
                    try:
                        effective_ttl = int(ttl_days)
                    except (TypeError, ValueError):
                        return {"success": False, "error": "ttl_days must be a non-negative integer."}
                    if effective_ttl < 0:
                        return {"success": False, "error": "ttl_days must be a non-negative integer."}
                expires_at = None
                if effective_ttl > 0:
                    expires_at = _format_timestamp(_utcnow() + timedelta(days=effective_ttl))

                new_contents = existing_contents + [content]
                new_total = self._project_char_count(new_contents)
                if new_total > self.project_memory_char_limit:
                    current = self._project_char_count(existing_contents)
                    return {
                        "success": False,
                        "error": (
                            f"Project memory at {current:,}/{self.project_memory_char_limit:,} chars. "
                            f"Adding this entry ({len(content)} chars) would exceed the limit. "
                            f"Replace or remove existing project entries first."
                        ),
                        "current_entries": existing_contents,
                        "usage": f"{current:,}/{self.project_memory_char_limit:,}",
                    }

                record = {
                    "content": content,
                    "created_at": _format_timestamp(_utcnow()),
                }
                if expires_at:
                    record["expires_at"] = expires_at
                entries.append(record)
                projects[self.current_project_key] = payload
                self._write_project_db(project_db)
                self.project_memory_entries = list(dict.fromkeys(new_contents))
            return self._project_success_response("Entry added.")

        with self._file_lock(self._path_for(target)):
            # Re-read from disk under lock to pick up writes from other sessions
            self._reload_target(target)

            entries = self._entries_for(target)
            limit = self._char_limit(target)

            # Reject exact duplicates
            if content in entries:
                return self._success_response(target, "Entry already exists (no duplicate added).")

            # Calculate what the new total would be
            new_entries = entries + [content]
            new_total = len(ENTRY_DELIMITER.join(new_entries))

            if new_total > limit:
                current = self._char_count(target)
                return {
                    "success": False,
                    "error": (
                        f"Memory at {current:,}/{limit:,} chars. "
                        f"Adding this entry ({len(content)} chars) would exceed the limit. "
                        f"Replace or remove existing entries first."
                    ),
                    "current_entries": entries,
                    "usage": f"{current:,}/{limit:,}",
                }

            entries.append(content)
            self._set_entries(target, entries)
            self.save_to_disk(target)

        return self._success_response(target, "Entry added.")

    def replace(
        self,
        target: str,
        old_text: str,
        new_content: str,
        scope: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Find entry containing old_text substring, replace it with new_content."""
        old_text = old_text.strip()
        new_content = new_content.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}
        if not new_content:
            return {"success": False, "error": "new_content cannot be empty. Use 'remove' to delete entries."}

        # Scan replacement content for injection/exfiltration
        scan_error = _scan_memory_content(new_content)
        if scan_error:
            return {"success": False, "error": scan_error}

        normalized_scope = _normalize_scope(scope)
        if scope is not None and normalized_scope is None:
            return {"success": False, "error": "Invalid scope. Use 'global' or 'project'."}
        if target == "user" and normalized_scope == "project":
            return {"success": False, "error": "Project scope is only supported for target='memory'."}
        if target == "memory" and normalized_scope == "project":
            if not self.current_project_key:
                return {"success": False, "error": "Project-scoped memory requires an active project context."}
            project_path = get_project_memory_path()
            with self._file_lock(project_path):
                project_db = self._read_project_db()
                self._prune_expired_project_entries(project_db)
                payload = project_db.get("projects", {}).get(self.current_project_key, {})
                entries = payload.get("entries", []) if isinstance(payload, dict) else []
                visible_entries = [str(entry.get("content", "")).strip() for entry in entries if isinstance(entry, dict)]
                matches = self._find_entry_matches(visible_entries, old_text)
                if not matches:
                    return {"success": False, "error": f"No entry matched '{old_text}'."}
                if len(matches) > 1:
                    unique_texts = set(e for _, e in matches)
                    if len(unique_texts) > 1:
                        previews = [e[:80] + ("..." if len(e) > 80 else "") for _, e in matches]
                        return {
                            "success": False,
                            "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                            "matches": previews,
                        }
                idx = matches[0][0]
                test_entries = visible_entries.copy()
                test_entries[idx] = new_content
                new_total = self._project_char_count(test_entries)
                if new_total > self.project_memory_char_limit:
                    return {
                        "success": False,
                        "error": (
                            f"Replacement would put project memory at {new_total:,}/{self.project_memory_char_limit:,} chars. "
                            f"Shorten the new content or remove other entries first."
                        ),
                    }
                entries[idx]["content"] = new_content
                payload["entries"] = entries
                project_db.setdefault("projects", {})[self.current_project_key] = payload
                self._write_project_db(project_db)
                self.project_memory_entries = test_entries
            return self._project_success_response("Entry replaced.")

        with self._file_lock(self._path_for(target)):
            self._reload_target(target)

            entries = self._entries_for(target)
            matches = self._find_entry_matches(entries, old_text)

            if not matches:
                return {"success": False, "error": f"No entry matched '{old_text}'."}

            if len(matches) > 1:
                # If all matches are identical (exact duplicates), operate on the first one
                unique_texts = set(e for _, e in matches)
                if len(unique_texts) > 1:
                    previews = [e[:80] + ("..." if len(e) > 80 else "") for _, e in matches]
                    return {
                        "success": False,
                        "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                        "matches": previews,
                    }
                # All identical -- safe to replace just the first

            idx = matches[0][0]
            limit = self._char_limit(target)

            # Check that replacement doesn't blow the budget
            test_entries = entries.copy()
            test_entries[idx] = new_content
            new_total = len(ENTRY_DELIMITER.join(test_entries))

            if new_total > limit:
                return {
                    "success": False,
                    "error": (
                        f"Replacement would put memory at {new_total:,}/{limit:,} chars. "
                        f"Shorten the new content or remove other entries first."
                    ),
                }

            entries[idx] = new_content
            self._set_entries(target, entries)
            self.save_to_disk(target)

        return self._success_response(target, "Entry replaced.")

    def remove(self, target: str, old_text: str, scope: Optional[str] = None) -> Dict[str, Any]:
        """Remove the entry containing old_text substring."""
        old_text = old_text.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}

        normalized_scope = _normalize_scope(scope)
        if scope is not None and normalized_scope is None:
            return {"success": False, "error": "Invalid scope. Use 'global' or 'project'."}
        if target == "user" and normalized_scope == "project":
            return {"success": False, "error": "Project scope is only supported for target='memory'."}
        if target == "memory" and normalized_scope == "project":
            if not self.current_project_key:
                return {"success": False, "error": "Project-scoped memory requires an active project context."}
            project_path = get_project_memory_path()
            with self._file_lock(project_path):
                project_db = self._read_project_db()
                self._prune_expired_project_entries(project_db)
                payload = project_db.get("projects", {}).get(self.current_project_key, {})
                entries = payload.get("entries", []) if isinstance(payload, dict) else []
                visible_entries = [str(entry.get("content", "")).strip() for entry in entries if isinstance(entry, dict)]
                matches = self._find_entry_matches(visible_entries, old_text)
                if not matches:
                    return {"success": False, "error": f"No entry matched '{old_text}'."}
                if len(matches) > 1:
                    unique_texts = set(e for _, e in matches)
                    if len(unique_texts) > 1:
                        previews = [e[:80] + ("..." if len(e) > 80 else "") for _, e in matches]
                        return {
                            "success": False,
                            "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                            "matches": previews,
                        }
                idx = matches[0][0]
                entries.pop(idx)
                if entries:
                    payload["entries"] = entries
                    project_db.setdefault("projects", {})[self.current_project_key] = payload
                else:
                    project_db.setdefault("projects", {}).pop(self.current_project_key, None)
                self._write_project_db(project_db)
                self.project_memory_entries = [
                    str(entry.get("content", "")).strip()
                    for entry in entries
                    if isinstance(entry, dict) and str(entry.get("content", "")).strip()
                ]
            return self._project_success_response("Entry removed.")

        with self._file_lock(self._path_for(target)):
            self._reload_target(target)

            entries = self._entries_for(target)
            matches = self._find_entry_matches(entries, old_text)

            if not matches:
                return {"success": False, "error": f"No entry matched '{old_text}'."}

            if len(matches) > 1:
                # If all matches are identical (exact duplicates), remove the first one
                unique_texts = set(e for _, e in matches)
                if len(unique_texts) > 1:
                    previews = [e[:80] + ("..." if len(e) > 80 else "") for _, e in matches]
                    return {
                        "success": False,
                        "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                        "matches": previews,
                    }
                # All identical -- safe to remove just the first

            idx = matches[0][0]
            entries.pop(idx)
            self._set_entries(target, entries)
            self.save_to_disk(target)

        return self._success_response(target, "Entry removed.")

    def format_for_system_prompt(self, target: str) -> Optional[str]:
        """
        Return the frozen snapshot for system prompt injection.

        This returns the state captured at load_from_disk() time, NOT the live
        state. Mid-session writes do not affect this. This keeps the system
        prompt stable across all turns, preserving the prefix cache.

        Returns None if the snapshot is empty (no entries at load time).
        """
        block = self._system_prompt_snapshot.get(target, "")
        return block if block else None

    # -- Internal helpers --

    def _success_response(self, target: str, message: str = None) -> Dict[str, Any]:
        entries = self._entries_for(target)
        current = self._char_count(target)
        limit = self._char_limit(target)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0

        resp = {
            "success": True,
            "target": target,
            "entries": entries,
            "usage": f"{pct}% — {current:,}/{limit:,} chars",
            "entry_count": len(entries),
        }
        if message:
            resp["message"] = message
        return resp

    def _render_block(self, target: str, entries: List[str]) -> str:
        """Render a system prompt block with header and usage indicator."""
        if not entries:
            return ""

        limit = self._char_limit(target)
        content = ENTRY_DELIMITER.join(entries)
        current = len(content)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0

        if target == "user":
            header = f"USER PROFILE (who the user is) [{pct}% — {current:,}/{limit:,} chars]"
        else:
            header = f"MEMORY (your personal notes) [{pct}% — {current:,}/{limit:,} chars]"

        separator = "═" * 46
        return f"{separator}\n{header}\n{separator}\n{content}"

    def _render_memory_block(self) -> str:
        """Render global memory plus any current-project scoped memory."""
        if not self.memory_entries and not self.project_memory_entries:
            return ""

        global_content = ENTRY_DELIMITER.join(self.memory_entries) if self.memory_entries else ""
        global_current = len(global_content)
        global_pct = min(100, int((global_current / self.memory_char_limit) * 100)) if self.memory_char_limit > 0 else 0

        header = f"MEMORY (your personal notes) [global {global_pct}% — {global_current:,}/{self.memory_char_limit:,} chars"
        sections = []
        if global_content:
            sections.append(global_content)

        if self.project_memory_entries:
            project_content = ENTRY_DELIMITER.join(self.project_memory_entries)
            project_current = len(project_content)
            project_pct = min(100, int((project_current / self.project_memory_char_limit) * 100)) if self.project_memory_char_limit > 0 else 0
            header += f" | project {project_pct}% — {project_current:,}/{self.project_memory_char_limit:,} chars"
            sections.append(f"[PROJECT MEMORY]\n{project_content}")

        header += "]"
        separator = "═" * 46
        return f"{separator}\n{header}\n{separator}\n{ENTRY_DELIMITER.join(sections)}"

    @staticmethod
    def _read_file(path: Path) -> List[str]:
        """Read a memory file and split into entries.

        No file locking needed: _write_file uses atomic rename, so readers
        always see either the previous complete file or the new complete file.
        """
        if not path.exists():
            return []
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, IOError):
            return []

        if not raw.strip():
            return []

        # Use ENTRY_DELIMITER for consistency with _write_file. Splitting by "§"
        # alone would incorrectly split entries that contain "§" in their content.
        entries = [e.strip() for e in raw.split(ENTRY_DELIMITER)]
        return [e for e in entries if e]

    @staticmethod
    def _write_file(path: Path, entries: List[str]):
        """Write entries to a memory file using atomic temp-file + rename.

        Previous implementation used open("w") + flock, but "w" truncates the
        file *before* the lock is acquired, creating a race window where
        concurrent readers see an empty file. Atomic rename avoids this:
        readers always see either the old complete file or the new one.
        """
        content = ENTRY_DELIMITER.join(entries) if entries else ""
        try:
            # Write to temp file in same directory (same filesystem for atomic rename)
            fd, tmp_path = tempfile.mkstemp(
                dir=str(path.parent), suffix=".tmp", prefix=".mem_"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, str(path))  # Atomic on same filesystem
            except BaseException:
                # Clean up temp file on any failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except (OSError, IOError) as e:
            raise RuntimeError(f"Failed to write memory file {path}: {e}")


def memory_tool(
    action: str,
    target: str = "memory",
    content: str = None,
    old_text: str = None,
    scope: str = None,
    ttl_days: int = None,
    store: Optional[MemoryStore] = None,
    working_dir: Optional[os.PathLike | str] = None,
) -> str:
    """
    Single entry point for the memory tool. Dispatches to MemoryStore methods.

    Returns JSON string with results.
    """
    if store is None:
        return tool_error("Memory is not available. It may be disabled in config or this environment.", success=False)

    if working_dir is not None:
        try:
            store.refresh_project_context(working_dir)
        except Exception as exc:
            return tool_error(f"Failed to resolve project scope from working_dir: {exc}", success=False)

    if target not in ("memory", "user"):
        return tool_error(f"Invalid target '{target}'. Use 'memory' or 'user'.", success=False)

    if action == "add":
        if not content:
            return tool_error("Content is required for 'add' action.", success=False)
        result = store.add(target, content, scope=scope or "global", ttl_days=ttl_days)

    elif action == "replace":
        if not old_text:
            return tool_error("old_text is required for 'replace' action.", success=False)
        if not content:
            return tool_error("content is required for 'replace' action.", success=False)
        result = store.replace(target, old_text, content, scope=scope)

    elif action == "remove":
        if not old_text:
            return tool_error("old_text is required for 'remove' action.", success=False)
        result = store.remove(target, old_text, scope=scope)

    else:
        return tool_error(f"Unknown action '{action}'. Use: add, replace, remove", success=False)

    return json.dumps(result, ensure_ascii=False)


def check_memory_requirements() -> bool:
    """Memory tool has no external requirements -- always available."""
    return True


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================

MEMORY_SCHEMA = {
    "name": "memory",
    "description": (
        "Save durable information to persistent memory that survives across sessions. "
        "Memory is injected into future turns, so keep it compact and focused on facts "
        "that will still matter later.\n\n"
        "WHEN TO SAVE (do this proactively, don't wait to be asked):\n"
        "- User corrects you or says 'remember this' / 'don't do that again'\n"
        "- User shares a preference, habit, or personal detail (name, role, timezone, coding style)\n"
        "- You discover something about the environment (OS, installed tools, project structure)\n"
        "- You learn a convention, API quirk, or workflow specific to this user's setup\n"
        "- You identify a stable fact that will be useful again in future sessions\n\n"
        "PRIORITY: User preferences and corrections > environment facts > procedural knowledge. "
        "The most valuable memory prevents the user from having to repeat themselves.\n\n"
        "Do NOT save task progress, session outcomes, completed-work logs, or temporary TODO "
        "state to memory; use session_search to recall those from past transcripts.\n"
        "If you've discovered a new way to do something, solved a problem that could be "
        "necessary later, save it as a skill with the skill tool.\n\n"
        "TWO TARGETS:\n"
        "- 'user': who the user is -- name, role, preferences, communication style, pet peeves\n"
        "- 'memory': your notes -- environment facts, project conventions, tool quirks, lessons learned\n\n"
        "SCOPES:\n"
        "- default/global: use for broad facts that should apply in all future sessions\n"
        "- project: use for repo- or project-specific details (paths, architecture choices, temporary conventions); these are only injected when Hermes is back in the same project\n"
        "- project memories can also expire via ttl_days to prevent stale context creep\n\n"
        "ACTIONS: add (new entry), replace (update existing -- old_text identifies it), "
        "remove (delete -- old_text identifies it).\n\n"
        "SKIP: trivial/obvious info, things easily re-discovered, raw data dumps, and temporary task state."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "replace", "remove"],
                "description": "The action to perform."
            },
            "target": {
                "type": "string",
                "enum": ["memory", "user"],
                "description": "Which memory store: 'memory' for personal notes, 'user' for user profile."
            },
            "content": {
                "type": "string",
                "description": "The entry content. Required for 'add' and 'replace'."
            },
            "old_text": {
                "type": "string",
                "description": "Short unique substring identifying the entry to replace or remove."
            },
            "scope": {
                "type": "string",
                "enum": ["global", "project"],
                "description": "Optional scope for memory target. Use 'project' for repo-specific memories that should only load in the same project."
            },
            "ttl_days": {
                "type": "integer",
                "description": "Optional expiry for project-scoped memories. After this many days, the memory stops being injected."
            },
        },
        "required": ["action", "target"],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="memory",
    toolset="memory",
    schema=MEMORY_SCHEMA,
    handler=lambda args, **kw: memory_tool(
        action=args.get("action", ""),
        target=args.get("target", "memory"),
        content=args.get("content"),
        old_text=args.get("old_text"),
        scope=args.get("scope"),
        ttl_days=args.get("ttl_days"),
        store=kw.get("store")),
    check_fn=check_memory_requirements,
    emoji="🧠",
)



