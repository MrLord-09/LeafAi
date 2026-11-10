"""SQLite helpers for disease guidance and private scan history."""

from __future__ import annotations

from contextlib import contextmanager
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator


@contextmanager
def connect(db_path: str | Path) -> Generator[sqlite3.Connection, None, None]:
    connection = sqlite3.connect(db_path, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database(
    db_path: str | Path,
    class_names: list[str],
    catalog: dict[str, dict[str, str]],
) -> None:
    with connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS plant_diseases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                class_name TEXT UNIQUE NOT NULL,
                disease_name TEXT NOT NULL,
                symptoms TEXT NOT NULL,
                organic_control TEXT NOT NULL,
                chemical_control TEXT NOT NULL,
                source TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS scan_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scanned_at TEXT NOT NULL,
                class_name TEXT NOT NULL,
                disease_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                original_filename TEXT NOT NULL,
                content_type TEXT NOT NULL,
                image_name TEXT NOT NULL UNIQUE
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ip_camera_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_id INTEGER NOT NULL REFERENCES scan_history(id) ON DELETE CASCADE,
                captured_at TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                content_type TEXT NOT NULL,
                class_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                image_data BLOB NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_scan_history_date ON scan_history(scanned_at)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_scan_history_class ON scan_history(class_name)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_ip_camera_snapshots_scan ON ip_camera_snapshots(scan_id)"
        )
        for class_name in class_names:
            record = catalog.get(class_name) or {}
            display_name = record.get(
                "disease_name",
                class_name.replace("___", " — ").replace("_", " "),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO plant_diseases
                    (class_name, disease_name, symptoms, organic_control, chemical_control, source)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    class_name,
                    display_name,
                    record.get(
                        "symptoms",
                        "Visual classification only; symptoms are not yet curated. Confirm the diagnosis with a local agricultural expert.",
                    ),
                    record.get(
                        "organic_control",
                        "Use sanitation, crop-specific cultural practices and integrated pest management after confirming the cause.",
                    ),
                    record.get(
                        "chemical_control",
                        "Do not choose a pesticide from this label alone. Confirm the target and use only a product currently registered in India for this crop and disease, strictly following its label.",
                    ),
                    record.get("source", "Local agricultural extension; confirm current Indian label"),
                ),
            )
            if record:
                connection.execute(
                    """
                    UPDATE plant_diseases
                    SET disease_name = ?, symptoms = ?, organic_control = ?, chemical_control = ?, source = ?
                    WHERE class_name = ? AND source IN (?, ?, ?)
                    """,
                    (
                        record["disease_name"],
                        record["symptoms"],
                        record["organic_control"],
                        record["chemical_control"],
                        record["source"],
                        class_name,
                        "Plant disease IPM guidance; verify current local label",
                        "General crop health/IPM guidance",
                        "Local agricultural extension; confirm current Indian label",
                    ),
                )


def _row_to_guide(row: sqlite3.Row) -> dict[str, str]:
    return {
        "class_name": row["class_name"],
        "disease_name": row["disease_name"],
        "symptoms": row["symptoms"],
        "organic_control": row["organic_control"],
        "chemical_control": row["chemical_control"],
        "source": row["source"],
    }


def get_remedy_data(db_path: str | Path, class_name: str) -> dict[str, str] | None:
    with connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT class_name, disease_name, symptoms, organic_control, chemical_control, source
            FROM plant_diseases
            WHERE class_name = ?
            """,
            (class_name,),
        ).fetchone()
    return _row_to_guide(row) if row else None


def list_remedies(db_path: str | Path, class_names: list[str]) -> list[dict[str, str]]:
    with connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT class_name, disease_name, symptoms, organic_control, chemical_control, source
            FROM plant_diseases
            ORDER BY disease_name COLLATE NOCASE
            """
        ).fetchall()
    by_class = {row["class_name"]: _row_to_guide(row) for row in rows}
    return [by_class[name] for name in class_names if name in by_class]


def update_remedy_data(db_path: str | Path, class_name: str, values: dict[str, str]) -> None:
    with connect(db_path) as connection:
        cursor = connection.execute(
            """
            UPDATE plant_diseases
            SET disease_name = ?, symptoms = ?, organic_control = ?, chemical_control = ?, source = ?
            WHERE class_name = ?
            """,
            (
                values["disease_name"].strip(),
                values["symptoms"].strip(),
                values["organic_control"].strip(),
                values["chemical_control"].strip(),
                values["source"].strip(),
                class_name,
            ),
        )
        if cursor.rowcount != 1:
            raise KeyError(class_name)


def save_scan(
    db_path: str | Path,
    image_dir: str | Path,
    image_bytes: bytes,
    content_type: str,
    filename: str,
    class_name: str,
    disease_name: str,
    confidence: float,
    snapshots: list[dict[str, Any]] | None = None,
) -> int:
    extension = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }.get(content_type)
    if extension is None:
        raise ValueError(f"Unsupported image content type: {content_type}")
    safe_filename = re.sub(r"[\x00-\x1f\x7f]", "", Path(filename).name).strip()[:255]
    if not safe_filename:
        safe_filename = f"scan{extension}"

    image_root = Path(image_dir)
    image_root.mkdir(parents=True, exist_ok=True)
    image_name = f"{uuid.uuid4().hex}{extension}"
    image_path = image_root / image_name
    image_path.write_bytes(image_bytes)
    try:
        with connect(db_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO scan_history
                    (scanned_at, class_name, disease_name, confidence, original_filename, content_type, image_name)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    class_name,
                    disease_name,
                    confidence,
                    safe_filename,
                    content_type,
                    image_name,
                ),
            )
            scan_id = int(cursor.lastrowid)
            if snapshots:
                captured_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                connection.executemany(
                    """
                    INSERT INTO ip_camera_snapshots
                        (scan_id, captured_at, original_filename, content_type, class_name, confidence, image_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            scan_id,
                            captured_at,
                            snapshot["filename"],
                            snapshot["content_type"],
                            snapshot["class_name"],
                            snapshot["confidence"],
                            sqlite3.Binary(snapshot["bytes"]),
                        )
                        for snapshot in snapshots
                    ],
                )
            return scan_id
    except Exception:
        image_path.unlink(missing_ok=True)
        raise


def get_scan_history(
    db_path: str | Path,
    query: str,
    class_name: str | None,
    limit: int,
    offset: int,
) -> tuple[list[dict[str, Any]], int]:
    conditions: list[str] = []
    parameters: list[Any] = []
    if query:
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        conditions.append(
            "(disease_name LIKE ? ESCAPE '\\' OR class_name LIKE ? ESCAPE '\\' OR original_filename LIKE ? ESCAPE '\\')"
        )
        parameters.extend([f"%{escaped}%"] * 3)
    if class_name:
        conditions.append("class_name = ?")
        parameters.append(class_name)
    where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""

    with connect(db_path) as connection:
        total = connection.execute(
            f"SELECT COUNT(*) FROM scan_history{where_clause}",
            parameters,
        ).fetchone()[0]
        rows = connection.execute(
            f"""
            SELECT id, scanned_at, class_name, disease_name, confidence, original_filename, content_type
            FROM scan_history{where_clause}
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            [*parameters, limit, offset],
        ).fetchall()
    return ([dict(row) for row in rows], int(total))


def get_scan_image(db_path: str | Path, scan_id: int) -> tuple[str, str] | None:
    with connect(db_path) as connection:
        row = connection.execute(
            "SELECT image_name, content_type FROM scan_history WHERE id = ?",
            (scan_id,),
        ).fetchone()
    return (row["image_name"], row["content_type"]) if row else None


def delete_scan(db_path: str | Path, image_dir: str | Path, scan_id: int) -> bool:
    image = get_scan_image(db_path, scan_id)
    if image is None:
        return False
    image_root = Path(image_dir).resolve()
    image_path = (image_root / image[0]).resolve()
    if image_path.parent != image_root:
        raise ValueError("Invalid stored image path.")
    with connect(db_path) as connection:
        connection.execute("DELETE FROM scan_history WHERE id = ?", (scan_id,))
    image_path.unlink(missing_ok=True)
    return True


def delete_all_scans(db_path: str | Path, image_dir: str | Path) -> int:
    image_root = Path(image_dir)
    with connect(db_path) as connection:
        names = [
            row["image_name"]
            for row in connection.execute("SELECT image_name FROM scan_history").fetchall()
        ]
        connection.execute("DELETE FROM scan_history")
    for name in names:
        image_path = (image_root.resolve() / name).resolve()
        if image_path.parent != image_root.resolve():
            raise ValueError("Invalid stored image path.")
        image_path.unlink(missing_ok=True)
    return len(names)


def history_summary(db_path: str | Path) -> dict[str, int]:
    with connect(db_path) as connection:
        total = connection.execute("SELECT COUNT(*) FROM scan_history").fetchone()[0]
        diseases = connection.execute(
            "SELECT COUNT(DISTINCT class_name) FROM scan_history"
        ).fetchone()[0]
        today = datetime.now(timezone.utc).date().isoformat()
        today_count = connection.execute(
            "SELECT COUNT(*) FROM scan_history WHERE substr(scanned_at, 1, 10) = ?",
            (today,),
        ).fetchone()[0]
    return {"total_scans": int(total), "diseases_seen": int(diseases), "scans_today": int(today_count)}
