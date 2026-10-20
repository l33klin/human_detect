import json
import sqlite3
import threading
from pathlib import Path

import numpy as np

from human_detect.schemas import Identity


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.all(np.isfinite(vector)) or norm < 1e-8:
        raise ValueError("invalid face embedding")
    return vector / norm


class IdentityStore:
    """Stores normalized embeddings only, bound to one model/preprocessing fingerprint."""

    def __init__(self, path: Path, fingerprint: str, dimension: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.revision = 0
        self.connection = sqlite3.connect(str(path), check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS people (id TEXT PRIMARY KEY, name TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY, person_id TEXT NOT NULL, embedding TEXT NOT NULL
            );
        """)
        self.dimension = dimension
        signature = f"{fingerprint}:{dimension}"
        previous = self.connection.execute(
            "SELECT value FROM metadata WHERE key='embedding_signature'"
        ).fetchone()
        if previous and previous[0] != signature:
            self.connection.close()
            raise RuntimeError("identity database model changed; use a new database and re-enroll")
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO metadata VALUES ('embedding_signature', ?)", (signature,)
            )

    def enroll(self, person_id, name, vector):
        vector = normalize(vector)
        if len(vector) != self.dimension:
            raise ValueError("embedding dimension mismatch")
        with self.lock, self.connection:
            self.connection.execute(
                "INSERT INTO people VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET name=excluded.name",
                (person_id, name),
            )
            self.connection.execute(
                "INSERT INTO samples(person_id, embedding) VALUES (?, ?)",
                (person_id, json.dumps(vector.tolist())),
            )
            self.revision += 1
        return next(row for row in self.list_people() if row["person_id"] == person_id)

    def list_people(self):
        with self.lock:
            rows = self.connection.execute(
                "SELECT p.id, p.name, COUNT(s.id) FROM people p "
                "LEFT JOIN samples s ON s.person_id=p.id GROUP BY p.id ORDER BY p.id"
            ).fetchall()
        return [dict(person_id=i, name=n, samples=c) for i, n, c in rows]

    def delete(self, person_id):
        with self.lock, self.connection:
            self.connection.execute("DELETE FROM samples WHERE person_id=?", (person_id,))
            deleted = (
                self.connection.execute("DELETE FROM people WHERE id=?", (person_id,)).rowcount > 0
            )
            if deleted:
                self.revision += 1
            return deleted

    def match(self, vector, threshold, margin):
        vector = normalize(vector)
        with self.lock:
            rows = self.connection.execute(
                "SELECT p.id, p.name, s.embedding FROM samples s JOIN people p ON p.id=s.person_id"
            ).fetchall()
        if not rows:
            return Identity()
        # Best sample per identity; the runner-up must be a different identity.
        scores = {}
        for person_id, name, raw in rows:
            score = float(np.clip(np.dot(vector, np.asarray(json.loads(raw))), -1, 1))
            if person_id not in scores or score > scores[person_id][0]:
                scores[person_id] = (score, name)
        ranked = sorted(scores.items(), key=lambda item: item[1][0], reverse=True)
        person_id, (score, name) = ranked[0]
        ambiguous = len(ranked) > 1 and score - ranked[1][1][0] < margin
        if score < threshold or ambiguous:
            return Identity(similarity=round(score, 5))
        return Identity(
            status="matched", person_id=person_id, name=name, similarity=round(score, 5)
        )

    def close(self):
        with self.lock:
            self.connection.close()
