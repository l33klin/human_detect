"""Restartable stream definitions. Sources contain secrets; never expose raw rows in APIs."""

import json
import os
import sqlite3


class StreamStore:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # Set permissions before SQLite creates any journal containing credentials.
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(descriptor)
        os.chmod(path, 0o600)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS streams "
            "(id TEXT PRIMARY KEY, definition TEXT NOT NULL, enabled INTEGER NOT NULL)"
        )

    def list(self):
        return [
            (key, json.loads(definition), bool(enabled))
            for key, definition, enabled in self.connection.execute("SELECT * FROM streams")
        ]

    def save(self, key, definition, enabled):
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO streams VALUES (?, ?, ?)",
                (key, json.dumps(definition), int(enabled)),
            )

    def delete(self, key):
        with self.connection:
            self.connection.execute("DELETE FROM streams WHERE id=?", (key,))

    def close(self):
        self.connection.close()
