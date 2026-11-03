import argparse
import hashlib
import json
import sqlite3
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


class History:
    def __init__(self, root: Path, cache: Path):
        self.root = root.resolve()
        self.cache = cache
        cache.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cache / "history.sqlite", check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS records (run TEXT, offset INTEGER, body TEXT, PRIMARY KEY(run,offset))"
        )
        if "active" not in [r[1] for r in self.db.execute("PRAGMA table_info(records)")]:
            self.db.execute("ALTER TABLE records ADD COLUMN active INTEGER DEFAULT 1")
        self.db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS event_identity ON records(run,json_extract(body, '$.event_id')) WHERE json_extract(body, '$.event_id') IS NOT NULL"
        )
        self.db.execute("CREATE TABLE IF NOT EXISTS cursors (run TEXT PRIMARY KEY, offset INTEGER)")
        if "identity" not in [r[1] for r in self.db.execute("PRAGMA table_info(cursors)")]:
            self.db.execute("ALTER TABLE cursors ADD COLUMN identity TEXT")
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS event_rows ON records(run,offset) WHERE json_extract(body,'$.event') IS NOT NULL AND json_extract(body,'$.event') != 'step_completed'"
        )
        self.positions: dict[str, int] = dict(self.db.execute("SELECT run,offset FROM cursors"))
        self.identities = dict(self.db.execute("SELECT run,identity FROM cursors"))
        self.views: dict[str, dict] = {}
        for snapshot in list(cache.glob("*.view.json"))[:8]:
            try:
                value = json.loads(snapshot.read_text())
                self.views[value["id"]] = value
            except (ValueError, KeyError):
                pass
        self.lock = threading.Lock()
        self.corrupt: dict[str, int] = {}

    def runs(self) -> dict[str, Path]:
        found = {}
        for name in ("events.jsonl", "metrics.jsonl"):
            for p in self.root.rglob(name):
                if p.is_symlink() or self.root not in p.resolve().parents:
                    continue
                if name == "metrics.jsonl" and (p.parent / "events.jsonl").exists():
                    continue
                key = hashlib.sha256(str(p.relative_to(self.root)).encode()).hexdigest()[:16]
                found[key] = p
        return found

    def update(self, key: str, p: Path) -> None:
        with self.lock:
            offset = self.positions.get(key, 0)
            stat = p.stat()
            identity = f"{stat.st_dev}:{stat.st_ino}"
            if stat.st_size < offset or self.identities.get(key) != identity:
                self.db.execute("DELETE FROM records WHERE run=?", (key,))
                self.views.pop(key, None)
                offset = 0
            with p.open("rb") as f:
                f.seek(offset)
                chunk = f.read(1048576)
            end = chunk.rfind(b"\n") + 1
            for line in chunk[:end].splitlines(keepends=True):
                try:
                    row = json.loads(line, parse_constant=lambda _: None)
                    if not isinstance(row, dict):
                        raise TypeError("record")
                    if not isinstance(row.get("step"), int) or row["step"] < 0:
                        raise ValueError("Invalid step")
                    if row.get("event") is not None and (
                        row.get("schema_version") != 1 or not isinstance(row.get("data"), dict)
                    ):
                        raise ValueError("Unsupported event schema")
                    if row.get("event") == "started":
                        self.db.execute(
                            "UPDATE records SET active=0 WHERE run=? AND json_extract(body,'$.step')>?",
                            (key, row.get("step", 0)),
                        )
                    self.db.execute(
                        "INSERT OR REPLACE INTO records(run,offset,body) VALUES(?,?,?)",
                        (key, offset, json.dumps(row)),
                    )
                except (ValueError, UnicodeError, TypeError):
                    self.corrupt[key] = self.corrupt.get(key, 0) + 1
                offset += len(line)
            if not end and len(chunk) == 1048576:
                offset += len(chunk)
                self.corrupt[key] = self.corrupt.get(key, 0) + 1
            self.db.execute("INSERT OR REPLACE INTO cursors VALUES(?,?,?)", (key, offset, identity))
            self.identities[key] = identity
            self.positions[key] = offset
            self.db.commit()

    def view(self, key: str, p: Path) -> dict:
        p = p.resolve()
        self.update(key, p)
        if key in self.views and self.views[key].get("indexed_bytes") == self.positions[key]:
            return self.views[key]
        with self.lock:
            count = self.db.execute("SELECT count(*) FROM records WHERE run=?", (key,)).fetchone()[
                0
            ]
            stride = max(1, count // 2000)
            rows = [
                json.loads(r[0])
                for r in self.db.execute(
                    "SELECT body FROM records WHERE active=1 AND run=? AND rowid%?=0 ORDER BY offset LIMIT 2500",
                    (key, stride),
                )
            ]
            extras = [
                json.loads(r[0])
                for r in self.db.execute(
                    "SELECT body FROM records WHERE run=? AND active=1 AND json_extract(body,'$.event') IS NOT NULL AND json_extract(body,'$.event') != 'step_completed' ORDER BY offset DESC LIMIT 1000",
                    (key,),
                )
            ]
            metric_tail = self.db.execute(
                "SELECT body FROM records WHERE run=? AND active=1 AND (json_extract(body,'$.event')='step_completed' OR json_extract(body,'$.loss') IS NOT NULL) ORDER BY offset DESC LIMIT 1",
                (key,),
            ).fetchone()
            if metric_tail:
                rows.append(json.loads(metric_tail[0]))
            known = {(r.get("session_id"), r.get("seq")) for r in rows if r.get("event")}
            rows.extend(
                r for r in reversed(extras) if (r.get("session_id"), r.get("seq")) not in known
            )
            rows.sort(key=lambda r: (r.get("timestamp", ""), r.get("step", 0)))
            last = self.db.execute(
                "SELECT body FROM records WHERE run=? ORDER BY offset DESC LIMIT 1", (key,)
            ).fetchone()
        metadata = {}
        for name in ("run.json", "summary.json"):
            f = p.parent / name
            if f.exists() and not f.is_symlink():
                try:
                    metadata.update(json.loads(f.read_text()))
                except ValueError:
                    pass
        result = {
            "id": key,
            "name": str(p.parent.relative_to(self.root)),
            "rows": rows,
            "last": json.loads(last[0]) if last else None,
            "metadata": metadata,
            "records": count,
            "indexed_bytes": self.positions[key],
            "corrupt": self.corrupt.get(key, 0),
            "indexing": self.positions[key] < p.stat().st_size,
            "legacy": p.name == "metrics.jsonl",
        }
        if len(self.views) > 8:
            self.views.pop(next(iter(self.views)))
        self.views[key] = result
        if not result["indexing"]:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=self.cache, delete=False
            ) as snapshot_stream:
                json.dump(result, snapshot_stream)
                temporary = Path(snapshot_stream.name)
            temporary.replace(self.cache / (key + ".view.json"))
        return result


class DashboardServer(ThreadingHTTPServer):
    def __init__(self, *args, **kwargs):
        self.stopping = threading.Event()
        super().__init__(*args, **kwargs)

    def shutdown(self):
        self.stopping.set()
        super().shutdown()

    def server_close(self):
        self.stopping.set()
        super().server_close()

    daemon_threads = True
    clients = threading.BoundedSemaphore(16)


def serve(root: Path, port: int = 8765) -> DashboardServer:
    history = History(
        root,
        Path("outputs/dashboard-cache")
        / hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:12],
    )
    assets = Path(__file__).parent / "static"

    class Handler(BaseHTTPRequestHandler):
        server: DashboardServer

        def log_message(self, *args):
            pass

        def do_GET(self):
            host = self.headers.get("Host", "").split(":")[0]
            if host not in ("127.0.0.1", "localhost"):
                self.send_error(403)
                return
            origin = self.headers.get("Origin")
            if origin and origin not in (
                f"http://127.0.0.1:{self.server.server_port}",
                f"http://localhost:{self.server.server_port}",
            ):
                self.send_error(403)
                return
            u = urlparse(self.path)
            runs = history.runs()
            if u.path == "/api/context":
                evidence = Path("outputs/benchmarks/m25-context.json")
                data = (
                    evidence.read_bytes()
                    if evidence.is_file() and not evidence.is_symlink()
                    else b"[]"
                )
                mime = "application/json"
            elif u.path == "/api/runs":
                data = json.dumps(
                    [
                        {"id": k, "name": str(p.parent.relative_to(history.root))}
                        for k, p in runs.items()
                    ]
                ).encode()
                mime = "application/json"
            elif u.path in ("/api/run", "/api/events"):
                key = parse_qs(u.query).get("id", [""])[0]
                if key not in runs:
                    self.send_error(404)
                    return
                if u.path == "/api/events":
                    if not self.server.clients.acquire(blocking=False):
                        self.send_error(503)
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.connection.settimeout(2)
                    try:
                        for _ in range(60):
                            if self.server.stopping.is_set():
                                break
                            self.wfile.write(
                                (
                                    "data: " + json.dumps(history.view(key, runs[key])) + "\n\n"
                                ).encode()
                            )
                            self.wfile.flush()
                            time.sleep(1)
                    except (OSError, TimeoutError):
                        pass
                    finally:
                        self.server.clients.release()
                    return
                data = json.dumps(history.view(key, runs[key])).encode()
                mime = "application/json"
            else:
                name = {
                    "/": "index.html",
                    "/app.js": "app.js",
                    "/style.css": "style.css",
                    "/uplot.js": "uplot.js",
                    "/uplot.css": "uplot.css",
                }.get(u.path)
                if not name:
                    self.send_error(404)
                    return
                data = (assets / name).read_bytes()
                mime = (
                    "text/html"
                    if name.endswith("html")
                    else "text/css"
                    if name.endswith("css")
                    else "text/javascript"
                )
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(data)

    return DashboardServer(("127.0.0.1", port), Handler)


def main(argv=None):
    p = argparse.ArgumentParser(description="Read-only local Codexa telemetry dashboard")
    p.add_argument("--root", type=Path, default=Path("outputs/experiments"))
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--run", help="Initial opaque run ID")
    p.add_argument("--list", action="store_true")
    a = p.parse_args(argv)
    if a.list:
        h = History(a.root, Path("outputs/dashboard-list-cache"))
        print(
            json.dumps(
                [{"id": k, "name": str(v.parent.relative_to(h.root))} for k, v in h.runs().items()],
                indent=2,
            )
        )
        return
    server = serve(a.root, a.port)
    print(
        f"http://127.0.0.1:{server.server_port}/" + (f"?run={a.run}" if a.run else ""), flush=True
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
