"""Serve a local video preview with HTTP byte ranges for exact video seeking."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import re


class RangeHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        requested = self.headers.get("Range")
        if not requested:
            return super().send_head()
        from pathlib import Path

        path = Path(self.translate_path(self.path))
        if not path.is_file():
            return super().send_head()
        size = path.stat().st_size
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
        if not match or not any(match.groups()):
            self.send_error(416, "Unsupported byte range")
            return None
        first, last = match.groups()
        start = int(first) if first else max(0, size - int(last))
        end = min(size - 1, int(last)) if first and last else size - 1
        if start > end or start >= size:
            self.send_error(416, "Byte range outside file")
            return None
        stream = path.open("rb")
        stream.seek(start)
        self.remaining = end - start + 1
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Length", str(self.remaining))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        return stream

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def copyfile(self, source, output):
        remaining = getattr(self, "remaining", None)
        if remaining is None:
            return super().copyfile(source, output)
        while remaining:
            chunk = source.read(min(65536, remaining))
            if not chunk:
                raise EOFError("file ended before byte range")
            output.write(chunk)
            remaining -= len(chunk)
        self.remaining = None


if __name__ == "__main__":
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--port", type=int, default=38765)
    args = parser.parse_args()
    ThreadingHTTPServer(
        ("127.0.0.1", args.port), partial(RangeHandler, directory=args.directory)
    ).serve_forever()
