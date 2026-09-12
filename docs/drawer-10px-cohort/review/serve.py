"""Local-only review server with video byte-range seeking."""
import argparse
import os
from pathlib import Path
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

class Handler(SimpleHTTPRequestHandler):
    def send_head(self):
        self.remaining = None
        requested = self.headers.get('Range')
        path = Path(self.translate_path(self.path))
        if not requested or not path.is_file():
            return super().send_head()
        import re
        match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested)
        size = path.stat().st_size
        if match is None or not any(match.groups()):
            self.send_error(416, 'Invalid byte range')
            return None
        first, last = match.groups()
        start = int(first) if first else max(0, size - int(last))
        end = min(size - 1, int(last)) if first and last else size - 1
        if start > end or start >= size:
            self.send_response(416)
            self.send_header('Content-Range', f'bytes */{size}')
            self.end_headers()
            return None
        stream = path.open('rb')
        stream.seek(start)
        self.remaining = end - start + 1
        self.send_response(206)
        self.send_header('Content-Type', self.guess_type(str(path)))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length', str(self.remaining))
        self.end_headers()
        return stream

    def copyfile(self, source, output):
        if self.remaining is None:
            return super().copyfile(source, output)
        while self.remaining:
            data = source.read(min(self.remaining, 64 * 1024))
            if not data:
                break
            output.write(data)
            self.remaining -= len(data)

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=38769)
args = parser.parse_args()
os.chdir(Path(__file__).resolve().parent)
print(f'Review: http://127.0.0.1:{args.port}/', flush=True)
ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
