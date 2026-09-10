from functools import partial
from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import pytest
from scripts.serve_action_preview import RangeHandler


def test_video_seeking_ranges_return_exact_bytes(tmp_path):
    payload = bytes(range(256)) * 4
    (tmp_path / "video.webm").write_bytes(payload)
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(RangeHandler, directory=tmp_path)
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/video.webm"
    try:
        for value, start, end in [
            ("bytes=0-0", 0, 0),
            ("bytes=233-499", 233, 499),
            ("bytes=1000-", 1000, 1023),
            ("bytes=-7", 1017, 1023),
        ]:
            with urlopen(Request(url, headers={"Range": value})) as response:
                assert response.status == 206
                assert response.headers["Content-Range"] == f"bytes {start}-{end}/1024"
                assert response.read() == payload[start : end + 1]
        with urlopen(Request(url, method="HEAD")) as response:
            assert response.headers["Accept-Ranges"] == "bytes"
            assert response.read() == b""
        with pytest.raises(HTTPError) as error:
            urlopen(Request(url, headers={"Range": "bytes=1024-"}))
        assert error.value.code == 416
        with urlopen(url) as response:
            assert response.read() == payload
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
