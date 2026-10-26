"""Local research-demo server for the ECG dashboard.

Serves static files and forwards inference while keeping HF_TOKEN in .env.
Not a production access-control or compliance service.
"""

from __future__ import annotations

import json
import math
import mimetypes
import os
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / '.env'
DEFAULT_SPACE_URL = 'https://md-ubaid-cardioai-backend.hf.space/call/predict_anomaly'


def load_env_file() -> None:
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            # A blank shell export must not mask the actual local .env secret.
            if value and not os.environ.get(key):
                os.environ[key] = value


def validate_signal(signal: object) -> list[list[float]]:
    if not isinstance(signal, list) or len(signal) != 4800:
        raise ValueError('Expected exactly 4800 ECG samples.')
    if any(not isinstance(row, list) or len(row) != 12 for row in signal):
        raise ValueError('Each ECG sample must contain exactly 12 leads.')
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) for row in signal for value in row):
        raise ValueError('ECG values must be finite numbers.')
    return signal


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def send_json(self, status: HTTPStatus, body: dict) -> None:
        content = json.dumps(body).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != '/api/analyze':
            self.send_json(HTTPStatus.NOT_FOUND, {'error': 'Unknown API route.'})
            return
        try:
            content_length = int(self.headers.get('Content-Length', '0'))
            if content_length <= 0 or content_length > 25_000_000:
                raise ValueError('Invalid request size.')
            body = json.loads(self.rfile.read(content_length))
            result = self.forward_to_space(validate_signal(body.get('signal')))
            self.send_json(HTTPStatus.OK, result)
        except ValueError as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {'error': str(error)})
        except HTTPError as error:
            detail = error.read().decode('utf-8', errors='replace')[:1000]
            self.send_json(HTTPStatus.BAD_GATEWAY, {'error': f'Hugging Face returned {error.code}: {detail}'})
        except URLError as error:
            self.send_json(HTTPStatus.BAD_GATEWAY, {'error': f'Could not reach Hugging Face: {error.reason}'})
        except Exception as error:
            self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {'error': f'Inference proxy error: {error}'})

    @staticmethod
    def forward_to_space(signal: list[list[float]]) -> dict:
        token = os.environ.get('HF_TOKEN')
        if not token:
            raise ValueError('HF_TOKEN is missing. Copy .env.example to .env and add a new token.')
        
        try:
            from gradio_client import Client
            import httpx
        except ImportError:
            raise ValueError('gradio_client is not installed. Run: pip install gradio_client httpx')

        # 1. Override the strict default network timeouts (60 seconds)
        # This prevents the client from dropping the connection while ZeroGPU wakes up
        client = Client(
            "md-ubaid/cardioai-backend", 
            token=token,
            httpx_kwargs={"timeout": httpx.Timeout(60.0)}
        )
        
        # 2. Compress the payload (Ensure you keep this step!)
        # Converts long floats to 4 decimals, reducing upload size by ~75%
        compressed_signal = [[round(float(val), 4) for val in row] for row in signal]
        signal_string = json.dumps(compressed_signal)
        
        # 3. Call the API route
        result_string = client.predict(
            signal_str=signal_string,
            api_name="/predict_anomaly"
        )
        
        if isinstance(result_string, str):
            return json.loads(result_string)
        return result_string


if __name__ == '__main__':
    load_env_file()
    mimetypes.add_type('application/javascript', '.js')
    print('ECG dashboard: http://127.0.0.1:8765')
    ThreadingHTTPServer(('127.0.0.1', 8765), DashboardHandler).serve_forever()
