"""
API Client — frontend calls to the FastAPI backend.
Uses urllib (built-in) to avoid Windows firewall issues with requests.
"""
import os
import json
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


class LTAClient:
    def __init__(self, api_key: str = None):
        self.base = API_URL.rstrip("/")
        self.api_key = api_key

    def _headers(self):
        h = {}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        return h

    def _request(self, method, path, json_data=None, params=None):
        url = f"{self.base}{path}"
        if params:
            url += "?" + urlencode(params)

        body = None
        headers = self._headers()
        if json_data is not None:
            body = json.dumps(json_data).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = Request(url, data=body, headers=headers, method=method)
        try:
            with urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            error_body = e.read().decode("utf-8")
            raise Exception(f"HTTP {e.code}: {error_body}")

    def _upload(self, path, filename, file_bytes, params=None):
        """Multipart file upload."""
        url = f"{self.base}{path}"
        if params:
            url += "?" + urlencode(params)

        boundary = "----LTABoundary123456"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: text/csv\r\n\r\n"
        ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

        headers = self._headers()
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"

        req = Request(url, data=body, headers=headers, method="POST")
        try:
            with urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            error_body = e.read().decode("utf-8")
            raise Exception(f"HTTP {e.code}: {error_body}")

    def _get_text(self, path, params=None):
        url = f"{self.base}{path}"
        if params:
            url += "?" + urlencode(params)
        headers = self._headers()
        req = Request(url, headers=headers, method="GET")
        with urlopen(req, timeout=60) as resp:
            return resp.read().decode("utf-8")

    # ── Auth ──
    def create_user(self, name: str, email: str) -> dict:
        return self._request("POST", "/api/users", json_data={"name": name, "email": email})

    # ── Tapes ──
    def upload_tape(self, filename: str, csv_bytes: bytes) -> dict:
        return self._upload("/api/tapes", filename, csv_bytes)

    def list_tapes(self) -> list:
        return self._request("GET", "/api/tapes")

    def get_tape(self, tape_id: str) -> dict:
        return self._request("GET", f"/api/tapes/{tape_id}")

    def delete_tape(self, tape_id: str) -> dict:
        return self._request("DELETE", f"/api/tapes/{tape_id}")

    # ── Mapping ──
    def update_mapping(self, tape_id: str, mapping: dict) -> dict:
        return self._request("PUT", f"/api/tapes/{tape_id}/mapping", json_data={"mapping": mapping})

    def auto_match(self, tape_id: str, mode: str = "rule") -> dict:
        return self._request("POST", f"/api/tapes/{tape_id}/automatch", params={"mode": mode})

    # ── Analysis ──
    def get_analysis(self, tape_id: str) -> dict:
        return self._request("GET", f"/api/tapes/{tape_id}/analysis")

    def get_validation(self, tape_id: str) -> dict:
        return self._request("GET", f"/api/tapes/{tape_id}/validation")

    # ── Regression ──
    def run_regression(self, tape_id: str, x_col: str, y_col: str, z_col: str = None) -> dict:
        body = {"x_column": x_col, "y_column": y_col}
        if z_col: body["z_column"] = z_col
        return self._request("POST", f"/api/tapes/{tape_id}/regression", json_data=body)

    # ── Export ──
    def export_csv(self, tape_id: str, filter_col: str = None, filter_val: str = None) -> str:
        params = {}
        if filter_col: params["filter_col"] = filter_col
        if filter_val: params["filter_val"] = filter_val
        return self._get_text(f"/api/tapes/{tape_id}/export", params=params)

    # ── Templates ──
    def create_template(self, name: str, originator: str, mapping: dict) -> dict:
        return self._request("POST", "/api/templates",
                             json_data={"name": name, "originator": originator, "mapping": mapping})

    def list_templates(self) -> list:
        return self._request("GET", "/api/templates")

    def delete_template(self, template_id: str) -> dict:
        return self._request("DELETE", f"/api/templates/{template_id}")

    # ── Custom Fields ──
    def create_field(self, key: str, label: str, patterns: list) -> dict:
        return self._request("POST", "/api/fields",
                             json_data={"key": key, "label": label, "patterns": patterns})

    def list_fields(self) -> list:
        return self._request("GET", "/api/fields")

    def delete_field(self, field_id: str) -> dict:
        return self._request("DELETE", f"/api/fields/{field_id}")

    def list_standard_fields(self) -> dict:
        return self._request("GET", "/api/fields/standard")

    # ── Health ──
    def health(self) -> dict:
        return self._request("GET", "/api/health")
