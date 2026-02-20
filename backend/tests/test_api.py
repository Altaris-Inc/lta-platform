"""
API integration tests — test endpoints with httpx + async.

Run: pytest backend/tests/ -v
"""
import pytest
import io
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.db import init_db, engine, Base


@pytest.fixture(autouse=True)
async def setup_db():
    """Create fresh tables for each test."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
async def user_and_key(client):
    """Create a user and return (user_data, headers)."""
    resp = await client.post("/api/users", json={"name": "Test User", "email": "test@example.com"})
    assert resp.status_code == 200
    data = resp.json()
    headers = {"X-API-Key": data["api_key"]}
    return data, headers


@pytest.fixture
def sample_csv():
    return (
        "Loan_ID,Current_Balance,Original_Amount,Interest_Rate,FICO_Origination,Loan_Status,State\n"
        "L001,25000,30000,12.5,720,Current,CA\n"
        "L002,15000,20000,8.5,780,Current,NY\n"
        "L003,10000,12000,22.0,620,30DPD,TX\n"
    )


class TestHealth:
    @pytest.mark.anyio
    async def test_health(self, client):
        resp = await client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    @pytest.mark.anyio
    async def test_standard_fields(self, client):
        resp = await client.get("/api/fields/standard")
        assert resp.status_code == 200
        data = resp.json()
        assert "loan_id" in data
        assert len(data) == 47


class TestUsers:
    @pytest.mark.anyio
    async def test_create_user(self, client):
        resp = await client.post("/api/users", json={"name": "Alice", "email": "alice@test.com"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Alice"
        assert "api_key" in data
        assert len(data["api_key"]) == 32


class TestAuth:
    @pytest.mark.anyio
    async def test_invalid_key_rejected(self, client):
        resp = await client.get("/api/tapes", headers={"X-API-Key": "bad-key"})
        assert resp.status_code == 401

    @pytest.mark.anyio
    async def test_missing_key_rejected(self, client):
        resp = await client.get("/api/tapes")
        assert resp.status_code == 422  # missing header


class TestTapes:
    @pytest.mark.anyio
    async def test_upload_and_list(self, client, user_and_key, sample_csv):
        _, headers = user_and_key
        # Upload
        resp = await client.post(
            "/api/tapes", headers=headers,
            files={"file": ("test.csv", sample_csv, "text/csv")},
        )
        assert resp.status_code == 200
        tape = resp.json()
        assert tape["row_count"] == 3
        assert tape["col_count"] == 7
        assert len(tape["mapping"]) > 0

        # List
        resp = await client.get("/api/tapes", headers=headers)
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    @pytest.mark.anyio
    async def test_auto_match_on_upload(self, client, user_and_key, sample_csv):
        _, headers = user_and_key
        resp = await client.post(
            "/api/tapes", headers=headers,
            files={"file": ("test.csv", sample_csv, "text/csv")},
        )
        tape = resp.json()
        mp = tape["mapping"]
        assert mp.get("loan_id") == "Loan_ID"
        assert mp.get("current_balance") == "Current_Balance"

    @pytest.mark.anyio
    async def test_analysis_populated(self, client, user_and_key, sample_csv):
        _, headers = user_and_key
        resp = await client.post(
            "/api/tapes", headers=headers,
            files={"file": ("test.csv", sample_csv, "text/csv")},
        )
        tape = resp.json()
        resp = await client.get(f"/api/tapes/{tape['id']}/analysis", headers=headers)
        assert resp.status_code == 200
        an = resp.json()
        assert an["N"] == 3
        assert an["tb"] > 0

    @pytest.mark.anyio
    async def test_delete_tape(self, client, user_and_key, sample_csv):
        _, headers = user_and_key
        resp = await client.post(
            "/api/tapes", headers=headers,
            files={"file": ("test.csv", sample_csv, "text/csv")},
        )
        tape_id = resp.json()["id"]
        resp = await client.delete(f"/api/tapes/{tape_id}", headers=headers)
        assert resp.status_code == 200

        resp = await client.get("/api/tapes", headers=headers)
        assert len(resp.json()) == 0


class TestTemplates:
    @pytest.mark.anyio
    async def test_crud_template(self, client, user_and_key):
        _, headers = user_and_key

        # Create
        resp = await client.post("/api/templates", headers=headers, json={
            "name": "LendingClub v1", "originator": "LendingClub",
            "mapping": {"loan_id": "Loan_ID", "current_balance": "Current_Balance"},
        })
        assert resp.status_code == 200
        tpl = resp.json()
        assert tpl["name"] == "LendingClub v1"

        # List
        resp = await client.get("/api/templates", headers=headers)
        assert len(resp.json()) == 1

        # Delete
        resp = await client.delete(f"/api/templates/{tpl['id']}", headers=headers)
        assert resp.status_code == 200


class TestCustomFields:
    @pytest.mark.anyio
    async def test_crud_custom_field(self, client, user_and_key):
        _, headers = user_and_key

        resp = await client.post("/api/fields", headers=headers, json={
            "key": "vehicle_make", "label": "Vehicle Make", "patterns": ["make", "manufacturer"],
        })
        assert resp.status_code == 200

        resp = await client.get("/api/fields", headers=headers)
        fields = resp.json()
        assert len(fields) == 1
        assert fields[0]["key"] == "vehicle_make"


class TestDataIsolation:
    @pytest.mark.anyio
    async def test_users_cant_see_others_tapes(self, client, sample_csv):
        # Create two users
        r1 = await client.post("/api/users", json={"name": "A", "email": "a@test.com"})
        r2 = await client.post("/api/users", json={"name": "B", "email": "b@test.com"})
        h1 = {"X-API-Key": r1.json()["api_key"]}
        h2 = {"X-API-Key": r2.json()["api_key"]}

        # User A uploads
        await client.post("/api/tapes", headers=h1,
                          files={"file": ("test.csv", sample_csv, "text/csv")})

        # User B sees nothing
        resp = await client.get("/api/tapes", headers=h2)
        assert len(resp.json()) == 0

        # User A sees their tape
        resp = await client.get("/api/tapes", headers=h1)
        assert len(resp.json()) == 1
