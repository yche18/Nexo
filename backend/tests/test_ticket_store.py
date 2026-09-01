from concurrent.futures import ThreadPoolExecutor
import copy
import sqlite3
import uuid

import pytest

from core.ticket_store import TicketStore


SNAPSHOT = {
    "facts": {"issue_description": {"value": "无法开机", "source": "user_reported",
                                      "source_quote": "电脑无法开机", "request_id": "r0"}},
    "steps": [{"description": "检查外部供电", "status": "user_reported_done",
               "source_quote": "我检查过外部供电，还是无法开机", "result": "还是无法开机"}],
    "asset_lookup": {"source": "demo", "status": "found", "asset_id": "NB-001",
                     "asset": {"device_model": "DemoBook 14"}},
}


def create(store, **overrides):
    arguments = dict(user_id="u1", conv_id="c1", case_id="case1", summary="电脑无法开机，已检查外部供电",
                     snapshot=copy.deepcopy(SNAPSHOT), request_id="r1", submission_text="确认提交工单")
    arguments.update(overrides)
    return store.create_ticket(**arguments)


def test_ticket_persists_snapshot_sources_and_can_be_read_after_restart(tmp_path):
    path = tmp_path / "nested" / "support" / "tickets.sqlite3"
    original = copy.deepcopy(SNAPSHOT)
    result = create(TicketStore(path), snapshot=original)
    assert result["success"] and result["created"]
    ticket = result["ticket"]
    assert ticket["ticket_id"].startswith("NEXO-")
    assert uuid.UUID(hex=ticket["ticket_id"][5:]).hex.upper() == ticket["ticket_id"][5:]
    assert ticket["status"] == "pending"
    assert ticket["snapshot"] == SNAPSHOT
    assert original == SNAPSHOT
    assert ticket["request_id"] == "r1" and ticket["submission_text"] == "确认提交工单"
    original["facts"].clear()
    restarted = TicketStore(path)
    read_back = restarted.get_ticket(ticket["ticket_id"], "u1", "c1")
    assert read_back == {"success": True, "found": True, "ticket": ticket}
    assert read_back["ticket"]["snapshot"] == SNAPSHOT
    assert "snapshot_json" not in ticket and "idempotency_key" not in ticket


def test_duplicate_request_returns_original_ticket_without_overwriting(tmp_path):
    store = TicketStore(tmp_path / "tickets.db")
    original = create(store)
    repeated = create(store, summary="另一份摘要", snapshot={"facts": {}}, request_id="r2",
                      submission_text="再次确认提交")
    assert repeated == {"success": True, "created": False, "ticket": original["ticket"]}
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 1


def test_concurrent_creates_have_one_winner_and_one_persisted_ticket(tmp_path):
    path = tmp_path / "tickets.db"
    store = TicketStore(path)
    def submit(index):
        # Repeated initialization and distinct connections also work across workers.
        worker_store = TicketStore(path) if index % 2 else store
        return create(worker_store, summary=f"summary-{index}", request_id=f"r-{index}")
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(submit, range(16)))
    assert sum(result["created"] for result in results) == 1
    assert len({result["ticket"]["ticket_id"] for result in results}) == 1
    winner = next(result["ticket"] for result in results if result["created"])
    assert all(result["ticket"] == winner for result in results)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 1


@pytest.mark.parametrize("user_id,conv_id", [("other-user", "c1"), ("u1", "other-conv"), ("other", "other")])
def test_ticket_is_not_readable_outside_user_and_conversation_scope(tmp_path, user_id, conv_id):
    store = TicketStore(tmp_path / "tickets.db")
    ticket = create(store)["ticket"]
    assert store.get_ticket(ticket["ticket_id"], user_id, conv_id) == {"success": True, "found": False, "ticket": None}


def test_unique_key_includes_user_conversation_and_case(tmp_path):
    store = TicketStore(tmp_path / "tickets.db")
    results = [create(store), create(store, user_id="u2"), create(store, conv_id="c2"), create(store, case_id="case2")]
    assert all(result["created"] for result in results)
    assert len({result["ticket"]["ticket_id"] for result in results}) == 4


def test_unknown_ticket_and_sql_like_values_are_not_executable(tmp_path):
    store = TicketStore(tmp_path / "tickets.db")
    create(store)
    assert store.get_ticket("missing", "u1", "c1") == {"success": True, "found": False, "ticket": None}
    assert store.get_ticket("' OR 1=1 --", "u1", "c1")["found"] is False
    text = "'; DROP TABLE tickets; --"
    result = create(store, case_id="sql-like", summary=text)
    assert store.get_ticket(result["ticket"]["ticket_id"], "u1", "c1")["ticket"]["summary"] == text


def test_write_error_propagates_without_fake_success_or_partial_ticket(tmp_path, monkeypatch):
    path = tmp_path / "tickets.db"
    store = TicketStore(path)
    def read_only_connection():
        connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        return connection
    monkeypatch.setattr(store, "_connect", read_only_connection)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        create(store)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 0


def test_database_open_failure_is_propagated(tmp_path):
    directory = tmp_path / "database-is-a-directory"
    directory.mkdir()
    with pytest.raises(sqlite3.OperationalError):
        TicketStore(directory)


@pytest.mark.parametrize("overrides", [{"case_id": ""}, {"user_id": None}, {"snapshot": []}])
def test_invalid_input_does_not_create_ticket(tmp_path, overrides):
    store = TicketStore(tmp_path / "tickets.db")
    with pytest.raises(ValueError):
        create(store, **overrides)
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 0


@pytest.mark.parametrize("lowercase", [False, True])
def test_created_ticket_identifier_passes_real_query_parser_and_store_round_trip(tmp_path, lowercase):
    from agents.tools import parse_ticket_action
    store = TicketStore(tmp_path / "tickets.db")
    ticket = create(store)["ticket"]
    supplied_id = ticket["ticket_id"].lower() if lowercase else ticket["ticket_id"]
    action = parse_ticket_action(f"查询工单 {supplied_id}")
    assert action == {"action": "get_ticket", "ticket_id": ticket["ticket_id"]}
    assert store.get_ticket(action["ticket_id"], "u1", "c1") == {
        "success": True, "found": True, "ticket": ticket,
    }
