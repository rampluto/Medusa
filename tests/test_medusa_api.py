from __future__ import annotations

import asyncio

import httpx
from fastapi import FastAPI
from fastapi.routing import APIRoute

from medusa_env.server.custom_api import register_custom_routes


app_under_test = FastAPI()
register_custom_routes(app_under_test)


def request(method: str, url: str, **kwargs):
    async def _send():
        transport = httpx.ASGITransport(app=app_under_test)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.request(method, url, **kwargs)

    return asyncio.run(_send())


def test_tasks_endpoint_returns_catalog():
    response = request("GET", "/api/tasks")
    assert response.status_code == 200
    payload = response.json()
    assert "tasks" in payload
    task_ids = {task["id"] for task in payload["tasks"]}
    assert "basic_pipeline" in task_ids
    assert "gauntlet_30day" in task_ids


def test_action_space_exposes_known_actions():
    response = request("GET", "/api/action-space")
    assert response.status_code == 200
    payload = response.json()
    action_names = {item["action"] for item in payload["actions"]}
    assert "PROFILE_TABLE" in action_names
    assert "EXECUTE_MERGE" in action_names
    assert "COMMIT_DAY" in action_names


def test_agents_endpoint_returns_catalog():
    response = request("GET", "/api/agents")
    assert response.status_code == 200
    payload = response.json()
    assert payload["default_agent_id"] == "task_aware"
    agent_ids = {agent["id"] for agent in payload["agents"]}
    assert {"task_aware", "governance_first", "random", "always_commit"} <= agent_ids


def test_preview_returns_initial_state_without_sessions():
    response = request(
        "POST",
        "/api/run/preview",
        json={"task_id": "basic_pipeline", "actions": []},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["task"]["id"] == "basic_pipeline"
    assert payload["summary"]["stage"] == "running"
    assert payload["action_count"] == 0
    assert payload["observation"]["done"] is False
    assert payload["tables"]["bronze_a"]["rows"] > 0


def test_autorun_executes_trace_for_selected_agent():
    response = request(
        "POST",
        "/api/run/autorun/basic_pipeline",
        json={"agent_id": "task_aware"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["agent"]["id"] == "task_aware"
    assert payload["action_count"] > 0
    assert payload["actions"][-1]["action"] == "COMMIT_DAY"
    assert payload["summary"]["done"] is True


def test_reset_returns_initial_state_for_task():
    response = request("POST", "/api/run/reset/basic_pipeline", json={})
    assert response.status_code == 200
    payload = response.json()
    assert payload["task"]["id"] == "basic_pipeline"
    assert payload["action_count"] == 0
    assert payload["summary"]["stage"] == "running"


def test_step_appends_action_and_updates_trace():
    response = request(
        "POST",
        "/api/run/step",
        json={
            "task_id": "basic_pipeline",
            "actions": [],
            "next_action": {"action": "PROFILE_TABLE", "params": {"table": "bronze"}},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["action_count"] == 1
    assert payload["actions"][0]["action"] == "PROFILE_TABLE"
    assert payload["state"]["profiled_tables_today"]["bronze"] == 1
    assert payload["summary"]["step_idx"] == 1


def test_table_preview_returns_requested_rows():
    response = request(
        "POST",
        "/api/run/tables",
        json={
            "task_id": "basic_pipeline",
            "actions": [],
            "table": "daily_raw",
            "page": 1,
            "page_size": 5,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["table"] == "daily_raw"
    assert payload["page_size"] == 5
    assert payload["total_rows"] >= len(payload["rows"])
    assert "user_id" in payload["columns"]


def test_evaluate_scores_committed_trace():
    actions = [
        {"action": "PROFILE_TABLE", "params": {"table": "bronze"}},
        {"action": "DEDUPLICATE", "params": {"table": "bronze", "key": "user_id"}},
        {"action": "EXECUTE_MERGE", "params": {}},
        {"action": "COMMIT_DAY", "params": {}},
    ]
    response = request(
        "POST",
        "/api/run/evaluate",
        json={"task_id": "basic_pipeline", "actions": actions},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["task"]["id"] == "basic_pipeline"
    assert payload["evaluation"]["grade"] in {"S", "A", "B", "C", "F"}
    assert payload["evaluation"]["score"] >= 0.0


def test_autorun_returns_v4_action_trace():
    response = request(
        "POST",
        "/api/run/autorun/basic_pipeline",
        json={"agent_id": "task_aware"},
    )
    assert response.status_code == 200
    payload = response.json()
    action_names = [action["action"] for action in payload["actions"]]
    assert "PROFILE_TABLE" in action_names
    assert "EXECUTE_MERGE" in action_names
    assert "COMMIT_DAY" in action_names


def test_medusa_frontend_serves_custom_page():
    root_route = next(
        route for route in app_under_test.routes if isinstance(route, APIRoute) and route.path == "/medusa"
    )
    root_response = asyncio.run(root_route.endpoint())
    assert root_response.headers["location"] == "/medusa/studio"

    tasks_route = next(
        route for route in app_under_test.routes if isinstance(route, APIRoute) and route.path == "/medusa/tasks"
    )
    tasks_response = asyncio.run(tasks_route.endpoint())
    assert str(tasks_response.path).endswith("server/frontend/tasks.html")

    studio_route = next(
        route for route in app_under_test.routes if isinstance(route, APIRoute) and route.path == "/medusa/studio"
    )
    studio_response = asyncio.run(studio_route.endpoint())
    assert str(studio_response.path).endswith("server/frontend/studio.html")

    audit_route = next(
        route for route in app_under_test.routes if isinstance(route, APIRoute) and route.path == "/medusa/audit"
    )
    audit_response = asyncio.run(audit_route.endpoint())
    assert str(audit_response.path).endswith("server/frontend/audit.html")
