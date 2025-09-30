import streamlit as st
import httpx
import json
from crewai import Agent, Task, Crew, LLM

# ===========================
# LLM Setup with Ollama
# ===========================
ollama_llm = LLM(
    model="ollama/codellama:7b",
    base_url="http://localhost:11434"  # Default Ollama server
)

# ===========================
# Helpers
# ===========================
def run_test_cases(testcases: list, base_url: str):
    """Run API test cases using httpx and return logs."""
    logs = []
    client = httpx.Client(timeout=10)

    for tc in testcases:
        method = tc.get("method", "GET").upper()
        endpoint = tc.get("endpoint", "/")
        url = base_url.rstrip("/") + endpoint.replace("{chatId}", tc.get("chatId", "123"))  # replace path var
        data = tc.get("data", {})
        params = tc.get("params", {})

        try:
            response = client.request(method, url, json=data, params=params)
            passed = response.status_code in tc.get("expected_status", [200])
            logs.append({
                "name": tc.get("name", "Unnamed"),
                "endpoint": endpoint,
                "method": method,
                "status_code": response.status_code,
                "passed": passed,
                "response": response.text[:200]  # limit response preview
            })
        except Exception as e:
            logs.append({
                "name": tc.get("name", "Unnamed"),
                "endpoint": endpoint,
                "method": method,
                "error": str(e),
                "passed": False
            })
    client.close()
    return logs


# ===========================
# Streamlit UI
# ===========================
st.title("🚀 CrewAI Chat API Tester")

base_url = st.text_input(
    "Enter Base API URL", 
    placeholder="http://localhost:8080"
)

if st.button("Start Testing") and base_url:
    # ===========================
    # Agents
    # ===========================
    api_extractor_agent = Agent(
        role="API Extractor",
        goal="Provide all chat APIs",
        backstory="Knows the chat service endpoints.",
        llm=ollama_llm
    )

    planner_agent = Agent(
        role="Planner",
        goal="Ensure execution starts with POST /chat/message",
        backstory="Knows dependencies between chat APIs.",
        llm=ollama_llm
    )

    tester_agent = Agent(
        role="Tester",
        goal="Generate and run predefined testcases for chat APIs",
        backstory="Executes both positive and negative cases.",
        llm=ollama_llm
    )

    reporter_agent = Agent(
        role="Reporter",
        goal="Summarize test results",
        backstory="Generates final report with pass/fail summary.",
        llm=ollama_llm
    )

    # ===========================
    # Tasks
    # ===========================
    extract_task = Task(
        description="List chat APIs",
        agent=api_extractor_agent,
        expected_output=str([
            "/chat/message (POST)",
            "/chat/history (GET)",
            "/chat/{chatId} (GET)",
            "/chat/rename/{chatId} (PUT)",
            "/chat/{chatId} (DELETE)"
        ])
    )

    plan_task = Task(
        description="Ensure correct execution order: message → history → load → rename → delete",
        agent=planner_agent,
        context=[extract_task],
        expected_output="Ordered execution flow"
    )

    # Predefined Testcases
    predefined_testcases = [
        {
            "name": "Create new chat message",
            "method": "POST",
            "endpoint": "/chat/message",
            "data": {"prompt": "Hello from test"},
            "expected_status": [200]
        },
        {
            "name": "Get chat history",
            "method": "GET",
            "endpoint": "/chat/history",
            "expected_status": [200]
        },
        {
            "name": "Load chat by ID",
            "method": "GET",
            "endpoint": "/chat/{chatId}",
            "chatId": "123",  # sample
            "expected_status": [200, 404]  # allow 404 if not found
        },
        {
            "name": "Rename chat",
            "method": "PUT",
            "endpoint": "/chat/rename/{chatId}",
            "params": {"title": "Renamed Chat"},
            "chatId": "123",
            "expected_status": [200, 404]
        },
        {
            "name": "Delete chat",
            "method": "DELETE",
            "endpoint": "/chat/{chatId}",
            "chatId": "123",
            "expected_status": [200, 404]
        }
    ]

    testcase_task = Task(
        description="Use predefined chat API testcases",
        agent=tester_agent,
        context=[plan_task],
        expected_output=json.dumps(predefined_testcases, indent=2)
    )

    report_task = Task(
        description="Summarize execution logs into JSON report",
        agent=reporter_agent,
        context=[testcase_task],
        expected_output="JSON summary"
    )

    # ===========================
    # Crew Execution
    # ===========================
    crew = Crew(
        agents=[api_extractor_agent, planner_agent, tester_agent, reporter_agent],
        tasks=[extract_task, plan_task, testcase_task, report_task],
        verbose=True,
    )

    results = crew.kickoff()

    # ===========================
    # Execute Real API Testcases
    # ===========================
    logs = run_test_cases(predefined_testcases, base_url)

    # Convert CrewOutput into dict safely
    results_dict = results.dict() if hasattr(results, "dict") else {}

    final_summary = {
        "execution_logs": logs,
        "report": results_dict.get("Reporter", "No report generated")
    }

    # ===========================
    # Display Results
    # ===========================
    st.subheader("🧩 Agent Outputs")
    st.json(results_dict)

    st.subheader("✅ Final Report")
    st.json(final_summary)
