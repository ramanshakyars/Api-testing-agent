import streamlit as st
import httpx
import yaml
import json
import re
from urllib.parse import urlparse
from crewai import Agent, Task, Crew, LLM

# ===========================
# LLM Setup with Ollama
# ===========================
ollama_llm = LLM(
    model="ollama/codellama:7b",
    base_url="http://localhost:11434"  # Default Ollama server
)

# ===========================
# Helpers
# ===========================
def fetch_swagger(url: str):
    """Fetch Swagger/OpenAPI spec from a given URL."""
    try:
        r = httpx.get(url, timeout=10)
        r.raise_for_status()
        if r.headers.get("content-type", "").startswith("application/json"):
            return r.json()
        return yaml.safe_load(r.text)
    except Exception as e:
        return {"error": str(e)}

def run_test_cases(testcases: list, base_url: str, auth=None):
    """Run API test cases using httpx and return logs."""
    logs = []
    client = httpx.Client(auth=auth, timeout=10)

    for tc in testcases:
        method = tc.get("method", "GET").upper()
        endpoint = tc.get("endpoint", "/")
        url = base_url.rstrip("/") + endpoint
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
                "response": response.text[:300]  # limit size
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

def safe_json_parse(raw_output: str):
    """Extract and parse JSON array from raw LLM output safely."""
    try:
        return json.loads(raw_output)
    except:
        match = re.search(r'\[.*\]', raw_output, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                return []
        return []

# ===========================
# Streamlit UI
# ===========================
st.title("🚀 CrewAI API Tester")

swagger_url = st.text_input(
    "Enter Swagger/OpenAPI URL", 
    placeholder="http://api-resilixstream.mathesislabs.com/resilixstream/api/v3/api-docs"
)
username = st.text_input("Username (if required)")
password = st.text_input("Password (if required)", type="password")

if st.button("Start Testing") and swagger_url:
    # Auto-extract base_url from Swagger URL
    parsed = urlparse(swagger_url)
    path_parts = parsed.path.rstrip('/').split('/')
    base_path = "/".join(path_parts[:-1])
    base_url = f"{parsed.scheme}://{parsed.netloc}{base_path}"
    st.text(f"Base API URL automatically set to: {base_url}")

    with st.spinner("Fetching API documentation..."):
        spec = fetch_swagger(swagger_url)
        if "error" in spec:
            st.error(f"Failed to fetch spec: {spec['error']}")
        else:
            st.success("Swagger spec fetched!")

            # ===========================
            # Agents
            # ===========================
            planner_agent = Agent(
                role="Planner",
                goal="Plan which API endpoints to test in correct order",
                backstory="Specialist in analyzing Swagger/OpenAPI and planning API execution.",
                llm=ollama_llm
            )

            testcase_agent = Agent(
                role="Testcase Generator",
                goal="Generate positive and negative testcases for each endpoint",
                backstory="Expert in writing robust test cases covering happy path and failures.",
                llm=ollama_llm
            )

            reporter_agent = Agent(
                role="Reporter",
                goal="Generate summary of test results in JSON format",
                backstory="Expert in reporting test results with details of pass/fail.",
                llm=ollama_llm
            )

            # ===========================
            # Tasks
            # ===========================
            api_paths = list(spec.get("paths", {}).keys())

            plan_task = Task(
                description=f"Analyze API spec and decide execution order. Endpoints: {api_paths[:20]}...",
                agent=planner_agent,
                expected_output="Ordered list of endpoints with reasoning."
            )

            testcase_task = Task(
                description="""Generate ONLY valid JSON test cases.
Output MUST be a JSON array of objects with this schema:
[
  {
    "name": "string - testcase name",
    "method": "GET|POST|PUT|DELETE",
    "endpoint": "/string - API path",
    "params": { "key": "value" },   # optional
    "data": { "key": "value" },     # optional
    "expected_status": [200]        # list of status codes
  }
]
Do NOT include explanations or extra text outside the JSON.
""",
                agent=testcase_agent,
                context=[plan_task],
                expected_output="Strict JSON array of test cases."
            )

            report_task = Task(
                description="Summarize test execution logs into JSON report with pass/fail counts.",
                agent=reporter_agent,
                context=[testcase_task],
                expected_output="JSON summary"
            )

            # ===========================
            # Crew Execution
            # ===========================
            crew = Crew(
                agents=[planner_agent, testcase_agent, reporter_agent],
                tasks=[plan_task, testcase_task, report_task],
                verbose=True,
            )

            results = crew.kickoff()

            # ===========================
            # Execute Testcases (real API calls)
            # ===========================
            raw_testcases = results.get("Testcase Generator", "[]")
            testcases = safe_json_parse(raw_testcases)

            if not testcases:
                st.error("❌ No valid testcases generated.")
            else:
                with st.spinner("Executing real API test cases..."):
                    auth = (username, password) if username and password else None
                    logs = run_test_cases(testcases, base_url, auth=auth)

                # ===========================
                # Final Report
                # ===========================
                final_summary = {
                    "execution_logs": logs,
                    "report": results.get("Reporter", "No report generated")
                }

                st.subheader("✅ Final Report")
                st.json(final_summary)