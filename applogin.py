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
    base_url="http://localhost:11434",
    timeout=7200
)

# ===========================
# Helpers
# ===========================
def fetch_swagger(url: str):
    """Fetch Swagger/OpenAPI spec"""
    try:
        r = httpx.get(url, timeout=10)
        r.raise_for_status()
        if r.headers.get("content-type", "").startswith("application/json"):
            return r.json()
        return yaml.safe_load(r.text)
    except Exception as e:
        return {"error": str(e)}

def safe_json_parse(raw_output: str):
    """Safely parse JSON from LLM output"""
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

def perform_login_test(email: str, password: str):
    """Always hit the correct login endpoint"""
    login_url = "http://api-resilixstream.mathesislabs.com/resilixstream/api/public/login"
    credentials = {"email": email, "password": password}

    st.write(f"🔑 Hitting Login API: {login_url}")
    try:
        resp = httpx.post(
            login_url,
            json=credentials,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0"
            },
            timeout=10
        )
        st.write(f"Status: {resp.status_code}")

        try:
            data = resp.json()
            st.json(data)
            if resp.status_code == 200:
                token = data.get("token") or data.get("accessToken")
                return True, token
        except:
            st.error("Response not JSON")
            st.code(resp.text)
        return False, None
    except Exception as e:
        st.error(f"Login request failed: {e}")
        return False, None

def run_test_cases(testcases: list, base_url: str, token=None):
    """Execute generated testcases"""
    logs = []
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    client = httpx.Client(timeout=10, headers=headers)

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
                "response": response.text[:200]
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

def get_output_safe(task_output):
    # Works across crewai versions
    if hasattr(task_output, "output"):
        return task_output.output
    if hasattr(task_output, "final_output"):
        return task_output.final_output
    return str(task_output)

def print_report_section(report):
    if isinstance(report, dict):
        st.write("### 📊 Test Summary")
        if "summary" in report and isinstance(report["summary"], dict):
            summary = report["summary"]
            passed = summary.get("passed", summary.get("num_passed"))
            failed = summary.get("failed", summary.get("num_failed"))
            total = summary.get("total", summary.get("num_total"))
            st.write(f"**Total:** {total if total else 'N/A'}")
            st.write(f"**Passed:** {passed if passed is not None else 'N/A'}")
            st.write(f"**Failed:** {failed if failed is not None else 'N/A'}")
            st.write("**Breakdown by endpoint:**")
            if "breakdown" in summary:
                for endpoint, stat in summary["breakdown"].items():
                    st.write(f"- `{endpoint}`: {stat}")
            st.write("---")
        else:
            st.write(report["summary"])
        st.write("### 📝 Execution Logs")
        st.json(report["execution_logs"])
    else:
        st.write(report)

# ===========================
# Streamlit UI
# ===========================
st.title("🚀 CrewAI API Tester")

if "spec" not in st.session_state:
    st.session_state.spec = None
if "api_paths" not in st.session_state:
    st.session_state.api_paths = []
if "base_url" not in st.session_state:
    st.session_state.base_url = None
if "auth_token" not in st.session_state:
    st.session_state.auth_token = None
if "login_done" not in st.session_state:
    st.session_state.login_done = False
if "final_report" not in st.session_state:
    st.session_state.final_report = None

swagger_url = st.text_input(
    "Enter Swagger/OpenAPI URL", 
    placeholder="http://api-resilixstream.mathesislabs.com/resilixstream/api/v3/api-docs"
)

if st.button("Fetch APIs"):
    parsed = urlparse(swagger_url)
    path_parts = parsed.path.rstrip('/').split('/')
    base_path = "/".join(path_parts[:-1])
    base_url = f"{parsed.scheme}://{parsed.netloc}{base_path}"
    st.session_state.base_url = base_url

    with st.spinner("Fetching Swagger..."):
        spec = fetch_swagger(swagger_url)
        if "error" in spec:
            st.error(spec["error"])
        else:
            st.session_state.spec = spec
            st.session_state.api_paths = list(spec.get("paths", {}).keys())
            st.success("Swagger fetched!")

if st.session_state.api_paths:
    st.subheader("📌 API Endpoints")
    st.write(st.session_state.api_paths)

    test_choice = st.radio("Do you want to test APIs?", ["No", "Yes"])

    if test_choice == "Yes":
        # Step 1: Login first
        if not st.session_state.login_done:
            st.subheader("🔑 Login Required")
            email = st.text_input("Email", value="admin@mathesislabs.com")
            password = st.text_input("Password", type="password", value="Password@123")
            if st.button("Login"):
                success, token = perform_login_test(email, password)
                if success:
                    st.success("Login Success! Token captured.")
                    st.session_state.auth_token = token
                    st.session_state.login_done = True
                else:
                    st.error("Login Failed")

        # Step 2: Run CrewAI only after login
        if st.session_state.login_done:
            if st.button("Run API Tests"):
                # --- Agents ---
                planner_agent = Agent(
                    role="Planner",
                    goal="Decide execution order for APIs",
                    backstory="Expert at ordering endpoints logically.",
                    llm=ollama_llm
                )
                testcase_agent = Agent(
                    role="Testcase Generator",
                    goal="Generate testcases",
                    backstory="Covers positive and negative scenarios.",
                    llm=ollama_llm
                )
                reporter_agent = Agent(
                    role="Reporter",
                    goal="Summarize results",
                    backstory="Creates pass/fail reports.",
                    llm=ollama_llm
                )

                # --- Tasks ---
                plan_task = Task(
                    description=f"Order APIs logically: {st.session_state.api_paths[:20]}",
                    agent=planner_agent,
                    expected_output="Ordered list of APIs"
                )
                testcase_task = Task(
                    description="""Generate ONLY JSON testcases:
[
 {"name":"Test","method":"GET","endpoint":"/path","params":{},"data":{},"expected_status":[200]}
]""",
                    agent=testcase_agent,
                    context=[plan_task],
                    expected_output="Strict JSON array"
                )

                # --- Run Planner + Testcase Generator ---
                crew = Crew(
                    agents=[planner_agent, testcase_agent],
                    tasks=[plan_task, testcase_task],
                    verbose=True,
                )
                results = crew.kickoff()

                # --- Collect Outputs ---
                plan_output = get_output_safe(results.tasks_output[0])
                testcase_output = get_output_safe(results.tasks_output[1])

                # --- Parse Testcases ---
                testcases = safe_json_parse(testcase_output)
                if not testcases:
                    st.error("❌ No testcases generated")
                else:
                    st.success(f"✅ {len(testcases)} testcases generated")
                    st.json(testcases)

                    # --- Run Execution ---
                    logs = run_test_cases(testcases, st.session_state.base_url, token=st.session_state.auth_token)
                    st.subheader("📑 Execution Logs")
                    st.json(logs)

                    # --- Reporter Task ---
                    report_task = Task(
                        description=f"Summarize these API test execution results:\n{json.dumps(logs, indent=2)}",
                        agent=reporter_agent,
                        expected_output="JSON summary with counts of passed/failed and endpoint breakdown"
                    )
                    report_crew = Crew(
                        agents=[reporter_agent],
                        tasks=[report_task],
                        verbose=True,
                    )
                    report_results = report_crew.kickoff()
                    summary = get_output_safe(report_results.tasks_output[0])

                    # --- Final Report ---
                    final_report = {
                        "execution_logs": logs,
                        "summary": safe_json_parse(summary) or summary
                    }
                    st.session_state.final_report = final_report

        # Print report on UI if available
        if st.session_state.final_report:
            st.subheader("📊 Final Report")
            print_report_section(st.session_state.final_report)