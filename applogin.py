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
        match = re.search(r'\[\s*\{.*\}\s*\]', raw_output, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                return []
        return []

def perform_login_test(email: str, password: str):
    """Hit the login endpoint and extract token"""
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
            st.json(data)  # Debug: Show full response
            if resp.status_code == 200:
                # Try all possible token field names
                token = (
                    data.get("token") or
                    data.get("accessToken") or
                    data.get("jwt") or
                    data.get("idToken")
                )
                return True, token
        except:
            st.error("Response not JSON or no token found.")
            st.code(resp.text)
            return False, None
        return False, None
    except Exception as e:
        st.error(f"Login request failed: {e}")
        return False, None

def run_test_cases(testcases: list, base_url: str, token=None):
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
            expected_statuses = tc.get("expected_status", [200])
            passed = response.status_code in expected_statuses
            
            logs.append({
                "name": tc.get("name", "Unnamed"),
                "endpoint": endpoint,
                "method": method,
                "status_code": response.status_code,
                "expected": expected_statuses,
                "passed": passed,
                "response_body_snippet": response.text[:200]
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
            st.write(f"**Total Tests:** {total if total else 'N/A'}")
            st.write(f"**Passed:** {passed if passed is not None else 'N/A'}")
            st.write(f"**Failed:** {failed if failed is not None else 'N/A'}")
            
            if "breakdown" in summary:
                st.write("**Breakdown by endpoint:**")
                for endpoint, stat in summary["breakdown"].items():
                    st.write(f"- `{endpoint}`: {stat}")
            st.write("---")
        else:
            st.markdown(report.get("summary", "No detailed summary available."))
            st.write("---")
        st.write("### 📝 Execution Logs")
        st.json(report["execution_logs"])
    else:
        st.write(report)

# ===========================
# Streamlit UI
# ===========================
st.title("🚀 CrewAI API Tester")

# Init session state
for key, default in {
    "spec": None, "api_paths": [], "base_url": None,
    "auth_token": None, "login_done": False,
    "final_report": None, "flow_analysis": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

swagger_url = st.text_input(
    "Enter Swagger/OpenAPI URL", 
    placeholder="http://api-resilixstream.mathesislabs.com/resilixstream/api/v3/api-docs"
)

if st.button("Fetch APIs"):
    parsed = urlparse(swagger_url)
    path_parts = parsed.path.rstrip('/').split('/')
    base_path = "/".join(path_parts[:-1]) 
    st.session_state.base_url = f"{parsed.scheme}://{parsed.netloc}{base_path}"

    with st.spinner("Fetching Swagger..."):
        spec = fetch_swagger(swagger_url)
        if "error" in spec:
            st.error(spec["error"])
        else:
            st.session_state.spec = spec
            st.session_state.api_paths = list(spec.get("paths", {}).keys())
            st.session_state.login_done = False
            st.session_state.flow_analysis = None
            st.session_state.final_report = None
            st.success("Swagger fetched! Now log in to continue.")

if st.session_state.api_paths:
    st.subheader("📌 API Endpoints Found")
    st.markdown(f"Found **{len(st.session_state.api_paths)}** endpoints. Displaying first 5: `{st.session_state.api_paths[:5]}...`")

    if st.session_state.flow_analysis:
        st.subheader("✨ Feature Flow Analysis (From Agent)")
        st.markdown(st.session_state.flow_analysis)
        st.markdown("---")

    test_choice = st.radio("Do you want to run the full API test suite?", ["No", "Yes"], index=1 if st.session_state.login_done else 0)

    if test_choice == "Yes":
        if not st.session_state.login_done:
            st.subheader("🔑 Login Required")
            email = st.text_input("Email", value="admin@mathesislabs.com", key="login_email")
            password = st.text_input("Password", type="password", value="Password@123", key="login_pass")
            
            if st.button("Login"):
                with st.spinner("Attempting Login..."):
                    success, token = perform_login_test(email, password)
                if success and token:
                    st.success("Login Success! Token captured.")
                    st.session_state.auth_token = token
                    st.session_state.login_done = True
                    st.experimental_rerun()
                else:
                    st.error("Login Failed or token missing.")

        if st.session_state.login_done:
            st.subheader("🤖 Ready to Run Tests")
            if st.session_state.auth_token:
                st.success(f"Authenticated with token: ...{st.session_state.auth_token[-5:]}")
            else:
                st.warning("Login done but no token available.")

            if st.button("Run FULL API Test Suite"):
                st.session_state.final_report = None

                with st.spinner("Running Planning & Testcase Generation Crew..."):
                    planner_agent = Agent(
                        role="Planner",
                        goal="Decide execution order for APIs.",
                        backstory="Expert at ordering endpoints logically.",
                        llm=ollama_llm
                    )
                    testcase_agent = Agent(
                        role="Testcase Generator",
                        goal="Generate testcases for APIs.",
                        backstory="Meticulous tester ensuring coverage.",
                        llm=ollama_llm
                    )
                    reporter_agent = Agent(
                        role="Reporter",
                        goal="Summarize results.",
                        backstory="Clear pass/fail reports.",
                        llm=ollama_llm
                    )

                    plan_task = Task(
                        description=f"Order APIs logically: {st.session_state.api_paths}",
                        agent=planner_agent,
                        expected_output="A numbered list of API paths in suggested order."
                    )
                    testcase_task = Task(
                        description="""Generate ONLY JSON array of testcase objects.
                        Format: [
                          {"name":"Test Create User","method":"POST","endpoint":"/users","data":{"name":"John"},"expected_status":[201]},
                          {"name":"Test Get User","method":"GET","endpoint":"/users/1","expected_status":[200]}
                        ]""",
                        agent=testcase_agent,
                        context=[plan_task],
                        expected_output="Strict JSON array of testcases."
                    )

                    crew = Crew(agents=[planner_agent, testcase_agent], tasks=[plan_task, testcase_task], verbose=True)
                    results = crew.kickoff()

                plan_output = get_output_safe(results.tasks_output[0])
                testcase_output = get_output_safe(results.tasks_output[1])

                st.subheader("🛠️ Generated Plan & Testcases")
                st.info("Plan Output:")
                st.code(plan_output)

                testcases = safe_json_parse(testcase_output)
                if not testcases:
                    st.error(f"❌ No testcases generated. Raw LLM output: {testcase_output[:500]}...")
                else:
                    st.success(f"✅ {len(testcases)} testcases generated.")

                    st.subheader("🏃 Running API Execution")
                    with st.spinner("Executing API Calls..."):
                        logs = run_test_cases(testcases, st.session_state.base_url, token=st.session_state.auth_token)

                    st.success(f"Execution Complete. {len([l for l in logs if l.get('passed')])} Passed / {len([l for l in logs if not l.get('passed')])} Failed.")

                    with st.spinner("Generating Final Report..."):
                        report_task = Task(
                            description=f"""Analyze results:\n{json.dumps(logs, indent=2)}.
                            Generate ONLY JSON with 'summary' and 'execution_logs'.""",
                            agent=reporter_agent,
                            expected_output="Strict JSON report object"
                        )
                        report_crew = Crew(agents=[reporter_agent], tasks=[report_task], verbose=False)
                        report_results = report_crew.kickoff()
                        summary = get_output_safe(report_results)

                        final_report = {
                            "execution_logs": logs,
                            "summary": safe_json_parse(summary) or summary
                        }
                        st.session_state.final_report = final_report

            if st.session_state.final_report:
                st.subheader("✅ Final Report")
                print_report_section(st.session_state.final_report)
