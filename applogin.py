import streamlit as st
import httpx
import yaml
import json
import re
from urllib.parse import urlparse
from crewai import Agent, Task, Crew, LLM # Removed LLMTool import

# ===========================
# LLM Setup with Ollama
# ===========================
# NOTE: Assuming LLM is correctly imported/defined for CrewAI
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
        # Attempt to handle YAML, common for OpenAPI specs
        return yaml.safe_load(r.text)
    except Exception as e:
        return {"error": str(e)}

def safe_json_parse(raw_output: str):
    """Safely parse JSON from LLM output, handling markdown fences and extraneous text"""
    # Try direct parsing
    try:
        return json.loads(raw_output)
    except:
        # Regex to find the JSON array structure [ ... ]
        match = re.search(r'\[\s*\{.*\}\s*\]', raw_output, re.S)
        if match:
            try:
                # Attempt to parse the extracted JSON string
                return json.loads(match.group(0))
            except:
                return []
        return [] # Return empty list if all parsing attempts fail

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
                # Common token keys check
                token = data.get("token") or data.get("accessToken")
                return True, token
        except:
            st.error("Response not JSON or no token found in valid response.")
            st.code(resp.text)
            return False, None
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
            # Send the request
            response = client.request(method, url, json=data, params=params)
            # Check if the status code is one of the expected ones
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
    """Safely extract output from crewai task result objects."""
    if hasattr(task_output, "output"):
        return task_output.output
    if hasattr(task_output, "final_output"):
        return task_output.final_output
    # Handle older/other versions or simple strings
    return str(task_output)

def print_report_section(report):
    """Prints the final execution report summary and logs to Streamlit."""
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
            
            st.write("**Breakdown by endpoint:**")
            if "breakdown" in summary:
                for endpoint, stat in summary["breakdown"].items():
                    st.write(f"- `{endpoint}`: {stat}")
            st.write("---")
        else:
            # Handle case where summary is not a dict (e.g., raw text from LLM)
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

# Initialize session state variables
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
if "flow_analysis" not in st.session_state:
    st.session_state.flow_analysis = None # New state for flow report

swagger_url = st.text_input(
    "Enter Swagger/OpenAPI URL", 
    placeholder="http://api-resilixstream.mathesislabs.com/resilixstream/api/v3/api-docs"
)

if st.button("Fetch APIs"):
    # Logic to derive base_url from swagger_url
    parsed = urlparse(swagger_url)
    path_parts = parsed.path.rstrip('/').split('/')
    # Assuming base_path is everything before the last path segment (e.g., /api/v3/api-docs -> /api/v3)
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
            st.session_state.login_done = False # Reset login/flow on new fetch
            st.session_state.flow_analysis = None
            st.session_state.final_report = None
            st.success("Swagger fetched! Now log in to continue.")

# Only show further controls if APIs are fetched
if st.session_state.api_paths:
    st.subheader("📌 API Endpoints Found")
    # Show only a snippet of paths to keep UI clean
    st.markdown(f"Found **{len(st.session_state.api_paths)}** endpoints. Displaying first 5: `{st.session_state.api_paths[:5]}...`")

    # Display the flow analysis if it's been done
    if st.session_state.flow_analysis:
        st.markdown("---")
        st.subheader("✨ Feature Flow Analysis (From Agent)")
        st.markdown(st.session_state.flow_analysis)
        st.markdown("---")

    test_choice = st.radio("Do you want to run the full API test suite?", ["No", "Yes"], index=1 if st.session_state.login_done else 0)

    if test_choice == "Yes":
        # Step 1: Login first
        if not st.session_state.login_done:
            st.subheader("🔑 Login Required")
            email = st.text_input("Email", value="admin@mathesislabs.com", key="login_email")
            password = st.text_input("Password", type="password", value="Password@123", key="login_pass")
            
            if st.button("Login"):
                with st.spinner("Attempting Login..."):
                    success, token = perform_login_test(email, password)
                
                if success:
                    st.success("Login Success! Token captured.")
                    st.session_state.auth_token = token
                    st.session_state.login_done = True

                    # --- AGENT ACTION: Post-Login Flow Analysis ---
                    st.info("Agent is now analyzing the API paths to determine the feature flow...")
                    
                    flow_analyzer_agent = Agent(
                        role="Flow Analyst",
                        goal="Analyze the given API paths to determine the key features and their likely operational flow.",
                        backstory="An expert in reading Swagger/OpenAPI specifications to deduce the primary features and logical user journey (e.g., Create User -> Get User -> Update User).",
                        llm=ollama_llm,
                        verbose=True
                    )

                    flow_task = Task(
                        description=f"""Based on this list of API endpoints: {st.session_state.api_paths}. 
                        Identify the main functional features (e.g., 'User Management', 'Content Streaming', 'Reporting'). 
                        Then, for each feature, outline the logical flow of operations (e.g., POST to /users, then GET to /users/{{id}}, then DELETE to /users/{{id}}).
                        Output a clear, human-readable summary of the key features and their expected API workflow. DO NOT output JSON, just markdown text.""",
                        agent=flow_analyzer_agent,
                        expected_output="A summary report detailing the system's key features and their corresponding API request flow as readable markdown."
                    )

                    flow_crew = Crew(
                        agents=[flow_analyzer_agent],
                        tasks=[flow_task],
                        verbose=False # Set to False to keep main UI output clean
                    )
                    
                    with st.spinner("Running Flow Analysis Crew..."):
                        # Get the flow analysis result
                        flow_results = flow_crew.kickoff()
                        flow_analysis_output = get_output_safe(flow_results)
                        st.session_state.flow_analysis = flow_analysis_output
                    
                    st.success("✅ Flow Analysis Complete. Scroll up to see the report!")
                    # Rerun Streamlit to show the flow analysis in the dedicated section
                    st.experimental_rerun() 

                else:
                    st.error("Login Failed")

        # Step 2: Run CrewAI main tests only after login is done
        if st.session_state.login_done:
            st.subheader("🤖 Ready to Run Tests")
            st.success(f"Authenticated with token: ...{st.session_state.auth_token[-5:]}")
            
            if st.button("Run FULL API Test Suite"):
                st.session_state.final_report = None # Clear previous report
                
                with st.spinner("Running Planning & Testcase Generation Crew..."):
                    # --- Agents ---
                    planner_agent = Agent(
                        role="Planner",
                        goal="Decide execution order for APIs based on the feature flow and API dependencies (e.g., create before get).",
                        backstory="Expert at ordering endpoints logically and safely.",
                        llm=ollama_llm
                    )
                    testcase_agent = Agent(
                        role="Testcase Generator",
                        goal="Generate detailed testcases covering positive (200) and negative (400, 401, 404) scenarios for the ordered APIs.",
                        backstory="A meticulous tester who ensures coverage of all API parameters, generating mock data as needed.",
                        llm=ollama_llm
                    )
                    reporter_agent = Agent(
                        role="Reporter",
                        goal="Summarize and analyze the raw test execution results.",
                        backstory="Creates clear, concise pass/fail reports with quantitative summaries.",
                        llm=ollama_llm
                    )

                    # --- Tasks ---
                    plan_task = Task(
                        description=f"Logically order the full list of APIs: {st.session_state.api_paths}",
                        agent=planner_agent,
                        expected_output="A numbered list of API paths in the suggested execution order, including the method (GET/POST/etc.) for each."
                    )
                    testcase_task = Task(
                        description="""Based on the ordered plan, generate ONLY a strict JSON array of testcase objects. 
                        Each object MUST follow this format, including data and expected_status:
                        [
                         {"name":"Test Create User","method":"POST","endpoint":"/users","data":{"name":"John","email":"j@t.com"},"expected_status":[201]},
                         {"name":"Test Get User (Invalid ID)","method":"GET","endpoint":"/users/999","params":{},"data":{},"expected_status":[404]}
                        ]""",
                        agent=testcase_agent,
                        context=[plan_task],
                        expected_output="Strict JSON array of testcase objects."
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
                st.subheader("🛠️ Generated Plan & Testcases")
                st.info("Plan Output (for context):")
                st.code(plan_output)
                
                testcases = safe_json_parse(testcase_output)
                
                if not testcases:
                    st.error(f"❌ No testcases generated. Raw LLM output: {testcase_output[:500]}...")
                else:
                    st.success(f"✅ {len(testcases)} testcases generated and parsed.")
                    
                    # --- Run Execution ---
                    st.subheader("🏃 Running API Execution")
                    with st.spinner("Executing API Calls..."):
                        logs = run_test_cases(testcases, st.session_state.base_url, token=st.session_state.auth_token)
                    
                    st.success(f"Execution Complete. {len([l for l in logs if l.get('passed')])} Passed / {len([l for l in logs if not l.get('passed')])} Failed.")
                    
                    # --- Reporter Task ---
                    with st.spinner("Generating Final Report..."):
                        report_task = Task(
                            description=f"""Analyze these API test execution results:\n{json.dumps(logs, indent=2)}. 
                            Generate ONLY a JSON object containing a 'summary' key (with total, passed, failed counts, and endpoint breakdown) and the original 'execution_logs' key. 
                            Example: {{"summary":{{"total":10,"passed":8,"failed":2,"breakdown":{{"/api/users": "PASS: 3/5, FAIL: 2/5", ...}}}},"execution_logs":[...]}}""",
                            agent=reporter_agent,
                            expected_output="Strict JSON report object"
                        )
                        report_crew = Crew(
                            agents=[reporter_agent],
                            tasks=[report_task],
                            verbose=False,
                        )
                        report_results = report_crew.kickoff()
                        summary = get_output_safe(report_results)

                        # --- Final Report ---
                        final_report = {
                            "execution_logs": logs,
                            "summary": safe_json_parse(summary) or summary
                        }
                        st.session_state.final_report = final_report

            # Print report on UI if available
            if st.session_state.final_report:
                st.subheader("✅ Final Report")
                print_report_section(st.session_state.final_report)