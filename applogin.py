import streamlit as st
import httpx
import yaml
import json
import re
from urllib.parse import urlparse
from crewai import Agent, Task, Crew, LLM

# ===========================
# LLM Setup with CrewAI LLM
# ===========================
ollama_llm = LLM(
    model="ollama/codellama:7b",
    base_url="http://localhost:11434"
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
    # Create a client with a default authorization header if a token is provided
    headers = {"Authorization": f"Bearer {auth}"} if auth else {}
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

def perform_login_test(base_url: str, email: str, password: str):
    """Perform a specific login POST request with user-provided credentials."""
    login_url = base_url.rstrip("/") + "/api/public/login"
    login_credentials = {
        "email": email,
        "password": password
    }
    
    st.subheader("Performing Login Test...")
    st.write(f"Attempting to log in to: {login_url}")
    st.write(f"Using credentials: {email} / {'*' * len(password)}")

    try:
        response = httpx.post(login_url, json=login_credentials, timeout=10)
        st.write(f"Login API Status Code: **{response.status_code}**")
        
        try:
            response_json = response.json()
            st.write("Login Response:")
            st.json(response_json)
            # Print to console as requested
            print("Login API Response (Console Output):")
            print(json.dumps(response_json, indent=2))
        except json.JSONDecodeError:
            st.error("Login response is not a valid JSON.")
            st.write("Raw Response:")
            st.code(response.text)
            print("Login API Raw Response (Console Output):")
            print(response.text)

        response.raise_for_status()
        st.success("✅ Login successful!")
        return True, response.json().get("token") # Assuming a token is returned
    except httpx.HTTPStatusError as e:
        st.error(f"❌ Login failed! Status code: {e.response.status_code}")
        st.write("Response body:")
        st.code(e.response.text)
        return False, None
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        return False, None
        
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

# Initialize session state for UI control
if 'show_api_list' not in st.session_state:
    st.session_state.show_api_list = False
if 'show_login_section' not in st.session_state:
    st.session_state.show_login_section = False
if 'login_successful' not in st.session_state:
    st.session_state.login_successful = False
if 'auth_token' not in st.session_state:
    st.session_state.auth_token = None

swagger_url = st.text_input(
    "Enter Swagger/OpenAPI URL", 
    placeholder="http://api-resilixstream.mathesislabs.com/resilixstream/api/v3/api-docs"
)

# Main button to start the process
if st.button("Fetch APIs"):
    st.session_state.show_api_list = False
    st.session_state.show_login_section = False
    st.session_state.login_successful = False
    st.session_state.auth_token = None
    
    parsed = urlparse(swagger_url)
    path_parts = parsed.path.rstrip('/').split('/')
    base_path = "/".join(path_parts[:-1])
    base_url = f"{parsed.scheme}://{parsed.netloc}{base_path}"
    st.session_state.base_url = base_url
    st.text(f"Base API URL set to: {base_url}")

    with st.spinner("Fetching API documentation..."):
        spec = fetch_swagger(swagger_url)
        if "error" in spec:
            st.error(f"Failed to fetch spec: {spec['error']}")
        else:
            st.success("Swagger spec fetched successfully!")
            st.session_state.spec = spec
            st.session_state.api_paths = list(spec.get("paths", {}).keys())
            st.session_state.show_api_list = True
            st.session_state.show_login_section = True

# Display API paths and login section after fetching
if st.session_state.get('show_api_list', False):
    with st.expander("Show All API Endpoints"):
        for path in st.session_state.api_paths:
            st.markdown(f"- `{path}`")
    
    st.markdown("---")
    
    if st.session_state.get('show_login_section', False):
        st.subheader("Login to the API")
        user_email = st.text_input("Enter your email", key="login_email")
        user_password = st.text_input("Enter your password", type="password", key="login_password")
        
        if st.button("Test Login Endpoint"):
            if user_email and user_password:
                login_success, token = perform_login_test(st.session_state.base_url, user_email, user_password)
                st.session_state.login_successful = login_success
                if token:
                    st.session_state.auth_token = token
            else:
                st.warning("Please enter both email and password.")

# Run the full test suite with CrewAI
if st.session_state.get('login_successful', False) and st.button("Run All API Tests with CrewAI"):
    with st.spinner("Running CrewAI to generate and execute all API tests..."):
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
        plan_task = Task(
            description=f"Analyze API spec and decide execution order. Endpoints: {st.session_state.api_paths[:20]}...",
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
            # Pass the retrieved auth token for authenticated calls
            logs = run_test_cases(testcases, st.session_state.base_url, auth=st.session_state.auth_token)
    
        # ===========================
        # Final Report
        # ===========================
        final_summary = {
            "execution_logs": logs,
            "report": results.get("Reporter", "No report generated")
        }
        st.subheader("✅ Final Report")
        st.json(final_summary)
