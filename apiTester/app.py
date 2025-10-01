import streamlit as st
import json
from urllib.parse import urlparse
from crewai import Crew, Task
from helpers import fetch_swagger, perform_login_test, run_test_cases, safe_json_parse, get_output_safe, print_report_section
from crew_setup import create_agents, create_tasks
from llm_setup import ollama_llm

st.title("🚀 CrewAI API Tester")

# Session state
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

# Input
swagger_url = st.text_input(
    "Enter Swagger/OpenAPI URL", 
    placeholder="http://api-resilixstream.mathesislabs.com/resilixstream/api/v3/api-docs"
)

# Fetch APIs
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

# Show endpoints
if st.session_state.api_paths:
    st.subheader("📌 API Endpoints")
    st.write(st.session_state.api_paths)

    test_choice = st.radio("Do you want to test APIs?", ["No", "Yes"])

    if test_choice == "Yes":
        # Step 1: Login
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

        # Step 2: Run CrewAI
        if st.session_state.login_done:
            if st.button("Run API Tests"):
                planner_agent, testcase_agent, reporter_agent = create_agents()
                plan_task, testcase_task = create_tasks(st.session_state.api_paths, planner_agent, testcase_agent)

                crew = Crew(
                    agents=[planner_agent, testcase_agent],
                    tasks=[plan_task, testcase_task],
                    verbose=True,
                )
                results = crew.kickoff()

                plan_output = get_output_safe(results.tasks_output[0])
                testcase_output = get_output_safe(results.tasks_output[1])

                testcases = safe_json_parse(testcase_output)
                if not testcases:
                    st.error("❌ No testcases generated")
                else:
                    st.success(f"✅ {len(testcases)} testcases generated")
                    st.json(testcases)

                    logs = run_test_cases(testcases, st.session_state.base_url, token=st.session_state.auth_token)
                    st.subheader("📑 Execution Logs")
                    st.json(logs)

                    # Reporter
                    report_task = Task(
                        description=f"Summarize these API test execution results:\n{json.dumps(logs, indent=2)}",
                        agent=reporter_agent,
                        expected_output="JSON summary with counts of passed/failed and endpoint breakdown"
                    )
                    report_crew = Crew(agents=[reporter_agent], tasks=[report_task], verbose=True)
                    report_results = report_crew.kickoff()
                    summary = get_output_safe(report_results.tasks_output[0])

                    final_report = {
                        "execution_logs": logs,
                        "summary": safe_json_parse(summary) or summary
                    }
                    st.session_state.final_report = final_report

        # Step 3: Show Final Report
        if st.session_state.final_report:
            st.subheader("📊 Final Report")
            print_report_section(st.session_state.final_report)
