import streamlit as st
import httpx
import yaml
from crewai import Agent, Task, Crew

planner_agent = Agent(
    role="Planner",
    goal="Plan which API endpoints to test in correct order",
    backstory="Specialist in analyzing Swagger/OpenAPI and planning API execution.",
)

testcase_agent = Agent(
    role="Testcase Generator",
    goal="Generate positive and negative testcases for each endpoint",
    backstory="Expert in writing robust test cases covering happy path and failures.",
)

executor_agent = Agent(
    role="Executor",
    goal="Execute test cases and record results",
    backstory="Runs APIs using provided input and captures responses.",
)

reporter_agent = Agent(
    role="Reporter",
    goal="Generate summary of test results in JSON format",
    backstory="Expert in reporting test results with details of pass/fail.",
)


def fetch_swagger(url: str):
    try:
        r = httpx.get(url, timeout=10)
        r.raise_for_status()
        return r.json() if r.headers.get("content-type","").startswith("application/json") else yaml.safe_load(r.text)
    except Exception as e:
        return {"error": str(e)}


st.title("🚀 CrewAI API Tester")

swagger_url = st.text_input("Enter Swagger/OpenAPI URL", placeholder="https://petstore.swagger.io/v2/swagger.json")
username = st.text_input("Username (if required)")
password = st.text_input("Password (if required)", type="password")

if st.button("Start Testing"):
    with st.spinner("Fetching API documentation..."):
        spec = fetch_swagger(swagger_url)
        if "error" in spec:
            st.error(f"Failed to fetch spec: {spec['error']}")
        else:
            st.success("Swagger spec fetched!")

        
            plan_task = Task(
                description=f"Analyze the API spec and decide execution order. Spec: {str(spec)[:2000]}...",
                agent=planner_agent,
            )

            testcase_task = Task(
                description="Generate positive and negative test cases for planned endpoints.",
                agent=testcase_agent,
            )

            execute_task = Task(
                description=f"Run test cases using httpx. Credentials: username={username}, password={password}",
                agent=executor_agent,
            )

            report_task = Task(
                description="Generate structured JSON report of test execution with pass/fail details.",
                agent=reporter_agent,
            )

           
            crew = Crew(
                agents=[planner_agent, testcase_agent, executor_agent, reporter_agent],
                tasks=[plan_task, testcase_task, execute_task, report_task],
                verbose=True,
            )

            result = crew.kickoff()

            st.subheader("✅ Final Report")
            st.json(result)
