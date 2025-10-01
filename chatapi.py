import streamlit as st
import httpx
import json
from crewai import Agent, Task, Crew
from crewai_tools import LLM # Correct import for LLM configuration

# ===========================
# LLM Setup with Ollama
# ===========================
# NOTE: Ensure Ollama is running at http://localhost:11434 with the 'codellama:7b' model
try:
    ollama_llm = LLM(
        model="ollama/codellama:7b",
        base_url="http://localhost:11434"  # Default Ollama server
    )
except Exception as e:
    st.error(f"Error initializing Ollama LLM: {e}")
    st.warning("Please ensure Ollama server is running and the 'codellama:7b' model is available.")
    # Use a placeholder object if initialization fails to allow the UI to load
    ollama_llm = None 


# ===========================
# Helpers
# ===========================
def run_test_cases(testcases: list, base_url: str):
    """Run API test cases using httpx and return logs."""
    logs = []
    # Use a try-except block around client creation/request to catch connection errors
    try:
        # NOTE on Timeout: A low timeout (e.g., 5 seconds) is good for local testing
        client = httpx.Client(timeout=5.0) 
    except Exception as e:
        # This catches errors like invalid URL format in base_url before request
        logs.append({"error": f"HTTPX Client Initialization Error: {str(e)}", "passed": False})
        return logs

    for tc in testcases:
        method = tc.get("method", "GET").upper()
        endpoint = tc.get("endpoint", "/")
        
        # Safely construct the URL, replacing path variables
        url = base_url.rstrip("/") + endpoint.replace("{chatId}", tc.get("chatId", "123")) 
        
        data = tc.get("data", {})
        params = tc.get("params", {})
        expected_status = tc.get("expected_status", [200])

        try:
            response = client.request(method, url, json=data, params=params)
            
            # Use response.status_code check for 'passed' status
            passed = response.status_code in expected_status
            
            logs.append({
                "name": tc.get("name", "Unnamed"),
                "endpoint": endpoint,
                "method": method,
                "status_code": response.status_code,
                "expected_status": expected_status,
                "passed": passed,
                "response": response.text[:200]  # limit response preview
            })
        except httpx.ConnectError as e:
            # Explicitly catch connection errors (e.g., server not running)
            logs.append({
                "name": tc.get("name", "Unnamed"),
                "endpoint": endpoint,
                "method": method,
                "error": f"Connection Error: {str(e)}",
                "passed": False
            })
        except Exception as e:
            # Catch all other exceptions (e.g., timeout, protocol errors)
            logs.append({
                "name": tc.get("name", "Unnamed"),
                "endpoint": endpoint,
                "method": method,
                "error": f"Request Error: {str(e)}",
                "passed": False
            })
    client.close()
    return logs


# ===========================
# Streamlit UI
# ===========================
st.set_page_config(layout="wide")
st.title("🚀 CrewAI Chat API Tester PoC")
st.markdown("---")

base_url = st.text_input(
    "1. Enter Base API URL for Chat Service", 
    placeholder="http://localhost:8080"
)

if st.button("2. Start Testing") and base_url:
    
    if ollama_llm is None:
        st.error("Cannot run testing. Please fix the Ollama LLM initialization error first.")
        st.stop()
        
    st.info(f"Starting test run for base URL: {base_url}")
    
    # ===========================
    # Agents
    # ===========================
    with st.spinner("Initializing Agents..."):
        api_extractor_agent = Agent(
            role="API Extractor",
            goal="Provide all chat APIs",
            backstory="Knows the chat service endpoints.",
            llm=ollama_llm,
            verbose=True
        )

        planner_agent = Agent(
            role="Planner",
            goal="Ensure execution starts with POST /chat/message",
            backstory="Knows dependencies between chat APIs.",
            llm=ollama_llm,
            verbose=True
        )

        tester_agent = Agent(
            role="Tester",
            goal="Generate and run predefined testcases for chat APIs",
            backstory="Executes both positive and negative cases.",
            llm=ollama_llm,
            verbose=True
        )

        reporter_agent = Agent(
            role="Reporter",
            goal="Summarize the final API execution results into a concise JSON report.",
            backstory="Generates final report with pass/fail summary and execution logs.",
            llm=ollama_llm,
            verbose=True
        )

    # ===========================
    # Tasks and Testcases
    # ===========================
    predefined_testcases = [
        {"name": "Create new chat message", "method": "POST", "endpoint": "/chat/message", "data": {"prompt": "Hello from test"}, "expected_status": [200]},
        {"name": "Get chat history", "method": "GET", "endpoint": "/chat/history", "expected_status": [200]},
        {"name": "Load chat by ID", "method": "GET", "endpoint": "/chat/{chatId}", "chatId": "123", "expected_status": [200, 404]},
        {"name": "Rename chat", "method": "PUT", "endpoint": "/chat/rename/{chatId}", "data": {"title": "Renamed Chat"}, "chatId": "123", "expected_status": [200, 404]},
        {"name": "Delete chat", "method": "DELETE", "endpoint": "/chat/{chatId}", "chatId": "123", "expected_status": [200, 404]}
    ]

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

    # The expected output is the list of test cases, which the agent will confirm
    testcase_task = Task(
        description="Review and use the predefined chat API testcases. The final output must be the JSON string of the test cases.",
        agent=tester_agent,
        context=[plan_task],
        expected_output="JSON string of testcases" # Changed for better agent instruction
    )

    report_task = Task(
        description=f"Summarize the final API test results from the logs into a JSON report. The test execution logs are: {json.dumps(predefined_testcases)}. The final JSON report must include a 'summary_status' (e.g., PASSED, FAILED) and the 'execution_logs' array, following the structure provided in the initial agent thinking.",
        agent=reporter_agent,
        context=[testcase_task],
        expected_output="JSON summary"
    )

    # ===========================
    # Crew Execution
    # ===========================
    with st.spinner("Running CrewAI Agents (Extracting, Planning, Testing, Reporting)..."):
        crew = Crew(
            agents=[api_extractor_agent, planner_agent, tester_agent, reporter_agent],
            tasks=[extract_task, plan_task, testcase_task, report_task],
            verbose=False, # Set to False for cleaner Streamlit output, agents are verbose=True
        )

        results = crew.kickoff()
        
    st.subheader("🤖 CrewAI Agent Outputs")
    # Convert CrewOutput into dict safely (handling the V2 deprecation)
    results_dict = results.model_dump() if hasattr(results, "model_dump") else {}
    st.json(results_dict)


    # ===========================
    # Execute Real API Testcases
    # ===========================
    st.subheader("📡 Real API Test Execution")
    with st.spinner(f"Running real HTTP requests against {base_url} ..."):
        logs = run_test_cases(predefined_testcases, base_url)
        
        # Calculate overall pass/fail status
        overall_status = "PASSED"
        if not logs:
             overall_status = "ERROR: No logs generated"
        elif any(not log.get("passed", False) for log in logs if 'error' not in log):
            overall_status = "FAILED (Some tests failed)"
        elif any('error' in log for log in logs):
            overall_status = "FAILED (Connection/Request Errors)"
            
        st.table(logs) # Display execution logs in a clean table

    # ===========================
    # Display Final Report
    # ===========================
    st.subheader("✅ Final PoC Report (Aggregated)")
    final_summary = {
        "execution_status": overall_status,
        "base_api_url": base_url,
        "real_api_execution_logs": logs,
        "agent_report_summary": results_dict.get("Reporter", "No agent report generated")
    }

    # Display the final JSON summary
    st.json(final_summary)

    st.balloons()