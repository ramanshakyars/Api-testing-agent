import httpx
import yaml
import json
import re
import streamlit as st

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
    """Works across crewai versions"""
    if hasattr(task_output, "output"):
        return task_output.output
    if hasattr(task_output, "final_output"):
        return task_output.final_output
    return str(task_output)

def print_report_section(report):
    """Render final report in Streamlit"""
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
