from crewai import Agent, Task, Crew
from llm_setup import ollama_llm

def create_agents():
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
    return planner_agent, testcase_agent, reporter_agent

def create_tasks(api_paths, planner_agent, testcase_agent):
    plan_task = Task(
        description=f"Order APIs logically: {api_paths[:20]}",
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
    return plan_task, testcase_task
