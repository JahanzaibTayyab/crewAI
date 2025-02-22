import json
import os

from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel

# Get the GEMINI API key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = os.environ.get("MODEL")

gemini_llm = LLM(
    model=MODEL,
    api_key=GEMINI_API_KEY,
    temperature=0,
)


class UnionBudgetCheck(BaseModel):
    union_budget_flag: str


@CrewBase
class UnionBudgetCheckCrew:
    """UnionBudget Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def budget_query_agent(self) -> Agent:
        return Agent(config=self.agents_config["budget_query_agent"], llm=gemini_llm)

    @task
    def classify_budget_query_tasks(self) -> Task:
        return Task(
            config=self.tasks_config["classify_budget_query_tasks"],
            output_pydantic=UnionBudgetCheck,
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Research Crew"""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
