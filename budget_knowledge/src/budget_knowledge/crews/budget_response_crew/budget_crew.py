import os
from typing import Any, Dict

from crewai import LLM, Agent, Crew, Process, Task
from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource
from crewai.project import CrewBase, agent, crew, task

# Create a PDF knowledge source
union_data = PDFKnowledgeSource(file_paths=["Union_Budget_Analysis-2023-24.pdf"])

# Get the GEMINI API key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = os.environ.get("MODEL")

gemini_llm = LLM(
    model=MODEL,
    api_key=GEMINI_API_KEY,
    temperature=0,
)


@CrewBase
class UnionBudgetCrew:
    """UnionBudget Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def union_budget_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["union_budget_agent"],
            knowledge_sources=[union_data],
            llm=gemini_llm,
            embedder={
                "provider": "google",
                "config": {
                    "model": "models/text-embedding-004",
                    "api_key": GEMINI_API_KEY,
                },
            },
        )

    @task
    def union_budget_query_tasks(self) -> Task:
        return Task(
            config=self.tasks_config["union_budget_query_tasks"],
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
