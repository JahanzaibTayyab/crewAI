import os

from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool

tool = SerperDevTool()

# Get the GEMINI API key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = os.environ.get("MODEL")

gemini_llm = LLM(
    model=MODEL,
    api_key=GEMINI_API_KEY,
    temperature=0,
)


@CrewBase
class InternetCrew:
    """Internet Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def internet_search_agent(self) -> Agent:
        return Agent(config=self.agents_config["internet_search_agent"], llm=gemini_llm)

    @task
    def internet_search_query_tasks(self) -> Task:
        return Task(
            config=self.tasks_config["internet_search_query_tasks"], tools=[tool]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Internet Search Crew"""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
