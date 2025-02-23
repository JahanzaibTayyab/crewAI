import os

from crewai import LLM, Agent, Crew, Process, Task
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource
from crewai.project import CrewBase, agent, crew, task

# Get the GEMINI API key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Create a PDF knowledge source
pdf_source = PDFKnowledgeSource(file_paths=["example_home_inspection.pdf"])


# Create an LLM with a temperature of 0 to ensure deterministic outputs
gemini_llm = LLM(
    model="gemini/gemini-1.5-flash",
    api_key=GEMINI_API_KEY,
    temperature=0,
)


@CrewBase
class LatestAiDevelopment:
    """PDF RAG crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def researcherPDF(self) -> Agent:
        return Agent(
            config=self.agents_config["agent"],
            verbose=True,
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
    def research_task_PDF(self) -> Task:
        return Task(
            config=self.tasks_config["research_agent_task"],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the LatestAiDevelopment crew"""

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            knowledge_sources=[pdf_source],
            embedder={
                "provider": "google",
                "config": {
                    "model": "models/text-embedding-004",
                    "api_key": GEMINI_API_KEY,
                },
            },
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
