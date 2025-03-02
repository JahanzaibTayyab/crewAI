import os

from crewai import LLM, Agent, Crew, Process, Task
from crewai.memory import EntityMemory, LongTermMemory, ShortTermMemory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from crewai.memory.storage.rag_storage import RAGStorage
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, WebsiteSearchTool

# Get the GEMINI API key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = os.environ.get("MODEL")

gemini_llm = LLM(
    model=MODEL,
    api_key=GEMINI_API_KEY,
    temperature=0,
)

google_embedder = {
    "provider": "google",
    "config": {
        "model": "models/text-embedding-004",
        "api_key": GEMINI_API_KEY,
    },
}

search_tool = SerperDevTool()
web_rag_tool = WebsiteSearchTool(
    config=dict(
        llm=dict(
            provider="google",
            config=dict(model=MODEL, api_key=GEMINI_API_KEY),
        ),
        embedder=dict(
            provider="google",
            config=dict(
                model="models/text-embedding-004",
                task_type="retrieval_document",
            ),
        ),
    )
)


@CrewBase
class LastOnsiteClass:
    """LastOnsiteClass crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],
            verbose=True,
            llm=gemini_llm,
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["reporting_analyst"],
            verbose=True,
            llm=gemini_llm,
        )

    # @agent
    # def chatbot(self) -> Agent:
    #     return Agent(
    #         config=self.agents_config["chatbot"],
    #         verbose=True,
    #         llm=gemini_llm,
    #         tools=[search_tool],
    #     )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],
            output_file="research_task_report.md",
        )

    @task
    def reporting_task(self) -> Task:
        return Task(config=self.tasks_config["reporting_task"], output_file="report.md")

    # @task
    # def research_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config["chat_task"],
    #     )

    @crew
    def crew(self) -> Crew:
        """Creates the LastOnsiteClass crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            memory=True,
            embedder=google_embedder,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
            # Short-term memory for current context using RAG
            # short_term_memory=ShortTermMemory(
            #     storage=RAGStorage(
            #         embedder_config=google_embedder,
            #         type="short_term",
            #         path="./memory/short_term/",
            #     )
            # ),
            # Long-term memory for persistent storage across sessions
            # long_term_memory=LongTermMemory(
            #     storage=LTMSQLiteStorage(
            #         db_path="./memory/long_term/long_term_memory_storage.db"
            #     )
            # ),
            # Entity memory for tracking key information about entities
            # storage=RAGStorage(
            #     embedder_config=google_embedder,
            #     type="short_term",
            #     path="./memory/EntityMemory/",
            # ),
        )
