import os

from crewai import LLM, Agent, Crew, Process, Task
from crewai.memory import EntityMemory, LongTermMemory, ShortTermMemory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from crewai.memory.storage.rag_storage import RAGStorage
from crewai.project import CrewBase, agent, crew, task

# Importing crewAI tools
from crewai_tools import SerperDevTool, WebsiteSearchTool

# Get the GEMINI API key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = os.environ.get("MODEL")

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


gemini_llm = LLM(
    model=MODEL,
    api_key=GEMINI_API_KEY,
    temperature=0,
)


@CrewBase
class AugmentedLlms:
    """AugmentedLlms crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def chatbot(self) -> Agent:
        return Agent(
            config=self.agents_config["chatbot"],
            verbose=True,
            tools=[search_tool, web_rag_tool],
            llm=gemini_llm,
            embedder=google_embedder,
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["chat_task"],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the AugmentedLlms crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            memory=True,
            # Long-term memory for persistent storage across sessions
            long_term_memory=LongTermMemory(
                storage=LTMSQLiteStorage(
                    db_path="./memory/long_term/long_term_memory_storage.db"
                )
            ),
            # Short-term memory for current context using RAG
            short_term_memory=ShortTermMemory(
                storage=RAGStorage(
                    embedder_config=google_embedder,
                    type="short_term",
                    path="./memory/short_term/",
                )
            ),
            entity_memory=EntityMemory(
                storage=RAGStorage(
                    embedder_config=google_embedder,
                    type="short_term",
                    path="./memory/entity/",
                )
            ),
        )
