import os

from crewai import LLM, Agent, Crew, Process, Task
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource

# Create a PDF knowledge source
pdf_search_tool = PDFKnowledgeSource(file_paths=["example_home_inspection.pdf"])

# Get the GEMINI API key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = os.environ.get("MODEL")

gemini_llm = LLM(
    model=MODEL,
    api_key=GEMINI_API_KEY,
    temperature=0,
)

# --- Agents ---
research_agent = Agent(
    role="Research Agent",
    goal="Search through the PDF to find relevant answers",
    allow_delegation=False,
    verbose=True,
    backstory=(
        """
        The research agent is adept at searching and 
        extracting data from documents, ensuring accurate and prompt responses.
        """
    ),
    knowledge_sources=[pdf_search_tool],
    llm=gemini_llm,
    embedder={
        "provider": "google",
        "config": {
            "model": "models/text-embedding-004",
            "api_key": GEMINI_API_KEY,
        },
    },
)

professional_writer_agent = Agent(
    role="Professional Writer",
    goal="Write professional emails based on the research agent's findings",
    allow_delegation=False,
    verbose=True,
    backstory=(
        """
        The professional writer agent has excellent writing skills and is able to craft 
        clear and concise emails based on the provided information.
        """
    ),
    tools=[],
    llm=gemini_llm,
)


# --- Tasks ---
answer_customer_question_task = Task(
    description=(
        """
        Answer the customer's questions based on the home inspection PDF.
        The research agent will search through the PDF to find the relevant answers.
        Your final answer MUST be clear and accurate, based on the content of the home
        inspection PDF.

        Here is the customer's question:
        {customer_question}
        """
    ),
    expected_output="""
        Provide clear and accurate answers to the customer's questions based on 
        the content of the home inspection PDF.
        """,
    #  tools=[pdf_search_tool],
    agent=research_agent,
)

write_email_task = Task(
    description=(
        """
        Write a professional email to a contractor based on the research agent's findings.
        The email should clearly state the issues found in the specified section of the report
        and request a quote or action plan for fixing these issues.

        Ensure the email is signed with the following details:
        
        Best regards,

        Jahanzaib Tayyab,
        """
    ),
    expected_output="""
        Write a clear and concise email that can be sent to a contractor to address the 
        issues found in the home inspection report.
        """,
    tools=[],
    agent=professional_writer_agent,
)

# --- Crew ---
crew = Crew(
    tasks=[answer_customer_question_task, write_email_task],
    agents=[research_agent, professional_writer_agent],
    process=Process.sequential,
    verbose=True,
)

customer_question = input(
    "Which section of the report would you like to generate a work order for?\n"
)


def kickoff():
    result = crew.kickoff(inputs={"customer_question": customer_question})
    print(result)


if __name__ == "__main__":
    kickoff()
