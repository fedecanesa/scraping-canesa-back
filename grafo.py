# grafo.py

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.copywriter import copywriter_node
from agents.data_engineer import data_engineer_node
from agents.jobs_agent import jobs_node
from agents.profiler import profiler_node
from agents.reviews_agent import reviews_node


class SalesState(TypedDict, total=False):
    run_id: str
    target_url: str
    max_crawl_pages: int
    max_crawl_depth: int
    skip_cleaning: bool
    objective: str          # "sell" | "partnership"
    user_type: str          # "marketing_agency" | "dev_agency" | "other"

    # Agent 1 → 2 → 3
    cleaned_data: list[dict[str, Any]]

    # Agent: ReviewsAgent
    reviews_data: list[dict[str, Any]]
    business_name: str

    # Agent: JobsAgent
    jobs_data: dict[str, Any]   # {"jobs": [...], "hiring_summary": "..."}

    # Agent 2 → 3
    profile_data: dict[str, Any]

    # User inputs for Agent 3
    my_service_info: str
    company_tone: str

    # Final output
    final_email: str
    message_variants: list[dict[str, str]]


workflow = StateGraph(SalesState)

workflow.add_node("DataEngineer", data_engineer_node)
workflow.add_node("ReviewsAgent", reviews_node)
workflow.add_node("JobsAgent", jobs_node)
workflow.add_node("Profiler", profiler_node)
workflow.add_node("Copywriter", copywriter_node)

workflow.add_edge(START, "DataEngineer")
workflow.add_edge("DataEngineer", "ReviewsAgent")
workflow.add_edge("ReviewsAgent", "JobsAgent")
workflow.add_edge("JobsAgent", "Profiler")
workflow.add_edge("Profiler", "Copywriter")
workflow.add_edge("Copywriter", END)

app = workflow.compile()
