# grafo.py

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.copywriter import copywriter_node
from agents.data_engineer import data_engineer_node
from agents.profiler import profiler_node


class SalesState(TypedDict, total=False):
    run_id: str
    target_url: str
    max_crawl_pages: int
    max_crawl_depth: int
    skip_cleaning: bool

    cleaned_data: list[dict[str, Any]]
    profile_data: dict[str, Any]
    my_service_info: str
    company_tone: str
    final_email: str


workflow = StateGraph(SalesState)

workflow.add_node("DataEngineer", data_engineer_node)
workflow.add_node("Profiler", profiler_node)
workflow.add_node("Copywriter", copywriter_node)

workflow.add_edge(START, "DataEngineer")
workflow.add_edge("DataEngineer", "Profiler")
workflow.add_edge("Profiler", "Copywriter")
workflow.add_edge("Copywriter", END)

app = workflow.compile()