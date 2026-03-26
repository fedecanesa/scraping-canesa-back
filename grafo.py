# grafo.py
"""
Grafo LangGraph del pipeline: DataEngineer -> Profiler -> Copywriter.
Flujo no conversacional: scrapea web, analiza negocio, genera Cold Email.
"""

import importlib.util
import os
from typing import Any, TypedDict

from langgraph.graph import StateGraph, START, END


# --- Carga de módulos (los archivos tienen espacios en el nombre) ---
_BASE = os.path.dirname(os.path.abspath(__file__))


def _load_agent(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(_BASE, file_path)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


_agente1 = _load_agent("agente1", "agents/Agente 1 - Scraper y Data Engineer.py")
_agente2 = _load_agent("agente2", "agents/Agente 2 - The Profiler.py")
_agente3 = _load_agent("agente3", "agents/Agente 3 - The Copywriter.py")

data_engineer_node = _agente1.data_engineer_node
profiler_node = _agente2.profiler_node
copywriter_node = _agente3.copywriter_node


# --- Estado del grafo ---
class SalesState(TypedDict, total=False):
    run_id: str
    target_url: str
    max_crawl_pages: int
    max_crawl_depth: int
    skip_cleaning: bool
    cleaned_data: list[dict[str, Any]]
    profile_data: str
    my_service_info: str
    company_tone: str
    final_email: str


# --- Construcción del grafo ---
workflow = StateGraph(SalesState)

workflow.add_node("DataEngineer", data_engineer_node)
workflow.add_node("Profiler", profiler_node)
workflow.add_node("Copywriter", copywriter_node)

workflow.add_edge(START, "DataEngineer")
workflow.add_edge("DataEngineer", "Profiler")
workflow.add_edge("Profiler", "Copywriter")
workflow.add_edge("Copywriter", END)

app = workflow.compile()
