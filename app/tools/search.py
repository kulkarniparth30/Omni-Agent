from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun

# DuckDuckGo general web search tool
search_tool = DuckDuckGoSearchRun(
    name="duckduckgo_search",
    description="Search the web for current events, facts, news, and general information.",
    region="us-en"
)

# Semantic Scholar academic paper search tool
semantic_scholar_tool = SemanticScholarQueryRun()
