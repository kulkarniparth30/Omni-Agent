from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun

# Optimized DuckDuckGo general web search with max_results=3 for fast response times
ddg_wrapper = DuckDuckGoSearchAPIWrapper(max_results=3, time="y")
search_tool = DuckDuckGoSearchRun(
    api_wrapper=ddg_wrapper,
    name="duckduckgo_search",
    description="Search the web for current events, facts, news, and general information.",
    region="us-en"
)

# Semantic Scholar academic paper search tool
semantic_scholar_tool = SemanticScholarQueryRun()
