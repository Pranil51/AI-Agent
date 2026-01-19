from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal
from enum import Enum 
import json

# Query Analysis
class ComplexityLevel(str, Enum):
    SIMPLE = "Simple"
    MODERATE = "Moderate"
    COMPLEX = "Complex"

class QueryParameters(BaseModel):
    """Core parameters extracted from the user query"""
    keywords: List[str] = Field(default=[], description="All relevant keywords and alternatives")
    entities: List[str] = Field(default=[], description="Entities central to the user query")

class SearchComplexity(BaseModel):
    """Assessment of user query complexity"""
    complexity_level: ComplexityLevel = Field(description="Overall complexity assessment")
    multi_faceted: bool = Field(description="Whether user query has multiple aspects")

class QueryRefiner(BaseModel):
    refined_query: str = Field(description="Well-written version of the user query/question with clarity")
class QueryAnalysisResults(BaseModel):
    """Complete structured output for user query analysis"""
    #intent_classification: IntentClassification = Field(description="Classification of user intent")
    query_parameters: QueryParameters = Field(description="Core search parameters")
    search_complexity: SearchComplexity = Field(description="Complexity assessment")
    is_harmful: bool = Field(description="Indicates if the query has potentially harmful or sensitive content")
# Web Search Plan
class SearchQuery(BaseModel):
    """Docstring for SearchQuery"""
    query_id: int = Field(..., description="Unique identifier for the query",)
    query: str = Field(..., description="Optimized search query string")

class SearchPlan(BaseModel):
    """Schema for a web search query plan."""
    search_queries: List[SearchQuery] = Field(..., description="List of search queries to execute", min_items=1)
    @field_validator('search_queries', mode='before')
    @classmethod
    def ensure_list(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                print('Issue parsing JSON string for search_queries')
                return v # Fallback if it's just a single string
        return v

# Database Query Plan
class DBQuery(BaseModel):
    """Schema for a vector database query."""
    query: str = Field(..., description="The database query string for similarity search.")
    n_results: int = Field(5, description="Number of top similar results to retrieve based on query complexity and specificity.", ge=1, le=20)
class DBQueryPlan(BaseModel):
    """Schema for a vector database query plan."""
    queries: List[DBQuery] = Field(..., description="List of database queries to execute.")

class ResponseGeneration(BaseModel):
    response_to_user_query: str = Field(..., description="Response to the user query.")
    gaps_acknowledged: str = Field(..., description="List of information gaps acknowledged/issues in the response.")
# Response Evaluation
class ResponseEvaluation(BaseModel):
    """Schema for evaluating the generated response."""
    response_evaluation : str = Field(..., description="Evaluation of the generated response assessing relevance to user query, accuracy, completeness of information.")
    response_rating: Literal['highly_satisfactory', 'satisfactory', 'unsatisfactory', 'highly unsatisfactory'] = Field(..., description="Evaluation rating of the response")
    action_rationale: str = Field(
        ...,
        description="Strategic reasoning for the next action: identifies what specific information is missing, what data sources or methods are needed to obtain it, and how to proceed. Format as 'X is missing/unclear and needs to be obtained/clarified by Y method/source.' Focus on actionable next steps, not response critique."
    )
    next_step: Literal['finish', 'retriever', 'web_search', 'crawl_contexts'] = Field(..., description="Strategic next action for the agent")

# Context Crawler 
class URLExtraction(BaseModel):
    """Schema for extracting URLs from the given contexts."""
    contexts: List[str] = Field(..., description="List of contexts to extract URLs from.")
    urls: List[str] = Field(..., description="List of URLs extracted from the contexts.")

