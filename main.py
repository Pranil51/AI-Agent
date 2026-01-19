import logging
logger = logging.getLogger(__name__)
handler = logging.FileHandler('logs/test.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

from typing import Dict, List, Set
from langgraph.graph import  END, MessagesState
from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from prompts import *
from schemas import *
from utils import *

load_dotenv()
embed_model = load_embed_model()
model = load_llm()
vector_store = Chroma(
        collection_name="web_data_collection", 
        embedding_function = embed_model,
        persist_directory="./chroma_db_cosine", 
        collection_metadata={"hnsw:space": "cosine"})
class CustomState(MessagesState):
    user_request: str
    refined_user_request: str
    user_request_analysis: Dict 
    search_queries: List 
    sources_data: List
    relevant_links: Set 
    contexts: List 
    retrieved_docs: List 
    visited_links: Set 
    
    action_rationale: str = ""
    next_step: str = ""
    crawl_depth: int = 1
    max_crawl_depth: int = 3
    iteration: int = 0
    max_iterations: int = 5
# TODO: Add Reddit toolnode
# TODO: Add instructions to avoid Reddit, discord, qoura
def user_query_analyzer(state: CustomState, config: RunnableConfig=None) -> CustomState:
    
    """Analyze the user query and create structured analysis."""
    
    user_request_analysis = model.with_structured_output(QueryAnalysisResults).invoke(
        [SystemPrompts.user_request_analyser,
        HumanMessage(f"""## User Query: 
                    {state['user_request']}
                    """)
        ])
    # state['refined_user_request'] = user_request_analysis.refined_query
    state['user_request_analysis'] = user_request_analysis
    if user_request_analysis.is_harmful == True:
        state['messages'].append(AIMessage(content="The query has been identified as potentially harmful or sensitive. Therefore, no further processing will be done."))
        return state
    state['messages'].append(HumanMessage(content=state['user_request']))
    logger.info(f"**user_query_analyzer**=> Refined User Request: {state['user_request']}")
    return state

def search_query_planner(state: CustomState, config: RunnableConfig=None):
    
    """Plan the search queries based on user input."""
    
    previous_queries = state.get('search_queries', "N/A")
    action_rationale = state.get('action_rationale', "N/A")
    planner = model.with_structured_output(SearchPlan, include_raw=True)
    output = planner.invoke(
        [SystemPrompts.search_query_planner,
        HumanMessage(f"""## User Request:
            {state['user_request']}

            ## Previous Queries for Reference(if any):
            {previous_queries}

            ## Action Rationale (if any):
            {action_rationale}""")])
    if output['parsing_error']:
        logger.error(f"**search_query_planner**=> Parsing Error: {output}")
        raise ValueError("Error parsing search queries from the model output.")
    else:
        query_data = output['parsed']    
    search_queries = query_data.model_dump()['search_queries']
    state['search_queries'].extend(search_queries)
    logger.info(f"**search_query_planner**=> Planned {len(state['search_queries'])} search queries. Queries: {[q['query'] for q in search_queries]}")
    return state

def web_search(state: CustomState, config: RunnableConfig=None):
    
    """Retrieve URLs using search queries."""
    
    web_queries = state['search_queries']
    all_sources = []
    for query_item in web_queries:
        if query_item.get('search_performed') ==True:
            continue
        query = query_item['query']
        search_data = retrieve_search_results(query=query )
        query_item['search_performed'] = True
        for item in search_data:
            item['metadata']['query_id'] = query_item['query_id']
        all_sources.extend(search_data)
    state['sources_data'].extend(all_sources)
    logger.info(f"**web_search**=> Total URL retrieved from Web: {len(all_sources)}")
    return state

# sources_data: list[sources] 
# sources: dict[metadata:{'source': str, 'link': str, 'date': str, 'sitelinks': dict, }, ]
# TODO: load urls in parallel and adjust the for loops => done
import asyncio
def data_extracter(state: CustomState, config: RunnableConfig =None):
    
    """Load, process and Store data from the URLs by filtering relevant content."""
    
    text_splitter = AdvancedMarkdownSplitter(chunk_size=5000, chunk_overlap=50)    
    links_to_scrape = [source['metadata']['link'] for source in state['sources_data'] if 'link' in source['metadata'] and source['metadata']['link'] not in state['visited_links']]
    
    # Ethical scrapping: Filter links based on robots.txt
    links_to_scrape = [link for link in links_to_scrape if can_fetch_url((link))]
    # TODO: Prioritize links based on count
    links_to_scrape = links_to_scrape[:10]  # Limit to 10 links per iteration to manage load
    target_terms = state['user_request_analysis'].model_dump()['query_parameters']
    contentfilter = ContentFilter(target_terms=target_terms, header_threshold=0.3, chunk_threshold=0.3)
    logger.info(f"**data_extracter**=> Links to scrape: {links_to_scrape}")
    # Load content from URLs
    if links_to_scrape:
        output = asyncio.run(url_loader(links_to_scrape))
        if output:
            for source in state['sources_data']:
                if source['metadata']['link'] in output:
                    # Process raw content
                    content = text_splitter.split_text(output[source['metadata']['link']]['page_content'])
                    # source['metadata']['visited'] = True
                    # source['metadata']['links_extracted'] = False
                    # source['metadata']['claims_extracted']= False
                    source['metadata'].update(output[source['metadata']['link']]['metadata'])
                    
                    # Filter and add content to vector store
                    if not source.get('stored_in_db', False):
                        asyncio.run(contentfilter.filter_and_add_to_vectorestore(content, source['metadata'], vector_store))
                        source['stored_in_db'] = True
                    state['visited_links'].add(source['metadata']['link'])

    # Process additional crawled links
    # TODO: Add a logger info for logging additional links: Done
    additional_links = [link for link in state['relevant_links'] if link not in state['visited_links']]
    additional_links = [link for link in additional_links if can_fetch_url((link))]
    logger.info(f"**data_extracter**=> Additional links to scrape: {additional_links}")
    if additional_links:
        state['crawl_depth'] += 1
        output = url_loader(additional_links)
        for link, data in output.items():
            data['metadata']['link'] = link            
            new_source = {
                'metadata': data['metadata'],
            }
            content= text_splitter.split_text(data['page_content'])
            # new_source['metadata']['visited'] = True
            # Filter content and add to vector store
            if not new_source.get('stored_in_db', False):
                asyncio.run(contentfilter.filter_and_add_to_vectorestore(content, new_source['metadata'], vector_store))
                new_source['stored_in_db'] = True
            state['sources_data'].append(new_source)
            state['visited_links'].add(link)
    return state            

def retriever(state: CustomState, config: RunnableConfig=None) -> CustomState:
    
    """Generate response based on the query."""
    
    retriever = AdvancedRetriever(vector_store=vector_store, llm=model)
    action_rationale = state.get('action_rationale', "N/A")
    search_queries = [q['query'] for q in state['search_queries']]
    prompt = f""" 
                **User Query**
                {state['user_request']}
                **Performed Web Search Queries**
                {"\n".join(search_queries)}
                **Action Rationale (CRITICAL):**
                {action_rationale}
                """
    retreived_docs = retriever(HumanMessage(content=prompt))
    # TODO: save retrieved docs into state instead of only contexts: Done
    new_docs = [doc for doc in retreived_docs if doc not in state['retrieved_docs']]
    logger.info(f"**retriever**=> Retrieved {len(new_docs)} new documents. First 3 contexts: {new_docs[:3]} ")
    state['retrieved_docs'].extend(new_docs)
    context_format = "Text:\n {content}\nSource: {source}\nURL: {link}\n Source Reliability: {source_reliability}\n\n"
    contexts = "\n".join([context_format.format(content=doc.page_content, source=doc.metadata.get('source', ''), link=doc.metadata['link'], source_reliability=doc.metadata['source_reliability']) for doc in new_docs])
    # Adding contexts as a tool message
    contexts_augmented = SystemMessage(f"""
                    Retrieved Contexts:
                    {contexts}
                    """, )
    state['contexts'] = contexts_augmented
    return state

def response_generator(state: CustomState, config: RunnableConfig=None) -> CustomState:
    
    """Generate response based on the query and retrieved contexts."""
    response_generator = model.with_structured_output(ResponseGeneration)
    
    response = response_generator.invoke([
        SystemPrompts.response_generator, 
        state['contexts'],
        #state['refined_user_request'],
        *state['messages']
        ]
        )
    logger.info(f"**response_generator**=> {response}")
    return {'messages': state['messages']+[AIMessage(content=f"**Response:** {response}", response = response.response_to_user_query)]}

def response_evaluator(state: CustomState, config: RunnableConfig=None) -> CustomState:
    
    """Evaluate the generated response and decide next steps."""
    
    response_evaluator = model.with_structured_output(ResponseEvaluation)
    evaluation = response_evaluator.invoke( [
        SystemPrompts.response_evaluator, 
        state['contexts'],
        #state['user_request'],
        *state['messages']
        ])
    if evaluation.next_step !='finish':
        state['messages'].append(HumanMessage(content=f"**Evaluator:** {evaluation.response_evaluation}\nNext Step: {evaluation.next_step}")) 
    logger.info(f"**response_evaluator**=> {evaluation}")
    return {'messages':state['messages'],'action_rationale': evaluation.action_rationale, 'next_step': evaluation.next_step, 'iteration': state['iteration'] + 1}

# TODO: Implement as a llm function and extract links: Done      
def crawl_contexts(state: CustomState, config: RunnableConfig=None):
    
    """Get relevant links from the sources."""
    
    link_extractor = model.with_structured_output(URLExtraction)
    response = link_extractor.invoke(
         [
            SystemPrompts.context_crawler,
            state['contexts'],
            *state['messages'][-2:]
        ]
    )
    logger.info(f"**crawl_contexts**=> Crawled {len(response.urls)} links.")
    return {'relevant_links': response.links, 'crawl_depth': state['crawl_depth']+1}

def conditional_edge(state: CustomState, config: RunnableConfig=None)-> Literal[END, 'retriever', 'search_query_planner', 'crawl_contexts']:
    
    """Route to the next step based on evaluation."""
    
    if state['next_step'] == 'finish':
        return END
    if state['next_step'] == 'web_search':
        return 'search_query_planner'
    if state['iteration']> state['max_iterations'] or state['crawl_depth']> state['max_crawl_depth']:
        logger.warning("Maximum iterations or crawl depth reached. Ending the process.")
        return END
    return state['next_step']
 
def route_harmful_query(state: CustomState, config: RunnableConfig=None) -> Literal[END, 'search_query_planner']:
    
    """Route the query based on harmfulness assessment."""
    
    if state['user_request_analysis'].is_harmful == True:
        logger.warning("Harmful or sensitive content detected in the query. Ending the process.")
        state['messages'].append(AIMessage(content="The query has been identified as potentially harmful or sensitive. Therefore, no further processing will be done."))
        return END
    return 'search_query_planner'
