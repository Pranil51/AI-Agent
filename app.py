from main import *
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import InMemorySaver
import random
import gradio as gr
subgraph_builder = StateGraph(CustomState)

subgraph_builder.add_node(user_query_analyzer)
subgraph_builder.add_node(search_query_planner)
subgraph_builder.add_node(web_search)
subgraph_builder.add_node(crawl_contexts)
subgraph_builder.add_node(data_extracter)
subgraph_builder.add_node(retriever)
subgraph_builder.add_node(response_generator)
subgraph_builder.add_node(response_evaluator)
subgraph_builder.add_edge(START, 'user_query_analyzer')
subgraph_builder.add_conditional_edges('user_query_analyzer', route_harmful_query)
subgraph_builder.add_edge('search_query_planner', 'web_search')
subgraph_builder.add_edge('web_search', 'data_extracter')
subgraph_builder.add_edge('data_extracter', 'retriever')
subgraph_builder.add_edge('retriever', 'response_generator')
subgraph_builder.add_edge('response_generator',  'response_evaluator')
subgraph_builder.add_edge('crawl_contexts', 'data_extracter')
subgraph_builder.add_conditional_edges('response_evaluator', conditional_edge)

Subgraph_checkpointer = InMemorySaver()
sub_graph_config = {"configurable": {"thread_id": str(random.randint(1, 999))}, 'recursion_limit': 100} 
websearch_agent = subgraph_builder.compile(
    checkpointer=Subgraph_checkpointer
    )
class MainState(MessagesState):
    user_request: str

def generate_websearch_response(state: MainState):
    user_request = state['user_request']
    
    state['messages'].append(HumanMessage(content=user_request))
    
    # Creating a single refined query from Overall Chat for Web Search Agent
    user_request_refiner = model.with_structured_output(QueryRefiner)
    output = user_request_refiner.invoke([
        SystemPrompts.user_query_refiner,
        *state['messages']])
    refined_query = output.refined_query
    response = websearch_agent.invoke(
        CustomState(user_request=refined_query, 
                    iteration=0,
                    crawl_depth=0,
                    max_crawl_depth=2,
                    max_iterations=3,
                    messages=[],
                    sources_data=[],
                    relevant_links=set(),
                    visited_links=set(),
                    retrieved_docs=[], 
                    search_queries=[],
                ),
                config=sub_graph_config
    )
    state['messages'].append(AIMessage(content=response['messages'][-1].response))
    return state

graph_builder = StateGraph(MainState)
graph_builder.add_node(generate_websearch_response)
graph_builder.set_entry_point('generate_websearch_response')
graph_checkpointer = InMemorySaver()
graph_config = {"configurable": {"thread_id": str(random.randint(999, 9999))}, 'recursion_limit': 100} 

chat_agent = graph_builder.compile(
    checkpointer = graph_checkpointer
)


def response_generator(message, history):
    response = chat_agent.invoke(
        MainState(
            user_request=message, 
            history=[]
                    ),
        config=graph_config
    )
    return response['messages'][-1].content
# fn: with message and history as arg and response as output
demo = gr.ChatInterface(
    fn=response_generator, 
    title="Web Equiped Chatbot"
)
demo.launch(inbrowser=True)

# print(response_generator("What is the weather like today?", []))