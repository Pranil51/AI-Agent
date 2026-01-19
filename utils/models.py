import os
from langchain_aws import ChatBedrock, BedrockEmbeddings
import config

def load_embed_model():
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID') 
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY') 
    region = os.getenv('AWS_REGION')  
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
        region_name=region, 
        aws_access_key_id=aws_access_key, 
        aws_secret_access_key=aws_secret_key
        ) 
    return embeddings
def load_llm(model_id = config.LLM_MODEL_ID, region = os.getenv('AWS_REGION')):
    llm = ChatBedrock( 
    model_id=model_id,  
    region_name=region 
    )  
    return llm