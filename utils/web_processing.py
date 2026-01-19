from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import MarkdownifyTransformer
from serpapi import GoogleSearch
import logging
import os
import time
weblogger = logging.getLogger(__name__)
handler = logging.FileHandler('logs/web_processing.log', encoding='utf-8')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
weblogger.addHandler(handler)
weblogger.setLevel(logging.DEBUG)

def retrieve_search_results(query, api_params={}, max_results= 5):
    
    """Retrieve search results from SerpAPI given a query."""

    api_params['api_key'] = os.getenv('SERP_API_KEY')
    api_params['q'] = query
    search = GoogleSearch(api_params)
    output = search.get_dict()
    results = [{'metadata':{key : val for key, val in res.items() if key in ['source','link', 'date',]}} for res in output.get('organic_results', [])]
    return results[:max_results]

# Source/Domain evaluation: Fast Pass
from urllib.parse import urlparse
import re
class SourceEvaluator:
    """Basic domain evaluator for reliability"""
    def __init__(self):
        # Quick lookup dictionary for known high-authority domains
        self.authority_domains = {
            r'\.org': 0.9,
            r'\.edu\.?': 0.85,
            r'\.gov\.?': 0.9,
            # todo: More trusted domains
        }
        # Common spam/low-quality indicators
        self.red_flags = [
            r'\.xyz$', r'\.club$', r'\.biz$', r'\.click$', r'\.tk$', r'\.ga$', r'\.cf$', r'\.top$'
            r'\.info$',
            r'free-',
            r'-free',
            r'win-',  r'bonus', r'prize',
            # todo: More patterns - shortened urls
        ]
    def __call__(self, url: str):
        domain = urlparse(url).netloc.lower()
        score = 0.5  # Default score for unknown domains
        # Check for authority domains
        for pattern, auth_score in self.authority_domains.items():
            if re.search(pattern, domain):
                score = max(score, auth_score)
                break
        for pattern in self.red_flags:
            if re.search(pattern, domain):
                score *= 0.7
        if domain.count('.') > 3:
            score *= 0.7
        return score
# TODO: add functinality to load multiple urls; use asychromium loader => done
# TODO: Ethical Scrapping: add robots.txt checking: Done
from urllib.robotparser import RobotFileParser
from urllib.parse import urljoin, urlsplit

def can_fetch_url(url: str, user_agent='*'):
    """
    Check if a URL can be scraped according to robots.txt
    """
    rp = RobotFileParser()
    base_url = '{uri.scheme}://{uri.netloc}/'.format(uri=urlsplit(url))
    robots_url = urljoin(base_url, '/robots.txt')
    rp.set_url(robots_url)
    try:
        rp.read()
    except TimeoutError:
        try:
            time.sleep(1)  # avoid rapid retries
            rp.read()
        except Exception as e:
            weblogger.debug(f"Could not read robots.txt at {robots_url}: {e}")
    except Exception as e:
        weblogger.debug(f"Error reading robots.txt at {robots_url}: {e}")
        return True  # Assume allowed if robots.txt cannot be read
    if rp.can_fetch(user_agent, url):
        return True
    else:
        weblogger.debug(f"Blocked by robots.txt: {url}")
        return False

async def url_loader(urls: list): 
    """
    loads content from multiple urls.
    """
    try:
        loader = AsyncChromiumLoader(urls, user_agent="AutoScrapperBot")
        html_documents =  await loader.aload()
        html_documents = [doc for doc in html_documents if doc.page_content.strip() != '']
        
        weblogger.info(f"**URL Loader**=> Loaded {len(html_documents)} documents from {len(urls)} urls.")
        
        html2md = MarkdownifyTransformer(ignore_links=False)
        docs_transformed = html2md.transform_documents(html_documents)
        domain_evaluator = SourceEvaluator()
        source_reliability = [domain_evaluator(url) for url in urls]
        outputs = {}
        for doc, url, source_reliability in zip(docs_transformed, urls, source_reliability):
            doc.metadata['source_reliability'] = source_reliability
            outputs[url]={'metadata':doc.metadata, "page_content":doc.page_content}
        return outputs
    except Exception as e:
        print(f"{e}: failed loading url {urls}.")