from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from keybert import KeyBERT

kw_model = KeyBERT()
import spacy
try:
    nlp = spacy.load("en_core_web_lg")
except Exception as e:
    print("Downloading 'en_core_web_lg' model...")
    from spacy.cli import download
    download("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")

import logging
textlogger = logging.getLogger(__name__)
handler = logging.FileHandler('logs/text_processing.log', encoding='utf-8')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
textlogger.addHandler(handler)
textlogger.setLevel(logging.DEBUG)

class AdvancedMarkdownSplitter:
    def __init__(self, chunk_size=500, chunk_overlap=0):
        self.text_splitter = RecursiveCharacterTextSplitter(
                                              chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            )
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        # Initialize the first splitter
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on = headers_to_split_on, 
                                                    strip_headers=True,
                                                    
                                                    )
        self.chunk_size = chunk_size

    def split_text(self, markdown_text):
        """
            Processes the input data and organizes it into a dictionary.

            Parameters
            ----------
            markdown_text : text
                Large text corpus to be split

            Returns
            -------
            Dict[str, List[str]]
                A dictionary where:
                - Keys strings, representing hierarchical headers (e.g., ("Header 1", "Header 2")).
                - Values are lists of strings, each representing a context chunk associated with the headers.
                
            Examples
            --------
            >>> split_text(["text_content"])
            {
                "Header 1": ["item1_sec1"],
                "Header 1\nHeader 2": ["item2_sec1_subA"],
            }
        """
        md_header_splits = self.markdown_splitter.split_text(markdown_text)
        final_chunks = {}
        for doc in md_header_splits:
            headers = "\n".join(doc.metadata.values())
            if len(doc.page_content)>self.chunk_size:
                sub_chunks = self.text_splitter.split_text(doc.page_content)
                for ch_ in sub_chunks:
                    if headers not in final_chunks:
                        final_chunks[headers] = [ch_]   
                    else:
                        final_chunks[headers].append(ch_)
            else:
                final_chunks[headers] = [doc.page_content]   
        return final_chunks



# TODO: Combine search ent doc and kw ent doc into single set to avoid duplicate computation.
# TODO: Log similarity scores for analysis.
class ContentFilter:
    """
    Filters and embeds relevant content chunks based on the target terms.
    Args:
        target_terms: dict with 'entities' and 'keywords' as keys
        header_threshold: float 
        chunk_threshold: float
    """
    
    def __init__(self, target_terms: dict, header_threshold: float =0.20, chunk_threshold: float =0.10 ):
        self.header_threshold = header_threshold
        self.chunk_threshold = chunk_threshold
        self.target_terms = target_terms
    
    async def filter_and_add_to_vectorestore(self, content: dict, content_metadata, vector_store):
        
        """ 
            Adds relevant content chunks to the vector store based on semantic similarity with target terms.
            Parameters
            ----------
            content : Dict[str, List[str]]
                A dictionary where:
                - Keys are strings representing hierarchical headers.
                - Values are lists of strings, each representing a content chunk associated with the headers.
            
            content_metadata : dict
                Metadata of the content to be added to each chunk.
            
            vector_store : VectorStore
                The vector store where relevant chunks will be added.

        """
        def has_semantic_match(primary_docs: list, text_to_match:str, threshold: float =0.30):
            """
            Check if any of the primary_docs have semantic similarity above the threshold with the text_to_match.
            Returns True if text is a semantic match.
            """
            if not primary_docs or not text_to_match:
                return False
            text_entities = [ent.text.lower() for ent in nlp(text_to_match).ents]
            primary_doc_lengths = [len(doc) for doc in primary_docs]
            keyphrase_ngram_range = (min(primary_doc_lengths), max(primary_doc_lengths),)
            seed_keywords = [doc.__str__() for doc in primary_docs]
            text_keywords = [kw[0] for kw in kw_model.extract_keywords(text_to_match, keyphrase_ngram_range=keyphrase_ngram_range, seed_keywords=seed_keywords)]
            text_terms = text_entities + text_keywords
            if not text_terms:
                return False
            for p_doc in primary_docs:
                for term in text_terms:
                        term_doc = nlp(term)
                        if p_doc.vector_norm and term_doc.vector_norm:
                            if p_doc.similarity(term_doc) > threshold:
                                return True
            return False
        
        # Generate docs for target terms
        search_ent_docs = [nlp(kw) for kw in self.target_terms['entities'] if self.target_terms['entities']]
        search_kw_docs = [nlp(kw) for kw in self.target_terms['keywords'] if self.target_terms['keywords']]
        n_added = 0
        for headers, chunks in content.items():
            chunks_metadatas = [{'chunk_position':i+1, 'headers': headers} for i in range(len(chunks))]
            for md in chunks_metadatas:
                md.update(content_metadata)
            # stage 1: header level filtering
            if headers:
                # secondary_kw_doc = header_keywords['entities'] + header_keywords['nouns'] if (header_keywords['entities'] and header_keywords['nouns']) else (header_keywords['verbs'] + header_keywords['adjectives'])
                if has_semantic_match(search_ent_docs, headers, self.header_threshold) or has_semantic_match(search_kw_docs, headers, self.header_threshold):
                    await vector_store.aadd_texts(chunks, metadatas=chunks_metadatas)
                    n_added += +1
                    continue # Move to next header group
            # stage 2: chunk level filtering
            for  chunk in chunks:
                ## similarity match for text
                if has_semantic_match(search_ent_docs, chunk, self.header_threshold) or has_semantic_match(search_kw_docs, chunk, self.header_threshold):
                    await vector_store.aadd_texts(chunks, metadatas=chunks_metadatas)
                    n_added += 1
                    # print("Added chunk based on chunk-level match.: ", len(chunk))
                    break # Move to next header group
        textlogger.debug(f"**Content Filter**=> Total header groups added for Header-{headers}: {n_added} out of {len(content)}")