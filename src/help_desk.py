import sys
import load_db
import collections
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM


class HelpDesk():
    """Create the necessary objects to create a QARetrieval chain"""
    def __init__(self, new_db=True):
        """
        Initialize the HelpDesk object.
        
        Args:
            new_db (bool): Whether to create a new database or use an existing one.
        """
        self.new_db = new_db
        self.template = self.get_template()
        self.llm = self.get_llm()
        self.prompt = self.get_prompt()

        # Create data loader
        self.data_loader = load_db.DataLoader()

        if self.new_db:
            self.db = self.data_loader.set_db()
        else:
            self.db = self.data_loader.get_db()

        self.retriever = self.db.as_retriever()
        self.retrieval_qa_chain = self.get_retrieval_qa()


    def get_template(self):
        template = """
        Given this text extracts:
        -----
        {context}
        -----
        Please answer with to the following question:
        Question: {question}
        Helpful Answer:
        """
        return template

    def get_prompt(self) -> PromptTemplate:
        prompt = PromptTemplate(
            template=self.template,
            input_variables=["context", "question"]
        )
        return prompt

    def get_llm(self):
        """
        Returns the Ollama LLM with Mistral model
        """
        # Initialize Ollama with the Mistral model and custom base URL
        llm = OllamaLLM(
            model="mistral", 
            temperature=0.7,
            streaming=True,
            base_url="http://192.168.178.106:11434"
        )
        return llm

    def get_retrieval_qa(self):
        chain_type_kwargs = {"prompt": self.prompt}
        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs
        )
        return qa

    def retrieval_qa_inference(self, question, verbose=True):
        query = {"query": question}
        answer = self.retrieval_qa_chain(query)
        sources = self.list_top_k_sources(answer, k=2)

        if verbose:
            print(sources)

        return answer["result"], sources

    def list_top_k_sources(self, answer, k=2):
        sources = [
            f'[{res.metadata["title"]}]({res.metadata["source"]})'
            for res in answer["source_documents"]
        ]

        if sources:
            k = min(k, len(sources))
            distinct_sources = list(zip(*collections.Counter(sources).most_common()))[0][:k]
            distinct_sources_str = "  \n- ".join(distinct_sources)

        if len(distinct_sources) == 1:
            return f"This might be useful to you:  \n- {distinct_sources_str}"

        elif len(distinct_sources) > 1:
            return f"Here are {len(distinct_sources)} sources that might be useful to you:  \n- {distinct_sources_str}"

        else:
            return "Sorry, I couldn't find any resources to answer your question."
