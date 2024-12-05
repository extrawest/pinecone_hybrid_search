import os

from dotenv import load_dotenv
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class HybridSearchSetup:
    def __init__(self):
        self._load_environment_variables()
        self.pc = self._initialize_pinecone()
        self.index_name = "hybrid-search-langchain-pinecone"
        self.embeddings = self._initialize_embeddings()
        self.bm25_encoder = self._initialize_bm25_encoder()
        self.index = self._get_or_create_index()

    def _load_environment_variables(self):
        load_dotenv()
        os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not self.pinecone_api_key:
            raise EnvironmentError("PINECONE_API_KEY is not set in .env")

    def _initialize_pinecone(self):
        return Pinecone(api_key=self.pinecone_api_key)

    @staticmethod
    def _initialize_embeddings():
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    @staticmethod
    def _initialize_bm25_encoder():
        return BM25Encoder().default()

    def _get_or_create_index(self):
        if self.index_name not in self.pc.list_indexes().names():
            print(f"Creating index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=384,
                metric="dotproduct",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        return self.pc.Index(self.index_name)

    def store_bm25_values(self, sentences, file_name="bm25_values.json"):
        self.bm25_encoder.fit(sentences)
        self.bm25_encoder.dump(file_name)
        print(f"BM25 values saved to {file_name}")

    @staticmethod
    def add_texts_to_retriever(retriever, sentences):
        retriever.add_texts(sentences)

    @staticmethod
    def retrieve_query(retriever, query):
        return retriever.invoke(query)

    def run(self, sentences=None, query=None):
        """
        Run the hybrid search with customizable sentences and query.

        :param sentences: List of sentences to add to the retriever.
        :param query: Query to retrieve results.
        """
        self.store_bm25_values(sentences)

        retriever = PineconeHybridSearchRetriever(
            embeddings=self.embeddings, sparse_encoder=self.bm25_encoder, index=self.index
        )
        self.add_texts_to_retriever(retriever, sentences)

        result = self.retrieve_query(retriever, query)
        print(f"Query result: {result}")


if __name__ == "__main__":
    try:
        sentences = [
            "In 2019, I visited Hungary",
            "In 2020, I visited Czech Republic",
            "In 2021, I visited Georgia",
        ]
        print(f'Input sentences: {sentences}')
        custom_query = "What country did I visit first?"
        print(f'Custom query: {custom_query}')

        hybrid_search = HybridSearchSetup()
        hybrid_search.run(sentences=sentences, query=custom_query)
    except Exception as e:
        print(f"Error occurred: {e}")
