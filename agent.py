import logging
from typing import Any, Dict, List, Optional
from llama_index import GPTIndex, Document
from mcp_server_client import MCPClient, MCPError
import asyncio

class DataAgent:
    """
    An agent responsible for data collection, processing, and visualization
    using LlamaIndex and MCP Servers.
    """

    def __init__(self, mcp_server_url: str, index_model: str = "gpt-3.5-turbo") -> None:
        """
        Initialize the DataAgent with MCP server URL and index model.

        :param mcp_server_url: URL of the MCP server to connect to.
        :param index_model: The language model to use for indexing.
        """
        self.mcp_server_url = mcp_server_url
        self.mcp_client: Optional[MCPClient] = None
        self.index_model = index_model
        self.index: Optional[GPTIndex] = None
        self.logger = logging.getLogger(__name__)
        self._initialize_mcp_client()

    def _initialize_mcp_client(self) -> None:
        """
        Initialize the MCP client with the server URL.
        """
        try:
            self.mcp_client = MCPClient(self.mcp_server_url)
            self.logger.info(f"Connected to MCP server at {self.mcp_server_url}")
        except MCPError as e:
            self.logger.error(f"Failed to connect to MCP server: {e}")
            self.mcp_client = None

    def collect_data(self, query: str) -> List[Document]:
        """
        Collect data from MCP server based on a query.

        :param query: The query string to fetch data.
        :return: List of Document objects.
        """
        if not self.mcp_client:
            self.logger.error("MCP client is not initialized.")
            return []

        try:
            raw_data = self.mcp_client.query_data(query)
            documents = [Document(text=str(item)) for item in raw_data]
            self.logger.info(f"Collected {len(documents)} documents for query: {query}")
            return documents
        except MCPError as e:
            self.logger.error(f"Error during data collection: {e}")
            return []

    def build_index(self, documents: List[Document]) -> None:
        """
        Build a LlamaIndex from the provided documents.

        :param documents: List of Document objects.
        """
        try:
            self.index = GPTIndex.from_documents(documents, model=self.index_model)
            self.logger.info("Index built successfully.")
        except Exception as e:
            self.logger.error(f"Failed to build index: {e}")

    def process_query(self, query: str) -> str:
        """
        Process a query against the index to generate a response.

        :param query: The query string.
        :return: Response string.
        """
        if not self.index:
            self.logger.warning("Index is not built yet.")
            return "Index not available."

        try:
            response = self.index.query(query)
            self.logger.info(f"Processed query: {query}")
            return response.response
        except Exception as e:
            self.logger.error(f"Error during query processing: {e}")
            return "Error processing query."

    def visualize_data(self, data: List[Dict[str, Any]]) -> None:
        """
        Visualize data using matplotlib.

        :param data: List of data points (dicts) to visualize.
        """
        import matplotlib.pyplot as plt
        import pandas as pd

        try:
            df = pd.DataFrame(data)
            df.plot()
            plt.show()
            self.logger.info("Data visualization completed.")
        except Exception as e:
            self.logger.error(f"Error during visualization: {e}")

# Example usage (to be placed in main.py or elsewhere):
# agent = DataAgent("http://mcp-server-url")
# documents = agent.collect_data("fetch latest data")
# agent.build_index(documents)
# response = agent.process_query("Summarize the data")
# agent.visualize_data([{"x": 1, "y": 2}, {"x": 2, "y": 3}])