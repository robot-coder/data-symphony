import requests
from typing import Any, Dict, Optional, List
from llama_index import GPTIndex, Document
from mcp_server_client import MCPClient
import pandas as pd
import matplotlib.pyplot as plt
from playwright.sync_api import sync_playwright

def fetch_data_from_url(url: str, headers: Optional[Dict[str, str]] = None) -> str:
    """
    Fetches raw data from a specified URL.

    Args:
        url (str): The URL to fetch data from.
        headers (Optional[Dict[str, str]]): Optional HTTP headers.

    Returns:
        str: The content retrieved from the URL.

    Raises:
        requests.HTTPError: If the request fails.
    """
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        raise

def process_data_to_dataframe(data: str, data_format: str = 'json') -> pd.DataFrame:
    """
    Processes raw data into a pandas DataFrame.

    Args:
        data (str): Raw data as string.
        data_format (str): Format of the data ('json', 'csv', 'txt').

    Returns:
        pd.DataFrame: Processed data as DataFrame.

    Raises:
        ValueError: If data_format is unsupported.
    """
    try:
        if data_format == 'json':
            return pd.read_json(data)
        elif data_format == 'csv':
            from io import StringIO
            return pd.read_csv(StringIO(data))
        elif data_format == 'txt':
            # For plain text, attempt to parse into DataFrame if structured
            # Placeholder: user should customize based on data structure
            print("Plain text data received. Please implement custom parsing.")
            return pd.DataFrame()
        else:
            raise ValueError(f"Unsupported data_format: {data_format}")
    except Exception as e:
        print(f"Error processing data: {e}")
        raise

def visualize_data(df: pd.DataFrame, x_axis: str, y_axis: str, chart_type: str = 'line') -> None:
    """
    Creates and displays a chart from DataFrame data.

    Args:
        df (pd.DataFrame): Data to visualize.
        x_axis (str): Column name for x-axis.
        y_axis (str): Column name for y-axis.
        chart_type (str): Type of chart ('line', 'bar', 'scatter').

    Raises:
        ValueError: If chart_type is unsupported.
    """
    try:
        plt.figure(figsize=(10, 6))
        if chart_type == 'line':
            plt.plot(df[x_axis], df[y_axis])
        elif chart_type == 'bar':
            plt.bar(df[x_axis], df[y_axis])
        elif chart_type == 'scatter':
            plt.scatter(df[x_axis], df[y_axis])
        else:
            raise ValueError(f"Unsupported chart_type: {chart_type}")
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.title(f"{chart_type.capitalize()} Chart of {y_axis} vs {x_axis}")
        plt.show()
    except Exception as e:
        print(f"Error visualizing data: {e}")
        raise

def initialize_llama_index(documents: List[str]) -> GPTIndex:
    """
    Initializes a GPTIndex with provided documents.

    Args:
        documents (List[str]): List of document texts.

    Returns:
        GPTIndex: Initialized index object.
    """
    docs = [Document(text=doc) for doc in documents]
    index = GPTIndex.from_documents(docs)
    return index

def create_mcp_client(server_url: str, api_key: str) -> MCPClient:
    """
    Creates an MCPClient instance for server communication.

    Args:
        server_url (str): URL of the MCP server.
        api_key (str): API key for authentication.

    Returns:
        MCPClient: Configured MCP client.
    """
    try:
        client = MCPClient(server_url=server_url, api_key=api_key)
        return client
    except Exception as e:
        print(f"Error initializing MCPClient: {e}")
        raise

def run_playwright_script(script_url: str) -> None:
    """
    Runs a web automation script using Playwright.

    Args:
        script_url (str): URL of the script or page to automate.

    Raises:
        Exception: If automation fails.
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(script_url)
            # Placeholder for further automation steps
            print(f"Loaded page: {script_url}")
            browser.close()
    except Exception as e:
        print(f"Error during Playwright automation: {e}")
        raise