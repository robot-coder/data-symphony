# README.md

# Agent System with LlamaIndex and MCP Servers

This project implements an agent system that leverages LlamaIndex and MCP Servers to perform data collection, processing, and visualization tasks. The system supports multi-agent coordination for complex problem-solving scenarios.

## Features

- Data retrieval from MCP Servers
- Data processing and indexing with LlamaIndex
- Multi-agent coordination for distributed tasks
- Data visualization using Matplotlib
- Modular and extensible architecture

## Requirements

Ensure you have Python 3.8+ installed. Install the required libraries:

```bash
pip install -r requirements.txt
```

## Files

- `main.py`: Entry point to initialize and run the agent system.
- `tools.py`: Utility functions for data handling, visualization, and MCP server interactions.
- `agent.py`: Defines the Agent class responsible for task execution and coordination.
- `requirements.txt`: Lists all dependencies.
- `README.md`: This documentation file.

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the main script:

```bash
python main.py
```

## Example

The system will initialize agents, connect to MCP servers, perform data collection, process data with LlamaIndex, and generate visualizations.

---

## Code Snippets

### main.py

```python
import asyncio
from agent import Agent
from tools import initialize_mcp_client, visualize_data

async def main():
    try:
        mcp_client = initialize_mcp_client()
        agents = [Agent(name=f"Agent_{i}", mcp_client=mcp_client) for i in range(3)]
        # Run agents concurrently
        await asyncio.gather(*(agent.run() for agent in agents))
        # Example data visualization
        data = await mcp_client.fetch_data()
        visualize_data(data)
    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

### tools.py

```python
from mcp_server_client import MCPClient
import pandas as pd
import matplotlib.pyplot as plt
import requests
from typing import Any, Dict, List

def initialize_mcp_client() -> MCPClient:
    """
    Initialize and return an MCPClient instance.
    """
    try:
        client = MCPClient(server_url="http://localhost:8000")
        return client
    except Exception as e:
        print(f"Failed to initialize MCPClient: {e}")
        raise

async def fetch_data_from_mcp(client: MCPClient) -> List[Dict[str, Any]]:
    """
    Fetch data asynchronously from MCP server.
    """
    try:
        data = await client.fetch_data()
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

def process_data(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert list of dicts to pandas DataFrame for processing.
    """
    try:
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        print(f"Error processing data: {e}")
        return pd.DataFrame()

def visualize_data(df: pd.DataFrame) -> None:
    """
    Generate a simple visualization of the data.
    """
    try:
        df.plot(kind='bar')
        plt.title("Data Visualization")
        plt.xlabel("Index")
        plt.ylabel("Values")
        plt.show()
    except Exception as e:
        print(f"Error during visualization: {e}")
```

### agent.py

```python
import asyncio
from llama_index import GPTIndex
from tools import fetch_data_from_mcp, process_data
from mcp_server_client import MCPClient
from typing import Optional

class Agent:
    """
    Represents an agent responsible for data collection, processing, and coordination.
    """

    def __init__(self, name: str, mcp_client: MCPClient):
        self.name: str = name
        self.mcp_client: MCPClient = mcp_client
        self.index: Optional[GPTIndex] = None

    async def run(self) -> None:
        """
        Main execution method for the agent.
        """
        try:
            data = await fetch_data_from_mcp(self.mcp_client)
            processed_data = process_data(data)
            self.index = self.build_index(processed_data)
            await self.coordinate()
        except Exception as e:
            print(f"{self.name} encountered an error: {e}")

    def build_index(self, data: pd.DataFrame) -> GPTIndex:
        """
        Build a LlamaIndex from processed data.
        """
        try:
            documents = data.to_dict(orient='records')
            index = GPTIndex.from_documents(documents)
            return index
        except Exception as e:
            print(f"Error building index in {self.name}: {e}")
            raise

    async def coordinate(self) -> None:
        """
        Coordinate with other agents for complex problem-solving.
        """
        # Placeholder for multi-agent coordination logic
        print(f"{self.name} is coordinating with other agents.")
        await asyncio.sleep(1)  # Simulate coordination delay
```