# main.py

import asyncio
import logging
from typing import List, Dict, Any

from tools import initialize_tools
from agent import MultiAgentSystem

logging.basicConfig(level=logging.INFO)

async def main() -> None:
    """
    Main entry point for the agent system.
    Initializes tools, sets up agents, and orchestrates data collection,
    processing, and visualization.
    """
    try:
        # Initialize tools and environment
        tools = initialize_tools()

        # Initialize multi-agent system
        agent_system = MultiAgentSystem(tools=tools)

        # Perform data collection
        data = await agent_system.collect_data()

        # Process data
        processed_data = agent_system.process_data(data)

        # Visualize data
        agent_system.visualize_data(processed_data)

    except Exception as e:
        logging.exception("An error occurred during execution: %s", e)

if __name__ == "__main__":
    asyncio.run(main())