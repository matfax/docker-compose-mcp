#!/usr/bin/env python3
"""
Entry point for running docker-compose-mcp as a module.
Usage: python -m docker_compose_mcp
"""

from . import main

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
