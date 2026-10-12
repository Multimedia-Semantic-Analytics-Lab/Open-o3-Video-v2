# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import ipaddress
import logging
import os
import socket

import uvicorn
from fastapi import FastAPI

logger = logging.getLogger(__file__)


def get_max_position_embeddings(hf_config) -> int:
    max_len = getattr(hf_config, "max_position_embeddings", None)
    if max_len is None:
        text_config = getattr(hf_config, "text_config", None)
        if text_config is not None:
            max_len = getattr(text_config, "max_position_embeddings", None)

    if max_len is None:
        raise ValueError("max_position_embeddings not found in HFModelConfig!")
    return int(max_len)


def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def get_free_port(address: str, max_retries: int = 10, start_port: int = 30000, end_port: int = 60000) -> tuple[int, socket.socket]:
    """Get a free port for TCPStore/TCP server. Includes retry logic for port conflicts.

    Args:
        address: The address to bind to.
        max_retries: Maximum number of retries when a port is occupied.
        start_port: Starting port for retry range (default: 30000).
        end_port: Ending port for retry range (default: 65000).

    Returns:
        A tuple of (port, socket).
    """
    family = socket.AF_INET
    if is_valid_ipv6_address(address):
        family = socket.AF_INET6

    for attempt in range(max_retries):
        sock = socket.socket(family=family, type=socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            if attempt == 0:
                sock.bind((address, 0))
            else:
                import random
                port = random.randint(start_port, end_port)
                logger.info(f"get_free_port retry {attempt}/{max_retries}: trying port {port}")
                sock.bind((address, port))
            port = sock.getsockname()[1]
            logger.info(f"get_free_port success: bound to port {port}")
            return port, sock
        except OSError as e:
            sock.close()
            logger.warning(f"get_free_port retry {attempt}/{max_retries}: port binding failed, error: {e}")
    raise RuntimeError(f"Failed to find a free port in range [{start_port}, {end_port}] after {max_retries} retries")


async def run_unvicorn(app: FastAPI, server_args, server_address, max_retries=5) -> tuple[int, asyncio.Task]:
    server_port, server_task = None, None

    for i in range(max_retries):
        try:
            server_port, sock = get_free_port(server_address)
            app.server_args = server_args
            config = uvicorn.Config(app, host=server_address, port=server_port, log_level="warning")
            server = uvicorn.Server(config)
            server.should_exit = True
            await server.serve()
            server_task = asyncio.create_task(server.main_loop())
            break
        except (OSError, SystemExit) as e:
            logger.error(f"Failed to start HTTP server on port {server_port} at try {i}, error: {e}")
    else:
        logger.error(f"Failed to start HTTP server after {max_retries} retries, exiting...")
        os._exit(-1)

    logger.info(f"HTTP server started on port {server_port}")
    return server_port, server_task
