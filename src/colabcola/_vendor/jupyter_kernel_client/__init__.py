# Modified for Colabcola: private module namespace relocation; see ../provenance.json.
# Copyright (c) 2023-2024 Datalayer, Inc.
#
# BSD 3-Clause License

"""Jupyter Kernel Client through websocket."""

from colabcola._vendor.jupyter_kernel_client._version import __version__
from colabcola._vendor.jupyter_kernel_client.client import KernelClient
from colabcola._vendor.jupyter_kernel_client.konsoleapp import KonsoleApp
from colabcola._vendor.jupyter_kernel_client.manager import KernelHttpManager
from colabcola._vendor.jupyter_kernel_client.models import VariableDescription
from colabcola._vendor.jupyter_kernel_client.snippets import SNIPPETS_REGISTRY, LanguageSnippets
from colabcola._vendor.jupyter_kernel_client.wsclient import KernelWebSocketClient
from colabcola._vendor.jupyter_kernel_client.wsclient import JupyterSubprotocol

__all__ = [
    "SNIPPETS_REGISTRY",
    "KernelClient",
    "KernelHttpManager",
    "KernelWebSocketClient",
    "KonsoleApp",
    "LanguageSnippets",
    "VariableDescription",
    "JupyterSubprotocol",
    "__version__",
]
