MCP Server
==========

dscompanion ships an optional local MCP (Model Context Protocol) server, exposing
five tools an AI agent can call directly: ``analyze_dataset``,
``train_and_compare_models``, ``check_model_readiness``, ``explain_model``, and
``generate_report``.

Everything runs locally over stdio transport -- your data and models never leave
your machine, and no hosted service or account is required.

Installation
------------

.. code-block:: bash

   pip install dscompanion[mcp]

Usage
-----

Configure your MCP client (e.g. Claude Desktop) to launch:

.. code-block:: bash

   python -m dscompanion.mcp

Each tool takes and returns plain file paths and JSON-serialisable data. A typical
workflow: ``analyze_dataset`` to understand your data, ``train_and_compare_models``
to train and pick a model, then ``check_model_readiness``, ``explain_model``, and
``generate_report`` against the run directory that call returned.

Limitations
-----------

- ``train_and_compare_models`` currently supports ``task="classification"`` only.
- ``explain_model`` returns feature-importance data only, no chart image.
