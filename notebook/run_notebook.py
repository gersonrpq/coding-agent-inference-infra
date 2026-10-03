#!/usr/bin/env python3
"""Executes notebook/part5_queue.ipynb in place (the `jupyter nbconvert` CLI is not needed)."""
import nbformat
from nbclient import NotebookClient

path = "notebook/part5_queue.ipynb"
nb = nbformat.read(path, as_version=4)
NotebookClient(nb, timeout=120, kernel_name="python3", resources={"metadata": {"path": "."}}).execute()
nbformat.write(nb, path)
errors = [o for c in nb.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"]
print("executed;", len(errors), "cell errors")
