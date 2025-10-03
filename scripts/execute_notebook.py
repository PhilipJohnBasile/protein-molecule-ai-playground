from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils import io as ipy_io


def execute_notebook(src: Path, dst: Path) -> None:
    nb = nbformat.read(src, as_version=4)
    shell = InteractiveShell.instance()
    shell.execution_count = 1
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        with ipy_io.capture_output() as captured:
            result = shell.run_cell(cell.source, store_history=False)
        outputs = list(captured.outputs)
        if result.result is not None:
            outputs.append(
                nbformat.v4.new_output(
                    "execute_result",
                    data={"text/plain": repr(result.result)},
                    execution_count=shell.execution_count,
                )
            )
        cell["outputs"] = outputs
        cell["execution_count"] = shell.execution_count
        shell.execution_count += 1
    nbformat.write(nb, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute a notebook without launching an external kernel.")
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    args = parser.parse_args()
    execute_notebook(args.src, args.dst)


if __name__ == "__main__":
    main()
