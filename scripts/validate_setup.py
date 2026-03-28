"""Stage 0 validation script."""

from importlib.metadata import version

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


def main() -> None:
    app = FastAPI(title="Autonomous Multi-hop Research Agent")

    class HealthCheck(BaseModel):
        status: str

    payload = HealthCheck(status="ok")
    frame = pd.DataFrame({"value": np.array([1, 2, 3])})

    print("Stage 0 validation passed.")
    print(f"FastAPI app title: {app.title}")
    print(f"Pydantic sample payload: {payload.model_dump()}")
    print(f"NumPy sum: {frame['value'].sum()}")
    print("Detected package versions:")
    for package_name in [
        "fastapi",
        "uvicorn",
        "pydantic",
        "numpy",
        "pandas",
        "scikit-learn",
        "datasets",
        "sentence-transformers",
        "faiss-cpu",
        "langgraph",
        "langchain",
        "python-dotenv",
    ]:
        print(f"  - {package_name}: {version(package_name)}")


if __name__ == "__main__":
    main()
