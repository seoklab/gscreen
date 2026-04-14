from pathlib import Path

import numpy as np
import pandas as pd
from typer import Typer

app = Typer(pretty_exceptions_enable=False)


def ecfp4_weight(df):
    ecfp4 = df["ecfp4"].to_numpy()
    return np.clip(6 * (ecfp4 - 0.1), 0, 0.9)


@app.command()
def main(
    results_csv: Path,
    key: str,
):
    df = pd.read_csv(results_csv)
    if "is_active" not in df.columns:
        df["is_active"] = df["type"] == "active"
        df = df.drop(columns=["type"])
        df = df.rename(
            columns={
                "pharma_score": "pharma",
                "shape_score": "shape",
                "tani_sim": "ecfp4",
            }
        )

    weight = ecfp4_weight(df)
    df["score"] = df["shape"] * weight + df["pharma"] * (1 - weight)

    df = df.set_index("id")
    key_score = df.loc[key, "score"]
    n_above = (df["score"] > key_score).sum()
    n_tied = (df["score"] == key_score).sum()
    topn = (n_above + (n_tied + 1) / 2) / len(df)
    print(f"{key}: {topn:.3%}")


if __name__ == "__main__":
    app()
