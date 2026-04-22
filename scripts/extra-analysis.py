from pathlib import Path

import pandas as pd
from typer import Typer

from shared_metrics import ecfp4_weight

app = Typer(pretty_exceptions_enable=False)


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
    topn = (df["score"] >= key_score).sum() / len(df)
    print(f"{key}: {topn:.3%}")


if __name__ == "__main__":
    app()
