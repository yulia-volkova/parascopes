import pandas as pd
from typing import List, Dict
from yulia.outlines.metrics import semantic_prf1, conciseness_ratio, redundancy_score
from yulia.outlines.config import THRESHOLDS

def evaluate_rows(rows: List[Dict], model_cols: List[str]) -> pd.DataFrame:
    
    out = []
    for row in rows:
        completion = row["completion"]
        example_id = row.get("example_id", None)
        for col in model_cols:
            outline = row.get(col, "")
            prf = semantic_prf1(completion, outline)
            conc = conciseness_ratio(completion, outline)
            red  = redundancy_score(outline)
            out.append({
                "example_id": example_id,
                "model": col.replace("_outline", ""),
                "precision": prf["precision"],
                "recall": prf["recall"],
                "f1": prf["f1"],
                "conciseness": conc,
                "redundancy": red,
            })
    return pd.DataFrame(out)

def summarize_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean metrics per model + pass/fail flags vs thresholds.
    """
    agg = df.groupby("model", as_index=False).mean(numeric_only=True)
    agg["recall_pass"]      = agg["recall"]      >= THRESHOLDS["recall"]
    agg["conciseness_pass"] = agg["conciseness"] <= THRESHOLDS["conciseness"]
    agg["redundancy_pass"]  = agg["redundancy"]  <= THRESHOLDS["redundancy"]
    return agg.sort_values(["f1", "recall"], ascending=False)
