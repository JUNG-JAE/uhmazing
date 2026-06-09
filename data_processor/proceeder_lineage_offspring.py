from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "Data" / "hanwoo_original"
TRAIN_PATH = DATA_DIR / "hanwoo_train.csv"
TEST_PATH = DATA_DIR / "hanwoo_test.csv"
LINEAGE_PATH = DATA_DIR / "hanwoo_lineage.csv"
OUTPUT_PATH = DATA_DIR / "hanwoo_lineage_offspring_by_year.csv"
METADATA_PATH = DATA_DIR / "hanwoo_lineage_offspring_by_year.meta.json"

OUTPUT_COLUMNS = [
    "LINEAGE_NO",
    "ROLE",
    "TARGET_YEAR",
    "PAST_OFFSPRING_COUNT",
]
MISSING_TOKENS = {"", "-99", "-99.0", "MISSING"}
UNKNOWN_LINEAGE_CODES = {
    "kluWj1LiM8I6nYWfDenO7q4tJySB2AVV8z9cMqweuXA=",
    "gQagjD++POKUI4kyvXKUoA==",
    "2XwK0r9Ij2yaHcePqO7Bwg==",
}


def normalize_identifier(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").str.strip()
    invalid = (
        normalized.isna()
        | normalized.isin(MISSING_TOKENS)
        | normalized.isin(UNKNOWN_LINEAGE_CODES)
    )
    return normalized.mask(invalid)


def parse_dates(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").str.strip()
    normalized = normalized.mask(
        normalized.isna() | normalized.isin(MISSING_TOKENS)
    )
    return pd.to_datetime(normalized, errors="coerce")


def collect_target_years(train_dates: pd.Series, test_path: Path) -> List[int]:
    test = pd.read_csv(
        test_path,
        usecols=["ABATT_DATE"],
        dtype={"ABATT_DATE": "string"},
        encoding="utf-8-sig",
    )
    test_dates = parse_dates(test["ABATT_DATE"])
    years = set(train_dates.dropna().dt.year.astype(int))
    years.update(test_dates.dropna().dt.year.astype(int))
    if not years:
        raise ValueError("No valid target year was found in train or test data.")
    return sorted(years)


def file_metadata(path: Path) -> Dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
        "sha256": digest.hexdigest(),
    }


def build_role_output(
    child_history: pd.DataFrame,
    parent_column: str,
    role: str,
    target_years: Iterable[int],
) -> pd.DataFrame:
    all_parents = (
        child_history[parent_column]
        .dropna()
        .drop_duplicates()
        .sort_values(kind="stable")
        .reset_index(drop=True)
    )
    valid_relations = child_history.loc[
        child_history[parent_column].notna()
        & child_history["CHILD_ABATT_YEAR"].notna(),
        ["CHILD_CATTLE_NO", parent_column, "CHILD_ABATT_YEAR"],
    ].drop_duplicates(["CHILD_CATTLE_NO", parent_column])

    outputs = []
    for target_year in target_years:
        counts = (
            valid_relations.loc[
                valid_relations["CHILD_ABATT_YEAR"].lt(target_year)
            ]
            .groupby(parent_column, sort=False)
            .size()
        )
        output = pd.DataFrame(
            {
                "LINEAGE_NO": all_parents,
                "ROLE": role,
                "TARGET_YEAR": target_year,
            }
        )
        output["PAST_OFFSPRING_COUNT"] = (
            output["LINEAGE_NO"].map(counts).fillna(0).astype("int64")
        )
        outputs.append(output)

    result = pd.concat(outputs, ignore_index=True)
    return result.sort_values(
        ["LINEAGE_NO", "TARGET_YEAR"],
        kind="stable",
        ignore_index=True,
    )


def validate_output(
    output: pd.DataFrame,
    child_history: pd.DataFrame,
    target_years: List[int],
) -> None:
    if list(output.columns) != OUTPUT_COLUMNS:
        raise RuntimeError("The output schema is not exactly the expected four columns.")
    if not set(output["ROLE"]).issubset({"FATHER", "MOTHER"}):
        raise RuntimeError("ROLE contains a value other than FATHER or MOTHER.")
    if not set(output["TARGET_YEAR"]).issubset(set(target_years)):
        raise RuntimeError("TARGET_YEAR contains an unexpected year.")
    if output["PAST_OFFSPRING_COUNT"].isna().any():
        raise RuntimeError("PAST_OFFSPRING_COUNT contains a missing value.")
    if not pd.api.types.is_integer_dtype(output["PAST_OFFSPRING_COUNT"]):
        raise RuntimeError("PAST_OFFSPRING_COUNT must be an integer column.")
    if output["PAST_OFFSPRING_COUNT"].lt(0).any():
        raise RuntimeError("PAST_OFFSPRING_COUNT contains a negative value.")
    if output.duplicated(["LINEAGE_NO", "ROLE", "TARGET_YEAR"]).any():
        raise RuntimeError("The output contains a duplicate lookup key.")
    if output["LINEAGE_NO"].isin(UNKNOWN_LINEAGE_CODES).any():
        raise RuntimeError("The output contains an unknown lineage sentinel code.")

    year_diff = output.groupby(
        ["ROLE", "LINEAGE_NO"],
        sort=False,
    )["PAST_OFFSPRING_COUNT"].diff()
    if year_diff.dropna().lt(0).any():
        raise RuntimeError("A cumulative offspring count decreases over time.")

    minimum_target_year = min(target_years)
    first_year = {
        "FATHER": child_history.groupby("FATHER_CATTLE_NO")[
            "CHILD_ABATT_YEAR"
        ].min(),
        "MOTHER": child_history.groupby("MOTHER_ANIMAL_NO")[
            "CHILD_ABATT_YEAR"
        ].min(),
    }
    for role, first_child_year in first_year.items():
        first_target_rows = output.loc[
            output["ROLE"].eq(role)
            & output["TARGET_YEAR"].eq(minimum_target_year)
        ].set_index("LINEAGE_NO")
        parents_without_prior_child = first_child_year[
            first_child_year.ge(minimum_target_year)
        ].index
        observed = first_target_rows.reindex(parents_without_prior_child)[
            "PAST_OFFSPRING_COUNT"
        ]
        if observed.dropna().ne(0).any():
            raise RuntimeError(
                "{} has a nonzero count before its first child year.".format(role)
            )

    maximum_target_year = max(target_years)
    role_columns = {
        "FATHER": "FATHER_CATTLE_NO",
        "MOTHER": "MOTHER_ANIMAL_NO",
    }
    for role, parent_column in role_columns.items():
        expected = (
            child_history.loc[
                child_history[parent_column].notna()
                & child_history["CHILD_ABATT_YEAR"].lt(maximum_target_year),
                ["CHILD_CATTLE_NO", parent_column],
            ]
            .drop_duplicates()
            .groupby(parent_column)
            .size()
        )
        actual = output.loc[
            output["ROLE"].eq(role)
            & output["TARGET_YEAR"].eq(maximum_target_year)
        ].set_index("LINEAGE_NO")["PAST_OFFSPRING_COUNT"]
        expected = expected.reindex(actual.index, fill_value=0).astype("int64")
        if not actual.equals(expected):
            raise RuntimeError(
                "{} final cumulative counts do not match source relations.".format(
                    role
                )
            )


def print_role_statistics(output: pd.DataFrame, role: str) -> None:
    values = output.loc[
        output["ROLE"].eq(role),
        "PAST_OFFSPRING_COUNT",
    ]
    zero_count = int(values.eq(0).sum())
    zero_pct = zero_count / len(values) * 100 if len(values) else 0.0
    print("{} PAST_OFFSPRING_COUNT:".format(role))
    print("  zero: {:,} ({:.4f}%)".format(zero_count, zero_pct))
    print("  mean: {:.4f}".format(values.mean()))
    print("  median: {:.4f}".format(values.median()))
    print("  max: {:,}".format(int(values.max())))


def main() -> None:
    train = pd.read_csv(
        TRAIN_PATH,
        usecols=["CATTLE_NO", "ABATT_DATE"],
        dtype={"CATTLE_NO": "string", "ABATT_DATE": "string"},
        encoding="utf-8-sig",
    )
    train_rows = len(train)
    train["CATTLE_NO"] = normalize_identifier(train["CATTLE_NO"])
    if train["CATTLE_NO"].isna().any():
        raise ValueError("train.CATTLE_NO contains an invalid identifier.")
    if train["CATTLE_NO"].duplicated().any():
        raise ValueError("train.CATTLE_NO must be unique.")
    train["CHILD_ABATT_DATE"] = parse_dates(train["ABATT_DATE"])
    train["CHILD_ABATT_YEAR"] = train["CHILD_ABATT_DATE"].dt.year.astype("Int64")
    target_years = collect_target_years(train["CHILD_ABATT_DATE"], TEST_PATH)

    child_dates = train[
        ["CATTLE_NO", "CHILD_ABATT_YEAR"]
    ].rename(columns={"CATTLE_NO": "CHILD_CATTLE_NO"})

    lineage = pd.read_csv(
        LINEAGE_PATH,
        usecols=["CATTLE_NO", "FATHER_CATTLE_NO", "MOTHER_ANIMAL_NO"],
        dtype="string",
        encoding="utf-8-sig",
    )
    lineage_rows = len(lineage)
    for column in lineage.columns:
        lineage[column] = normalize_identifier(lineage[column])
    if lineage["CATTLE_NO"].isna().any():
        raise ValueError("lineage.CATTLE_NO contains an invalid identifier.")
    if lineage["CATTLE_NO"].duplicated().any():
        raise ValueError("lineage.CATTLE_NO must be unique.")
    lineage = lineage.rename(columns={"CATTLE_NO": "CHILD_CATTLE_NO"})

    child_history = lineage.merge(
        child_dates,
        on="CHILD_CATTLE_NO",
        how="left",
        validate="one_to_one",
        indicator=True,
        sort=False,
    )
    if len(child_history) != lineage_rows:
        raise RuntimeError("Joining train dates changed the lineage row count.")

    matched_child_rows = int(child_history["_merge"].eq("both").sum())
    unmatched_child_rows = int(child_history["_merge"].eq("left_only").sum())
    invalid_child_date_rows = int(
        (
            child_history["_merge"].eq("both")
            & child_history["CHILD_ABATT_YEAR"].isna()
        ).sum()
    )
    child_history = child_history.drop(columns="_merge")

    father_output = build_role_output(
        child_history,
        "FATHER_CATTLE_NO",
        "FATHER",
        target_years,
    )
    mother_output = build_role_output(
        child_history,
        "MOTHER_ANIMAL_NO",
        "MOTHER",
        target_years,
    )
    output = pd.concat(
        [father_output, mother_output],
        ignore_index=True,
    )[OUTPUT_COLUMNS]
    output = output.sort_values(
        ["ROLE", "LINEAGE_NO", "TARGET_YEAR"],
        kind="stable",
        ignore_index=True,
    )

    validate_output(output, child_history, target_years)
    output.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    reloaded = pd.read_csv(
        OUTPUT_PATH,
        dtype={
            "LINEAGE_NO": "string",
            "ROLE": "string",
            "TARGET_YEAR": "int64",
            "PAST_OFFSPRING_COUNT": "int64",
        },
        encoding="utf-8-sig",
    )
    if len(reloaded) != len(output):
        raise RuntimeError("Reloaded CSV row count differs from generated output.")
    if list(reloaded.columns) != OUTPUT_COLUMNS:
        raise RuntimeError("Reloaded CSV columns differ from the expected schema.")

    valid_father_count = int(
        child_history["FATHER_CATTLE_NO"].dropna().nunique()
    )
    valid_mother_count = int(
        child_history["MOTHER_ANIMAL_NO"].dropna().nunique()
    )
    metadata = {
        "train_path": str(TRAIN_PATH),
        "lineage_path": str(LINEAGE_PATH),
        "test_path": str(TEST_PATH),
        "train_rows": train_rows,
        "lineage_rows": lineage_rows,
        "matched_child_rows": matched_child_rows,
        "unmatched_child_rows": unmatched_child_rows,
        "invalid_child_date_rows": invalid_child_date_rows,
        "valid_father_count": valid_father_count,
        "valid_mother_count": valid_mother_count,
        "target_years": target_years,
        "father_output_rows": len(father_output),
        "mother_output_rows": len(mother_output),
        "output_rows": len(output),
        "rule": "CHILD_ABATT_YEAR < TARGET_YEAR",
        "generated_at": datetime.now().astimezone().isoformat(),
        "source_files": {
            "train": file_metadata(TRAIN_PATH),
            "lineage": file_metadata(LINEAGE_PATH),
            "test": file_metadata(TEST_PATH),
        },
    }
    with METADATA_PATH.open("w", encoding="utf-8") as target:
        json.dump(metadata, target, ensure_ascii=False, indent=2)

    matched_pct = matched_child_rows / lineage_rows * 100 if lineage_rows else 0.0
    unmatched_pct = (
        unmatched_child_rows / lineage_rows * 100 if lineage_rows else 0.0
    )
    invalid_date_pct = (
        invalid_child_date_rows / matched_child_rows * 100
        if matched_child_rows
        else 0.0
    )
    print("train rows: {:,}".format(train_rows))
    print("lineage rows: {:,}".format(lineage_rows))
    print(
        "matched lineage children: {:,} ({:.4f}%)".format(
            matched_child_rows,
            matched_pct,
        )
    )
    print(
        "unmatched lineage children: {:,} ({:.4f}%)".format(
            unmatched_child_rows,
            unmatched_pct,
        )
    )
    print(
        "matched children with invalid ABATT_DATE: {:,} ({:.4f}%)".format(
            invalid_child_date_rows,
            invalid_date_pct,
        )
    )
    print("valid father lineages: {:,}".format(valid_father_count))
    print("valid mother lineages: {:,}".format(valid_mother_count))
    print("target years: {}".format(target_years))
    print("father output rows: {:,}".format(len(father_output)))
    print("mother output rows: {:,}".format(len(mother_output)))
    print("total output rows: {:,}".format(len(output)))
    print_role_statistics(output, "FATHER")
    print_role_statistics(output, "MOTHER")
    print("CSV written to: {}".format(OUTPUT_PATH))
    print("Metadata written to: {}".format(METADATA_PATH))


if __name__ == "__main__":
    main()
