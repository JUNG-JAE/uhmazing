from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "Data" / "hanwoo_original"
TRAIN_PATH = DATA_DIR / "hanwoo_train.csv"
LINEAGE_PATH = DATA_DIR / "hanwoo_lineage.csv"
OFFSPRING_PATH = DATA_DIR / "hanwoo_lineage_offspring_by_year.csv"
SAME_FARM_PATH = DATA_DIR / "hanwoo_same_farm_candidates.csv"
OUTPUT_PATH = DATA_DIR / "hanwoo_lineage_farm_offspring_by_year.csv"

MISSING_TOKENS = {"", "-99", "-99.0", "MISSING"}
UNKNOWN_LINEAGE_CODES = {
    "kluWj1LiM8I6nYWfDenO7q4tJySB2AVV8z9cMqweuXA=",
    "gQagjD++POKUI4kyvXKUoA==",
    "2XwK0r9Ij2yaHcePqO7Bwg==",
}
LINEAGE_COLUMNS = [
    "CATTLE_NO",
    "FATHER_CATTLE_NO",
    "MOTHER_ANIMAL_NO",
    "F_GMOTHER_ANIMAL_NO",
    "F_GFATHER_CATTLE_NO",
    "M_GMOTHER_ANIMAL_NO",
    "M_GFATHER_CATTLE_NO",
]
OUTPUT_COLUMNS = [
    "FARM_UNIQUE_NO",
    "TARGET_YEAR",
    "FARM_YEAR_FATHER_OFFSPRING_MEAN",
    "FARM_YEAR_FATHER_SAMPLE_COUNT",
    "FARM_YEAR_MOTHER_OFFSPRING_MEAN",
    "FARM_YEAR_MOTHER_SAMPLE_COUNT",
]
MIN_FARM_YEAR_IMPUTATION_SAMPLES = 3


def normalize_identifier(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").str.strip()
    invalid = (
        normalized.isna()
        | normalized.isin(MISSING_TOKENS)
        | normalized.isin(UNKNOWN_LINEAGE_CODES)
    )
    return normalized.mask(invalid)


def resolve_parent_ancestry(
    lineage: pd.DataFrame,
    parent_column: str,
    father_column: str,
    mother_column: str,
) -> pd.DataFrame:
    rows = lineage[
        [parent_column, father_column, mother_column]
    ].dropna(subset=[parent_column])
    resolved = rows.groupby(parent_column, sort=False).agg(
        father_count=(father_column, "nunique"),
        mother_count=(mother_column, "nunique"),
        father=(father_column, "first"),
        mother=(mother_column, "first"),
    )
    resolved["conflict"] = (
        resolved["father_count"].gt(1)
        | resolved["mother_count"].gt(1)
    )
    resolved.loc[resolved["father_count"].ne(1), "father"] = pd.NA
    resolved.loc[resolved["mother_count"].ne(1), "mother"] = pd.NA
    return resolved


def restore_parent_lineage(
    train: pd.DataFrame,
    lineage: pd.DataFrame,
) -> pd.DataFrame:
    direct = lineage.set_index("CATTLE_NO")[
        ["FATHER_CATTLE_NO", "MOTHER_ANIMAL_NO"]
    ]
    result = train.copy()
    result["LINEAGE_FATHER_NO"] = result["CATTLE_NO"].map(
        direct["FATHER_CATTLE_NO"]
    )
    result["LINEAGE_MOTHER_NO"] = result["CATTLE_NO"].map(
        direct["MOTHER_ANIMAL_NO"]
    )

    direct_mask = result["CATTLE_NO"].isin(direct.index)
    father_ancestry = resolve_parent_ancestry(
        lineage,
        "FATHER_CATTLE_NO",
        "F_GFATHER_CATTLE_NO",
        "F_GMOTHER_ANIMAL_NO",
    )
    mother_ancestry = resolve_parent_ancestry(
        lineage,
        "MOTHER_ANIMAL_NO",
        "M_GFATHER_CATTLE_NO",
        "M_GMOTHER_ANIMAL_NO",
    )

    unresolved = ~direct_mask
    has_father_match = result["CATTLE_NO"].isin(father_ancestry.index)
    has_mother_match = result["CATTLE_NO"].isin(mother_ancestry.index)
    fallback_specs = [
        (
            unresolved & has_father_match & ~has_mother_match,
            father_ancestry,
        ),
        (
            unresolved & has_mother_match & ~has_father_match,
            mother_ancestry,
        ),
    ]
    for candidate_mask, ancestry in fallback_specs:
        candidate_ids = result.loc[candidate_mask, "CATTLE_NO"]
        usable = ~candidate_ids.map(ancestry["conflict"]).fillna(True)
        usable_index = candidate_ids.index[usable]
        result.loc[usable_index, "LINEAGE_FATHER_NO"] = result.loc[
            usable_index, "CATTLE_NO"
        ].map(ancestry["father"])
        result.loc[usable_index, "LINEAGE_MOTHER_NO"] = result.loc[
            usable_index, "CATTLE_NO"
        ].map(ancestry["mother"])

    return result


def map_offspring_counts(
    train: pd.DataFrame,
    offspring: pd.DataFrame,
    role: str,
    parent_column: str,
    output_column: str,
) -> None:
    role_lookup = offspring.loc[
        offspring["ROLE"].eq(role),
        ["LINEAGE_NO", "TARGET_YEAR", "PAST_OFFSPRING_COUNT"],
    ].set_index(["LINEAGE_NO", "TARGET_YEAR"])["PAST_OFFSPRING_COUNT"]

    valid = train[parent_column].notna() & train["TARGET_YEAR"].notna()
    output = pd.Series(pd.NA, index=train.index, dtype="Float64")
    if valid.any():
        keys = pd.MultiIndex.from_arrays(
            [
                train.loc[valid, parent_column],
                train.loc[valid, "TARGET_YEAR"].astype("int64"),
            ],
            names=["LINEAGE_NO", "TARGET_YEAR"],
        )
        output.loc[valid] = role_lookup.reindex(keys).fillna(0).to_numpy()
    train[output_column] = output


def load_integrated_farm_mapping(
    train_farms: set,
) -> Tuple[pd.Series, pd.Series, Dict[str, int]]:
    candidates = pd.read_csv(
        SAME_FARM_PATH,
        usecols=[
            "CANDIDATE_GROUP_ID",
            "CANDIDATE_STRENGTH",
            "FARM_UNIQUE_NO",
        ],
        dtype="string",
        encoding="utf-8-sig",
    )
    for column in candidates.columns:
        candidates[column] = candidates[column].astype("string").str.strip()
    candidates["CANDIDATE_STRENGTH"] = (
        candidates["CANDIDATE_STRENGTH"].str.upper()
    )

    invalid_candidate = (
        candidates["CANDIDATE_GROUP_ID"].isna()
        | candidates["FARM_UNIQUE_NO"].isna()
    )
    invalid_candidate_count = int(invalid_candidate.sum())
    candidates = candidates.loc[~invalid_candidate].drop_duplicates(
        ["CANDIDATE_GROUP_ID", "CANDIDATE_STRENGTH", "FARM_UNIQUE_NO"]
    )
    integrated = candidates.loc[
        candidates["CANDIDATE_STRENGTH"].isin({"HIGH", "MEDIUM"})
    ].copy()

    groups_per_farm = integrated.groupby("FARM_UNIQUE_NO")[
        "CANDIDATE_GROUP_ID"
    ].nunique()
    if groups_per_farm.gt(1).any():
        raise ValueError(
            "{} farms belong to multiple HIGH/MEDIUM groups".format(
                int(groups_per_farm.gt(1).sum())
            )
        )
    strengths_per_group = integrated.groupby("CANDIDATE_GROUP_ID")[
        "CANDIDATE_STRENGTH"
    ].nunique()
    if strengths_per_group.gt(1).any():
        raise ValueError(
            "{} groups mix HIGH and MEDIUM strengths".format(
                int(strengths_per_group.gt(1).sum())
            )
        )

    absent_from_train = int(
        (~integrated["FARM_UNIQUE_NO"].isin(train_farms)).sum()
    )
    group_map = integrated.set_index("FARM_UNIQUE_NO")[
        "CANDIDATE_GROUP_ID"
    ]
    strength_map = integrated.set_index("FARM_UNIQUE_NO")[
        "CANDIDATE_STRENGTH"
    ]
    stats = {
        "invalid_candidate_rows": invalid_candidate_count,
        "candidate_farms_absent_from_train": absent_from_train,
        "high_groups": int(
            integrated.loc[
                integrated["CANDIDATE_STRENGTH"].eq("HIGH"),
                "CANDIDATE_GROUP_ID",
            ].nunique()
        ),
        "high_farms": int(
            integrated["CANDIDATE_STRENGTH"].eq("HIGH").sum()
        ),
        "medium_groups": int(
            integrated.loc[
                integrated["CANDIDATE_STRENGTH"].eq("MEDIUM"),
                "CANDIDATE_GROUP_ID",
            ].nunique()
        ),
        "medium_farms": int(
            integrated["CANDIDATE_STRENGTH"].eq("MEDIUM").sum()
        ),
    }
    return group_map, strength_map, stats


def aggregate_farm_year(
    train: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    aggregated = train.groupby(
        [group_column, "TARGET_YEAR"],
        dropna=False,
        sort=False,
    ).agg(
        FARM_YEAR_FATHER_OFFSPRING_MEAN=(
            "FATHER_PAST_OFFSPRING_COUNT",
            "mean",
        ),
        FARM_YEAR_FATHER_SAMPLE_COUNT=(
            "FATHER_PAST_OFFSPRING_COUNT",
            "count",
        ),
        FARM_YEAR_MOTHER_OFFSPRING_MEAN=(
            "MOTHER_PAST_OFFSPRING_COUNT",
            "mean",
        ),
        FARM_YEAR_MOTHER_SAMPLE_COUNT=(
            "MOTHER_PAST_OFFSPRING_COUNT",
            "count",
        ),
    )
    return aggregated.reset_index()


def count_imputable_rows(
    train: pd.DataFrame,
    lookup: pd.DataFrame,
    key_column: str,
    value_column: str,
    mean_column: str,
    sample_column: str,
) -> int:
    joined = train[
        [key_column, "TARGET_YEAR", value_column]
    ].merge(
        lookup[[key_column, "TARGET_YEAR", mean_column, sample_column]],
        on=[key_column, "TARGET_YEAR"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    return int(
        (
            joined[value_column].isna()
            & joined[mean_column].notna()
            & joined[sample_column].ge(MIN_FARM_YEAR_IMPUTATION_SAMPLES)
        ).sum()
    )


def print_sample_count_statistics(
    output: pd.DataFrame,
    label: str,
    sample_column: str,
) -> None:
    values = output[sample_column]
    print("{} sample-count groups:".format(label))
    print("  1: {:,}".format(int(values.eq(1).sum())))
    print("  2: {:,}".format(int(values.eq(2).sum())))
    print("  >=3: {:,}".format(int(values.ge(3).sum())))
    print("  >=5: {:,}".format(int(values.ge(5).sum())))


def main() -> None:
    train = pd.read_csv(
        TRAIN_PATH,
        usecols=["CATTLE_NO", "FARM_UNIQUE_NO", "ABATT_DATE"],
        dtype="string",
        encoding="utf-8-sig",
    )
    train_rows = len(train)
    train["CATTLE_NO"] = normalize_identifier(train["CATTLE_NO"])
    train["FARM_UNIQUE_NO"] = normalize_identifier(train["FARM_UNIQUE_NO"])
    train["TARGET_YEAR"] = pd.to_datetime(
        train["ABATT_DATE"],
        errors="coerce",
    ).dt.year.astype("Int64")
    if train["CATTLE_NO"].isna().any():
        raise ValueError("train.CATTLE_NO contains invalid identifiers")
    if train["CATTLE_NO"].duplicated().any():
        raise ValueError("train.CATTLE_NO must be unique")

    lineage = pd.read_csv(
        LINEAGE_PATH,
        usecols=LINEAGE_COLUMNS,
        dtype="string",
        encoding="utf-8-sig",
    )
    for column in LINEAGE_COLUMNS:
        lineage[column] = normalize_identifier(lineage[column])
    if lineage["CATTLE_NO"].isna().any():
        raise ValueError("lineage.CATTLE_NO contains invalid identifiers")
    if lineage["CATTLE_NO"].duplicated().any():
        raise ValueError("lineage.CATTLE_NO must be unique")

    train = restore_parent_lineage(train, lineage)
    offspring = pd.read_csv(
        OFFSPRING_PATH,
        dtype={
            "LINEAGE_NO": "string",
            "ROLE": "string",
            "TARGET_YEAR": "int16",
            "PAST_OFFSPRING_COUNT": "int32",
        },
        encoding="utf-8-sig",
    )
    if offspring.duplicated(["LINEAGE_NO", "ROLE", "TARGET_YEAR"]).any():
        raise ValueError("offspring lookup keys must be unique")
    map_offspring_counts(
        train,
        offspring,
        "FATHER",
        "LINEAGE_FATHER_NO",
        "FATHER_PAST_OFFSPRING_COUNT",
    )
    map_offspring_counts(
        train,
        offspring,
        "MOTHER",
        "LINEAGE_MOTHER_NO",
        "MOTHER_PAST_OFFSPRING_COUNT",
    )

    train_farms = set(train["FARM_UNIQUE_NO"].dropna())
    group_map, strength_map, candidate_stats = load_integrated_farm_mapping(
        train_farms
    )
    mapped_group = train["FARM_UNIQUE_NO"].map(group_map)
    train["FARM_GROUP_KEY"] = (
        "FARM:" + train["FARM_UNIQUE_NO"].fillna("MISSING")
    )
    grouped = mapped_group.notna()
    train.loc[grouped, "FARM_GROUP_KEY"] = (
        "GROUP:" + mapped_group.loc[grouped]
    )
    train["FARM_GROUP_STRENGTH"] = (
        train["FARM_UNIQUE_NO"].map(strength_map).fillna("INDIVIDUAL")
    )

    integrated_lookup = aggregate_farm_year(train, "FARM_GROUP_KEY")
    individual_train = train.copy()
    individual_train["INDIVIDUAL_FARM_KEY"] = (
        "FARM:" + individual_train["FARM_UNIQUE_NO"].fillna("MISSING")
    )
    individual_lookup = aggregate_farm_year(
        individual_train,
        "INDIVIDUAL_FARM_KEY",
    )

    farm_year_keys = train[
        ["FARM_UNIQUE_NO", "TARGET_YEAR", "FARM_GROUP_KEY"]
    ].drop_duplicates(["FARM_UNIQUE_NO", "TARGET_YEAR"])
    output = farm_year_keys.merge(
        integrated_lookup,
        on=["FARM_GROUP_KEY", "TARGET_YEAR"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    output = output[OUTPUT_COLUMNS].sort_values(
        ["FARM_UNIQUE_NO", "TARGET_YEAR"],
        kind="stable",
        ignore_index=True,
    )
    mean_columns = [
        "FARM_YEAR_FATHER_OFFSPRING_MEAN",
        "FARM_YEAR_MOTHER_OFFSPRING_MEAN",
    ]
    sample_columns = [
        "FARM_YEAR_FATHER_SAMPLE_COUNT",
        "FARM_YEAR_MOTHER_SAMPLE_COUNT",
    ]
    output[mean_columns] = output[mean_columns].round(4)
    output[sample_columns] = output[sample_columns].fillna(0).astype("int64")

    if output.duplicated(["FARM_UNIQUE_NO", "TARGET_YEAR"]).any():
        raise RuntimeError("output contains duplicate farm-year keys")
    expected_years = set(train["TARGET_YEAR"].dropna().astype(int))
    if not set(output["TARGET_YEAR"].dropna().astype(int)).issubset(
        expected_years
    ):
        raise RuntimeError("output contains an unexpected TARGET_YEAR")
    for mean_column, sample_column in zip(mean_columns, sample_columns):
        invalid_mean = output[mean_column].notna() & output[mean_column].lt(0)
        if invalid_mean.any():
            raise RuntimeError("{} contains a negative value".format(mean_column))
        mismatch = output[mean_column].isna() != output[sample_column].eq(0)
        if mismatch.any():
            raise RuntimeError(
                "{} and {} disagree about availability".format(
                    mean_column,
                    sample_column,
                )
            )

    # Every member of the same integrated group must receive identical values.
    validation = train[
        ["FARM_UNIQUE_NO", "TARGET_YEAR", "FARM_GROUP_KEY"]
    ].drop_duplicates().merge(
        output,
        on=["FARM_UNIQUE_NO", "TARGET_YEAR"],
        how="left",
        validate="one_to_one",
    )
    grouped_validation = validation.loc[
        validation["FARM_GROUP_KEY"].str.startswith("GROUP:", na=False)
    ]
    for column in mean_columns + sample_columns:
        inconsistent = grouped_validation.groupby(
            ["FARM_GROUP_KEY", "TARGET_YEAR"]
        )[column].nunique(dropna=False).gt(1)
        if inconsistent.any():
            raise RuntimeError(
                "Integrated group members disagree for {}".format(column)
            )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    reloaded = pd.read_csv(
        OUTPUT_PATH,
        dtype={
            "FARM_UNIQUE_NO": "string",
            "TARGET_YEAR": "int64",
            "FARM_YEAR_FATHER_OFFSPRING_MEAN": "float64",
            "FARM_YEAR_FATHER_SAMPLE_COUNT": "int64",
            "FARM_YEAR_MOTHER_OFFSPRING_MEAN": "float64",
            "FARM_YEAR_MOTHER_SAMPLE_COUNT": "int64",
        },
        encoding="utf-8-sig",
    )
    if len(reloaded) != len(output) or list(reloaded.columns) != OUTPUT_COLUMNS:
        raise RuntimeError("reloaded output does not match generated output")

    individual_father = count_imputable_rows(
        individual_train,
        individual_lookup,
        "INDIVIDUAL_FARM_KEY",
        "FATHER_PAST_OFFSPRING_COUNT",
        "FARM_YEAR_FATHER_OFFSPRING_MEAN",
        "FARM_YEAR_FATHER_SAMPLE_COUNT",
    )
    individual_mother = count_imputable_rows(
        individual_train,
        individual_lookup,
        "INDIVIDUAL_FARM_KEY",
        "MOTHER_PAST_OFFSPRING_COUNT",
        "FARM_YEAR_MOTHER_OFFSPRING_MEAN",
        "FARM_YEAR_MOTHER_SAMPLE_COUNT",
    )
    integrated_father = count_imputable_rows(
        train,
        integrated_lookup,
        "FARM_GROUP_KEY",
        "FATHER_PAST_OFFSPRING_COUNT",
        "FARM_YEAR_FATHER_OFFSPRING_MEAN",
        "FARM_YEAR_FATHER_SAMPLE_COUNT",
    )
    integrated_mother = count_imputable_rows(
        train,
        integrated_lookup,
        "FARM_GROUP_KEY",
        "MOTHER_PAST_OFFSPRING_COUNT",
        "FARM_YEAR_MOTHER_OFFSPRING_MEAN",
        "FARM_YEAR_MOTHER_SAMPLE_COUNT",
    )

    father_missing = int(train["FATHER_PAST_OFFSPRING_COUNT"].isna().sum())
    mother_missing = int(train["MOTHER_PAST_OFFSPRING_COUNT"].isna().sum())
    print("train rows: {:,}".format(train_rows))
    print("farm-year output rows: {:,}".format(len(output)))
    print("invalid candidate rows excluded: {:,}".format(
        candidate_stats["invalid_candidate_rows"]
    ))
    print("candidate farms absent from train: {:,}".format(
        candidate_stats["candidate_farms_absent_from_train"]
    ))
    print(
        "HIGH groups/farms: {:,}/{:,}".format(
            candidate_stats["high_groups"],
            candidate_stats["high_farms"],
        )
    )
    print(
        "MEDIUM groups/farms: {:,}/{:,}".format(
            candidate_stats["medium_groups"],
            candidate_stats["medium_farms"],
        )
    )
    for strength in ["HIGH", "MEDIUM", "INDIVIDUAL"]:
        row_count = int(train["FARM_GROUP_STRENGTH"].eq(strength).sum())
        print("{} train rows: {:,}".format(strength, row_count))
    print(
        "father mean available groups: {:,}".format(
            int(output["FARM_YEAR_FATHER_OFFSPRING_MEAN"].notna().sum())
        )
    )
    print(
        "mother mean available groups: {:,}".format(
            int(output["FARM_YEAR_MOTHER_OFFSPRING_MEAN"].notna().sum())
        )
    )
    print_sample_count_statistics(
        output,
        "FATHER",
        "FARM_YEAR_FATHER_SAMPLE_COUNT",
    )
    print_sample_count_statistics(
        output,
        "MOTHER",
        "FARM_YEAR_MOTHER_SAMPLE_COUNT",
    )
    print(
        "father missing: {:,}; individual imputable: {:,}; "
        "integrated imputable: {:,}; added by integration: {:,}".format(
            father_missing,
            individual_father,
            integrated_father,
            integrated_father - individual_father,
        )
    )
    print(
        "mother missing: {:,}; individual imputable: {:,}; "
        "integrated imputable: {:,}; added by integration: {:,}".format(
            mother_missing,
            individual_mother,
            integrated_mother,
            integrated_mother - individual_mother,
        )
    )
    print("output written to: {}".format(OUTPUT_PATH))


if __name__ == "__main__":
    main()
