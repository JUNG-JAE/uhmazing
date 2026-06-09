from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "Data" / "hanwoo_original"
TRAIN_PATH = DATA_DIR / "hanwoo_train.csv"
SMARTFARM_PATH = DATA_DIR / "hanwoo_smartfarm_grade.xlsx"
OUTPUT_PATH = DATA_DIR / "hanwoo_slaughter.csv"

TRAIN_COLUMNS = [
    "sido",
    "sigungu",
    "eupmyeondong",
    "stn",
    "ABATT_DATE",
    "JUDGE_DATE",
    "JUDGE_SEX",
    "WEIGHT",
    "BACKFAT",
    "REA",
    "WINDEX",
    "WGRADE",
    "INSFAT",
    "YUKSAK",
    "FATSAK",
    "TISSUE",
    "GROWTH",
    "COST_AMT",
    "AGE",
    "BIRTH_YMD",
    "CATTLE_NO",
    "FARM_UNIQUE_NO",
    "LAST_GRADE",
]
OUTPUT_COLUMNS = TRAIN_COLUMNS + ["SOURCE"]
DATE_COLUMNS = ["ABATT_DATE", "JUDGE_DATE", "BIRTH_YMD"]
NUMERIC_OR_MISSING_COLUMNS = [
    "WEIGHT",
    "BACKFAT",
    "REA",
    "WINDEX",
    "INSFAT",
    "YUKSAK",
    "FATSAK",
    "TISSUE",
    "GROWTH",
    "AGE",
]
MISSING_VALUE = "-99"
MISSING_TOKENS = {"", "nan", "NaN", "None", "NONE", "-99", "-99.0"}


def is_missing_scalar(value: object) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip() in MISSING_TOKENS


def normalize_text_series(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").str.strip()
    missing = normalized.isna() | normalized.isin(MISSING_TOKENS)
    return normalized.mask(missing, MISSING_VALUE)


def normalize_date_series(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").str.strip()
    normalized = normalized.mask(normalized.isna() | normalized.isin(MISSING_TOKENS))
    parsed = pd.to_datetime(normalized, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d").fillna(MISSING_VALUE)


def normalize_numeric_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.astype("object").where(numeric.notna(), MISSING_VALUE)


def normalize_train_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    output = chunk.copy()
    for column in DATE_COLUMNS:
        output[column] = normalize_date_series(output[column])
    output = output.fillna(MISSING_VALUE)
    output["SOURCE"] = "original"
    return output[OUTPUT_COLUMNS]


def parse_breeding_address(values: pd.Series) -> pd.DataFrame:
    normalized = normalize_text_series(values)
    parts = normalized.str.split()
    return pd.DataFrame(
        {
            "sido": parts.str[0].fillna(MISSING_VALUE),
            "sigungu": parts.str[1].fillna(MISSING_VALUE),
            "eupmyeondong": parts.str[2].fillna(MISSING_VALUE),
        }
    ).replace({"<NA>": MISSING_VALUE})


def normalize_quality_grade(values: pd.Series) -> pd.Series:
    mapping = {
        "1++등급": "1++",
        "1+등급": "1+",
        "1등급": "1",
        "2등급": "2",
        "3등급": "3",
        "등외등급": "등외",
        "등외": "등외",
    }
    normalized = normalize_text_series(values)
    return normalized.map(mapping).fillna(normalized)


def normalize_quantity_grade(values: pd.Series) -> pd.Series:
    normalized = normalize_text_series(values)
    valid = {"A", "B", "C", "D", MISSING_VALUE}
    return normalized.where(normalized.isin(valid), MISSING_VALUE)


def normalize_last_grade(smartfarm: pd.DataFrame) -> pd.Series:
    quality = normalize_quality_grade(smartfarm["육질등급"])
    quantity = normalize_quantity_grade(smartfarm["육량등급"])
    combined = quality.where(
        quality.eq("등외") | quality.eq(MISSING_VALUE),
        quality + quantity,
    )
    combined = combined.where(
        ~quality.eq(MISSING_VALUE) & ~quantity.eq(MISSING_VALUE),
        quality.where(quality.eq("등외"), MISSING_VALUE),
    )
    return combined


def prefix_identifier(prefix: str, values: pd.Series) -> pd.Series:
    normalized = normalize_text_series(values)
    normalized = normalized.str.replace(r"\.0$", "", regex=True)
    return normalized.where(
        normalized.eq(MISSING_VALUE),
        prefix + normalized,
    )


def build_smartfarm_rows() -> pd.DataFrame:
    usecols = [
        "indvdNo",
        "slauDe_dt",
        "carcassWt",
        "backFatThick",
        "loinEyeAr",
        "meatQuantityIdex",
        "marbling",
        "meatColor",
        "fatColor",
        "contextDgree",
        "mtrdg",
        "lastGrad",
        "출생년월일",
        "개월령",
        "성별",
        "육질등급",
        "육량등급",
        "최초 농장식별번호",
        "최초 신고 사육지",
        "최종 농장식별번호",
        "최종 신고 사육지",
    ]
    smartfarm = pd.read_excel(
        SMARTFARM_PATH,
        sheet_name="raw_data_joined",
        usecols=usecols,
    )

    breeding_address = smartfarm["최종 신고 사육지"].where(
        ~smartfarm["최종 신고 사육지"].map(is_missing_scalar),
        smartfarm["최초 신고 사육지"],
    )
    parsed_address = parse_breeding_address(breeding_address)

    farm_no = smartfarm["최종 농장식별번호"].where(
        ~smartfarm["최종 농장식별번호"].map(is_missing_scalar),
        smartfarm["최초 농장식별번호"],
    )

    output = pd.DataFrame(index=smartfarm.index, columns=TRAIN_COLUMNS)
    output[["sido", "sigungu", "eupmyeondong"]] = parsed_address
    output["stn"] = MISSING_VALUE
    output["ABATT_DATE"] = normalize_date_series(smartfarm["slauDe_dt"])
    output["JUDGE_DATE"] = output["ABATT_DATE"]
    output["JUDGE_SEX"] = normalize_text_series(smartfarm["성별"])
    output["WEIGHT"] = normalize_numeric_series(smartfarm["carcassWt"])
    output["BACKFAT"] = normalize_numeric_series(smartfarm["backFatThick"])
    output["REA"] = normalize_numeric_series(smartfarm["loinEyeAr"])
    output["WINDEX"] = normalize_numeric_series(smartfarm["meatQuantityIdex"])
    output["WGRADE"] = normalize_quantity_grade(smartfarm["육량등급"])
    output["INSFAT"] = normalize_numeric_series(smartfarm["marbling"])
    output["YUKSAK"] = normalize_numeric_series(smartfarm["meatColor"])
    output["FATSAK"] = normalize_numeric_series(smartfarm["fatColor"])
    output["TISSUE"] = normalize_numeric_series(smartfarm["contextDgree"])
    output["GROWTH"] = normalize_numeric_series(smartfarm["mtrdg"])
    output["COST_AMT"] = MISSING_VALUE
    output["AGE"] = normalize_numeric_series(smartfarm["개월령"])
    output["BIRTH_YMD"] = normalize_date_series(smartfarm["출생년월일"])
    output["CATTLE_NO"] = prefix_identifier("SMARTFARM_", smartfarm["indvdNo"])
    output["FARM_UNIQUE_NO"] = prefix_identifier("SMARTFARM_FARM_", farm_no)
    output["LAST_GRADE"] = normalize_last_grade(smartfarm)
    output["SOURCE"] = "smartfarm"

    return output[OUTPUT_COLUMNS].fillna(MISSING_VALUE)


def write_slaughter_file(smartfarm_rows: pd.DataFrame) -> int:
    train_rows = 0
    first_chunk = True
    for chunk in pd.read_csv(
        TRAIN_PATH,
        usecols=TRAIN_COLUMNS,
        dtype="string",
        chunksize=200_000,
        encoding="utf-8-sig",
    ):
        train_rows += len(chunk)
        normalized = normalize_train_chunk(chunk)
        normalized.to_csv(
            OUTPUT_PATH,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
            encoding="utf-8-sig",
        )
        first_chunk = False

    smartfarm_rows.to_csv(
        OUTPUT_PATH,
        mode="a",
        header=False,
        index=False,
        encoding="utf-8-sig",
    )
    return train_rows


def count_csv_rows(path: Path) -> int:
    count = 0
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            count += block.count(b"\n")
    return max(count - 1, 0)


def validate_dates(values: pd.Series) -> bool:
    pattern = re.compile(r"^(\d{4}-\d{2}-\d{2}|-99)$")
    return values.astype(str).map(lambda value: bool(pattern.match(value))).all()


def validate_numeric_or_missing(values: pd.Series) -> bool:
    normalized = values.astype(str).str.strip()
    valid_missing = normalized.eq(MISSING_VALUE)
    numeric = pd.to_numeric(normalized.where(~valid_missing), errors="coerce")
    return (valid_missing | numeric.notna()).all()


def validate_output(train_rows: int, smartfarm_rows: pd.DataFrame) -> None:
    header = pd.read_csv(OUTPUT_PATH, nrows=0, encoding="utf-8-sig")
    if list(header.columns) != OUTPUT_COLUMNS:
        raise RuntimeError("Output columns do not match train columns plus SOURCE.")

    output_rows = count_csv_rows(OUTPUT_PATH)
    expected_rows = train_rows + len(smartfarm_rows)
    if output_rows != expected_rows:
        raise RuntimeError(
            "Unexpected row count: expected {:,}, got {:,}".format(
                expected_rows,
                output_rows,
            )
        )

    source_counts = pd.read_csv(
        OUTPUT_PATH,
        usecols=["SOURCE"],
        dtype="string",
        encoding="utf-8-sig",
    )["SOURCE"].value_counts(dropna=False)
    if set(source_counts.index.astype(str)) != {"original", "smartfarm"}:
        raise RuntimeError("SOURCE contains a value other than original/smartfarm.")
    if int(source_counts["original"]) != train_rows:
        raise RuntimeError("original SOURCE row count does not match train rows.")
    if int(source_counts["smartfarm"]) != len(smartfarm_rows):
        raise RuntimeError("smartfarm SOURCE row count does not match smartfarm rows.")

    smart = smartfarm_rows.astype("string")
    if not smart["CATTLE_NO"].eq(MISSING_VALUE).all() and not smart.loc[
        ~smart["CATTLE_NO"].eq(MISSING_VALUE),
        "CATTLE_NO",
    ].str.startswith("SMARTFARM_").all():
        raise RuntimeError("A smartfarm CATTLE_NO does not have the SMARTFARM_ prefix.")
    if not smart.loc[
        ~smart["FARM_UNIQUE_NO"].eq(MISSING_VALUE),
        "FARM_UNIQUE_NO",
    ].str.startswith("SMARTFARM_FARM_").all():
        raise RuntimeError(
            "A smartfarm FARM_UNIQUE_NO does not have the SMARTFARM_FARM_ prefix."
        )
    for column in DATE_COLUMNS:
        if not validate_dates(smart[column]):
            raise RuntimeError("{} contains a non-normalized date.".format(column))
    for column in NUMERIC_OR_MISSING_COLUMNS:
        if not validate_numeric_or_missing(smart[column]):
            raise RuntimeError("{} contains a non-numeric value.".format(column))

    valid_wgrade = {"A", "B", "C", "D", MISSING_VALUE}
    if not set(smart["WGRADE"].astype(str)).issubset(valid_wgrade):
        raise RuntimeError("WGRADE contains a value outside the train value system.")
    valid_last_grade = {
        "1++A",
        "1++B",
        "1++C",
        "1+A",
        "1+B",
        "1+C",
        "1A",
        "1B",
        "1C",
        "2A",
        "2B",
        "2C",
        "3A",
        "3B",
        "3C",
        "등외",
        MISSING_VALUE,
    }
    if not set(smart["LAST_GRADE"].astype(str)).issubset(valid_last_grade):
        raise RuntimeError("LAST_GRADE contains a value outside the train value system.")


def print_summary(train_rows: int, smartfarm_rows: int) -> None:
    output_rows = count_csv_rows(OUTPUT_PATH)
    source_counts = pd.read_csv(
        OUTPUT_PATH,
        usecols=["SOURCE"],
        dtype="string",
        encoding="utf-8-sig",
    )["SOURCE"].value_counts(dropna=False)

    print("Created:", OUTPUT_PATH)
    print("train rows: {:,}".format(train_rows))
    print("smartfarm rows: {:,}".format(smartfarm_rows))
    print("output rows: {:,}".format(output_rows))
    print("SOURCE counts:")
    print(source_counts.to_string())


def main() -> int:
    train_header = pd.read_csv(TRAIN_PATH, nrows=0, encoding="utf-8-sig")
    if list(train_header.columns) != TRAIN_COLUMNS:
        raise RuntimeError("hanwoo_train.csv schema is not the expected schema.")

    smartfarm_rows = build_smartfarm_rows()
    train_rows = write_slaughter_file(smartfarm_rows)
    validate_output(train_rows, smartfarm_rows)
    print_summary(train_rows, len(smartfarm_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
