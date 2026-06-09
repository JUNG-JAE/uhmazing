from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEATH_PATH = ROOT / "Data/hanwoo_original/hanwoo_death.csv"
SUMMARY_PATH = ROOT / "Data/hanwoo_original/hanwoo_death_summary.csv"

CATEGORY_COLUMNS = {
    "BIOSECURITY": "FARM_BIOSECURITY_DEATH_RATIO",
    "HEALTH": "FARM_HEALTH_DEATH_RATIO",
    "ACCIDENT": "FARM_ACCIDENT_DEATH_RATIO",
    "WEATHER": "FARM_WEATHER_DEATH_RATIO",
    "UNKNOWN": "FARM_UNKNOWN_DEATH_RATIO",
}
REASON_RECORD_RATIO_COLUMN = "FARM_DEATH_REASON_RECORD_RATIO"

MISSING_REASON_RE = re.compile(r"^\s*(?:-99(?:\.0)?)?\s*$")
BIOSECURITY_RE = re.compile(
    r"결핵|브루셀|브르셀|구제역|럼피|럼프스킨|법정전염병|광우병"
)
VACCINE_RE = re.compile(r"백신|접종|부작용|후유증|쇼크")
BIOSECURITY_EVENT_RE = re.compile(r"살처분|양성|항원|발생|감염|의양성|도태")
WEATHER_RE = re.compile(
    r"열사|일사병|폭염|혹서|고온|더위|동사|동절기|한파|혹한|추위|저체온|"
    r"폭우|호우|수해|침수|홍수|태풍|낙뢰|벼락|산불|폭설|대설|재해|기상|"
    r"강풍|우사붕괴|축사붕괴|온열질환|화상"
)
ACCIDENT_RE = re.compile(
    r"사고|압사|질식|골절|철책|철창|울타리|목.?끼|끼임|끼여|싸워|싸움|"
    r"밟혀|밟힘|외상|추락|미끄러|감전|교통|차량|충돌|뇌진탕|탈골|타박|"
    r"목줄|목걸림|목꺾임|포획|탈출|익사|물에 빠|깔려|깔림|찔려|찔림|"
    r"부딪|공격|상처|절단|부상|타격|차여"
)
HEALTH_RE = re.compile(
    r"설사|고창|장염|장파열|복막염|식체|급체|장독|장출혈|장폐|장꼬|장중첩|위염|"
    r"소화|복부|가스|변비|탈수|호흡|폐렴|기관지|기침|감기|패혈|폐혈|"
    r"보튤|보툴|브툴|파상풍|콕시|질병|일반병|병사|급사|돌연사|쇼크|심정지|심장|"
    r"간질환|간염|신장|신부전|요로|뇌염|수막염|유방염|자궁|산욕|사산|"
    r"난산|조산|분만|유산|질탈|후산|새끼.?낳|송아지|허약|약하게.?태어|"
    r"기립불능|아사|저체중|선천|제대|"
    r"백신|접종|부작용|노령|노환|노화|고령|자연사|안락사|도태|혈증|"
    r"중독|독혈|괴사|마비|경련|염증|부종|종양|암|빈혈|출혈|피똥|열병|발열|"
    r"식욕|영양|대사|요독|결석|탈장|궤양|농양|화농|감염|병약|쇠약|"
    r"초유|과식|사료|이물질|폐색|꼬임|염전|천공|파열|복통|황달|기생충|"
    r"진드기|로타바이러스|소모성|구내염|각막|실명|족부|제엽염|관절염|"
    r"척추|근육|산통|심부전|기도폐쇄|페렴"
)


def classify_reason(value: object) -> str:
    reason = "" if pd.isna(value) else str(value).strip()
    if MISSING_REASON_RE.fullmatch(reason):
        return "UNKNOWN"

    biosecurity = bool(BIOSECURITY_RE.search(reason))
    vaccine = bool(VACCINE_RE.search(reason))
    if biosecurity and (not vaccine or BIOSECURITY_EVENT_RE.search(reason)):
        return "BIOSECURITY"
    if WEATHER_RE.search(reason):
        return "WEATHER"
    if ACCIDENT_RE.search(reason):
        return "ACCIDENT"
    if HEALTH_RE.search(reason):
        return "HEALTH"
    return "UNKNOWN"


def main() -> None:
    death = pd.read_csv(
        DEATH_PATH,
        usecols=["FARM_UNIQUE_NO", "DEAD_REASON"],
        dtype={"FARM_UNIQUE_NO": "string", "DEAD_REASON": "string"},
        keep_default_na=False,
    )
    summary = pd.read_csv(SUMMARY_PATH, dtype={"FARM_UNIQUE_NO": "string"})

    if summary["FARM_UNIQUE_NO"].duplicated().any():
        raise ValueError("FARM_UNIQUE_NO is duplicated in the summary file.")

    death["CATEGORY"] = death["DEAD_REASON"].map(classify_reason)
    counts = (
        death.groupby(["FARM_UNIQUE_NO", "CATEGORY"], sort=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=CATEGORY_COLUMNS, fill_value=0)
        .rename(columns=CATEGORY_COLUMNS)
    )

    raw_farms = set(death["FARM_UNIQUE_NO"])
    summary_farms = set(summary["FARM_UNIQUE_NO"])
    if raw_farms != summary_farms:
        raise ValueError(
            "Farm sets differ between death and summary files: "
            f"raw_only={len(raw_farms - summary_farms)}, "
            f"summary_only={len(summary_farms - raw_farms)}"
        )

    generated_columns = set(CATEGORY_COLUMNS.values()) | {REASON_RECORD_RATIO_COLUMN}
    original_columns = [
        column for column in summary.columns if column not in generated_columns
    ]
    summary = summary[original_columns].merge(
        counts,
        left_on="FARM_UNIQUE_NO",
        right_index=True,
        how="left",
        validate="one_to_one",
        sort=False,
    )

    count_columns = list(CATEGORY_COLUMNS.values())
    if summary[count_columns].isna().any().any():
        raise ValueError("Category counts were not mapped to every summary farm.")

    reconstructed_total = summary[count_columns].sum(axis=1)
    raw_total = death.groupby("FARM_UNIQUE_NO").size().reindex(summary["FARM_UNIQUE_NO"])
    if not reconstructed_total.reset_index(drop=True).equals(
        raw_total.reset_index(drop=True)
    ):
        raise ValueError("Category counts do not reconstruct each farm's death total.")

    known_columns = [
        column
        for category, column in CATEGORY_COLUMNS.items()
        if category != "UNKNOWN"
    ]
    summary[REASON_RECORD_RATIO_COLUMN] = (
        summary[known_columns].sum(axis=1) / reconstructed_total
    ).round(6)
    if not summary[REASON_RECORD_RATIO_COLUMN].between(0, 1).all():
        raise ValueError("Death reason record ratios must be between 0 and 1.")

    # A category with no records is represented by the source-data sentinel value.
    summary[count_columns] = summary[count_columns].replace(0, -99).astype("int64")
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    print(f"Updated: {SUMMARY_PATH}")
    print(f"Farms: {len(summary):,}")
    print("Death records by category:")
    for category, column in CATEGORY_COLUMNS.items():
        total = int(summary[column].replace(-99, 0).sum())
        print(f"  {category}: {total:,}")


if __name__ == "__main__":
    main()
