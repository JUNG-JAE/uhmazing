import argparse
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


"""
    === 실행 방법 ===
    data_builder.py --dataset-type train --save-type parquet
    data_builder.py --dataset-type test --save-type csv

    === 입력 데이터 ===
    sido: 시도
    sigungu: 시군구
    eupmyeondong: 읍면동
    stn: 기상관측 코드
    
    ABATT_DATE: 도축일
    JUDGE_DATE: 등급판정일
    JUDGE_SEX: 판정 성별
    WEIGHT: 도체중량
    AGE: 도축시 월령
    BIRTH_YMD: 출생일
    CATTLE_NO: 한우 개체 식별 번호
    FARM_UNIQUE_NO: 농가 식별 번호
    
    === 예측 데이터 ===
    BACKFAT: 등지방 두께 (regression)
    REA: 등심 단면적 (regression)
    
    INSFAT: 근내 지방도 (5 class classification)
    YUKSAK: 육색 (5 class classification)
    FATSAK: 지방색 (5 class classification)
    TISSUE: 조직감 (5 class classification)
    GROWTH: 성숙도 (binary class classification)
    
    === Derived 데이터 ===
    WINDEX: 육량 지수 (실제 예측에서 제외. WEIGHT, BACKFAT, REA, JUDGE_SEX로 계산)
    WGRADE: 육량 등급 (실제 예측에서 제외. WINDEX와 JUDGE_SEX별 기준으로 결정)
    LAST_GRADE: 최종 등급 (실제 예측에서 제외. 육량지수, 육량 등급으로 계산)
"""

# ============================================================
# 디렉토리 경로 설정
# ============================================================
ROOT_DIR = Path(__file__).resolve().parents[1]
# TRAIN_DATA_PATH = ROOT_DIR / "Data" / "hanwoo_original" / "hanwoo_train.csv"

# Main(Train, Test) 데이터 경로
TRAIN_DATA_PATH = ROOT_DIR / "Data" / "hanwoo_sample" / "sample_hanwoo_train.csv"
TEST_DATA_PATH = ROOT_DIR / "Data" / "hanwoo_original" / "hanwoo_test.csv"

# 한우 혈통 데이터 경로
LINEAGE_DATA_PATH = ROOT_DIR / "Data" / "hanwoo_original" / "hanwoo_lineage.csv" # 원본 혈통
LINEAGE_OFFSPRING_DATA_PATH = ROOT_DIR / "Data" / "hanwoo_original" / "hanwoo_lineage_offspring.csv" # 년도별 직계 부모의 자식수
LINEAGE_FARM_OFFSPRING_DATA_PATH = ROOT_DIR / "Data" / "hanwoo_original" / "hanwoo_lineage_farm_avg_offspring.csv" # 농가기준 년도별 직계 부모의 자식수 (결측 보간용)
SLAUGHTER_DATA_PATH = ROOT_DIR / "Data" / "hanwoo_original" / "hanwoo_slaughter.csv" # 도축일 기준 이전 평균 한우 등급 (년도, 계절, 나이 고려함)

# 한우 농가 데이터 경로
AREA_DATA_PATH = ROOT_DIR / "Data" / "hanwoo_original" / "hanwoo_area_with_avg.csv" # 농가 면적 및 평균 한우 수
DEATH_DATA_PATH = ROOT_DIR / "Data" / "hanwoo_original" / "hanwoo_death_summary.csv" # 농가별 한우 평균 사망 및 질병
SAME_FARM_CANDIDATES_PATH = ROOT_DIR / "Data" / "hanwoo_original" / "hanwoo_same_farm_candidates.csv" # 실제로 같은 농가 통합

# 통합된 데이터 출력 경로
PROCESSED_DATA_DIR = ROOT_DIR / "Data" / "processed"


# ============================================================
# hanwoo_train.csv
# ============================================================
X_COLUMNS = ["sido", "sigungu", "eupmyeondong", "stn", "ABATT_DATE", "JUDGE_DATE", "JUDGE_SEX", "WEIGHT", "AGE", "BIRTH_YMD", "CATTLE_NO", "FARM_UNIQUE_NO"]
SEASON_COLUMNS = ["ABATT_SEASON", "BIRTH_SEASON"]
Y_COLUMNS = ["BACKFAT", "REA", "INSFAT", "YUKSAK", "FATSAK", "TISSUE", "GROWTH"]
GRADE_CLASS_COLUMNS = ["INSFAT_GRADE_CLASS", "YUKSAK_GRADE_CLASS", "FATSAK_GRADE_CLASS", "TISSUE_GRADE_CLASS", "GROWTH_STATUS_CLASS"]
DERIVED_COLUMNS = ["WINDEX", "WGRADE", "LAST_GRADE"]

MISSING_TOKENS = {"", "-99", "-99.0", "MISSING"}
UNKNOWN_LINEAGE_CODES = {"kluWj1LiM8I6nYWfDenO7q4tJySB2AVV8z9cMqweuXA=", "gQagjD++POKUI4kyvXKUoA==", "2XwK0r9Ij2yaHcePqO7Bwg=="} # 혈통 데이터에서 반복되고 오류로 존재하는 값.

LINEAGE_SOURCE_COLUMNS = ["CATTLE_NO", "FATHER_CATTLE_NO", "MOTHER_ANIMAL_NO", "F_GMOTHER_ANIMAL_NO", "F_GFATHER_CATTLE_NO", "M_GMOTHER_ANIMAL_NO", "M_GFATHER_CATTLE_NO"]
LINEAGE_VALUE_COLUMNS = ["LINEAGE_FATHER_NO", "LINEAGE_MOTHER_NO"]
LINEAGE_OUTPUT_COLUMNS = LINEAGE_VALUE_COLUMNS
LINEAGE_OFFSPRING_OUTPUT_COLUMNS = ["FATHER_PAST_OFFSPRING_COUNT", "MOTHER_PAST_OFFSPRING_COUNT"]
LINEAGE_OFFSPRING_IMPUTATION_COLUMNS = ["FATHER_OFFSPRING_IMPUTED", "MOTHER_OFFSPRING_IMPUTED"]

AREA_OUTPUT_COLUMNS = ["FARM_AVG_HANWOO_COUNT", "FARM_AREA", "HANWOO_DENSITY"]
DEATH_OUTPUT_COLUMNS = ["FARM_DEATH_AVG_COUNT", "FARM_BIOSECURITY_STATUS", "FARM_HEALTH_STATUS", "FARM_ACCIDENT_STATUS"]
DEATH_STATUS_COLUMNS = DEATH_OUTPUT_COLUMNS[1:]
DEATH_REASON_COUNT_COLUMNS = {
    "FARM_BIOSECURITY_DEATH_RATIO": "FARM_BIOSECURITY_STATUS",
    "FARM_HEALTH_DEATH_RATIO": "FARM_HEALTH_STATUS",
    "FARM_ACCIDENT_DEATH_RATIO": "FARM_ACCIDENT_STATUS",
}
DEATH_YEAR_COLUMNS = ["2023", "2024", "2025"]
DEATH_RECORD_RATIO_THRESHOLD = 0.8
MIN_FARM_YEAR_IMPUTATION_SAMPLES = 3 # 혈통 자손 수 결측값을 농가·연도 평균으로 대체하기 위한 최소 표본 수

SLAUGHTER_CONTINUOUS_COLUMNS = ["WEIGHT", "BACKFAT", "REA", "WINDEX"]
SLAUGHTER_WGRADE_LEVELS = ["A", "B", "C", "D"]
SLAUGHTER_LAST_GRADE_LEVELS = ["1++A", "1++B", "1++C", "1+A", "1+B", "1+C", "1A", "1B", "1C", "2A", "2B", "2C", "3A", "3B", "3C", "등외"]
SLAUGHTER_CLASS_FEATURES = [
    ("INSFAT_GRADE_CLASS", "PAST_INSFAT_CLASS"),
    ("YUKSAK_GRADE_CLASS", "PAST_YUKSAK_CLASS"),
    ("FATSAK_GRADE_CLASS", "PAST_FATSAK_CLASS"),
    ("TISSUE_GRADE_CLASS", "PAST_TISSUE_CLASS"),
]
SLAUGHTER_BASE_OUTPUT_COLUMNS = ["PAST_SLAUGHTER_SAMPLE_COUNT", "PAST_SLAUGHTER_HISTORY_YEARS", "PAST_SLAUGHTER_AVAILABLE"]
SLAUGHTER_CONTINUOUS_OUTPUT_COLUMNS = [
    "PAST_{}_{}".format(column, stat)
    for column in SLAUGHTER_CONTINUOUS_COLUMNS
    for stat in ["MEAN", "VARIANCE"]
]
SLAUGHTER_WGRADE_OUTPUT_COLUMNS = ["PAST_WGRADE_{}_RATIO".format(level) for level in SLAUGHTER_WGRADE_LEVELS]
SLAUGHTER_LAST_GRADE_OUTPUT_COLUMNS = [
    "PAST_LAST_GRADE_{}_RATIO".format(
        level.replace("++", "PP").replace("+", "P").replace("등외", "OUT")
    )
    for level in SLAUGHTER_LAST_GRADE_LEVELS
]
SLAUGHTER_CLASS_OUTPUT_COLUMNS = [
    "{}_{}_RATIO".format(prefix, class_value)
    for _, prefix in SLAUGHTER_CLASS_FEATURES
    for class_value in range(5)
]
SLAUGHTER_GROWTH_OUTPUT_COLUMNS = ["PAST_GROWTH_NORMAL_RATIO", "PAST_GROWTH_ABNORMAL_RATIO"]
SLAUGHTER_STAT_COLUMNS = (
    SLAUGHTER_CONTINUOUS_OUTPUT_COLUMNS
    + SLAUGHTER_WGRADE_OUTPUT_COLUMNS
    + SLAUGHTER_LAST_GRADE_OUTPUT_COLUMNS
    + SLAUGHTER_CLASS_OUTPUT_COLUMNS
    + SLAUGHTER_GROWTH_OUTPUT_COLUMNS
)
SLAUGHTER_OUTPUT_COLUMNS = SLAUGHTER_BASE_OUTPUT_COLUMNS + SLAUGHTER_STAT_COLUMNS
AUXILIARY_OUTPUT_COLUMNS = (
    SEASON_COLUMNS
    + LINEAGE_OUTPUT_COLUMNS
    + LINEAGE_OFFSPRING_OUTPUT_COLUMNS
    + LINEAGE_OFFSPRING_IMPUTATION_COLUMNS
    + AREA_OUTPUT_COLUMNS
    + DEATH_OUTPUT_COLUMNS
    + SLAUGHTER_OUTPUT_COLUMNS
)


def add_grade_class_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    INSFAT, YUKSAK, FATSAK, TISSUE, GROWTH 클래스 변환 함수.
    INSFAT, YUKSAK, FATSAK, TISSUE (1~9)의 클래스를 1~5개의 등급별 클래스로 변환 (0=1++ level, 1=1+ level, 2=1 level, 3=2 level, 4=3 level)
    GROWTH (1~9)의 클래스를 0,1 클래스로 변환 (정상, 결격)
    
    main_data와 slaughter_data에 사용됨.  
    """
    
    required_columns = {"INSFAT", "YUKSAK", "FATSAK", "TISSUE", "GROWTH"}
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError("Data is missing grade score columns: {}".format(missing_columns))

    result = data.copy()

    def numeric_score(column: str) -> Tuple[pd.Series, pd.Series]:
        raw = result[column].astype("string").str.strip()
        missing = raw.isna() | raw.isin(MISSING_TOKENS)
        numeric = pd.to_numeric(raw, errors="coerce")
        return numeric, missing

    def missing_series() -> pd.Series:
        return pd.Series("MISSING", index=result.index, dtype=object)

    # 등급 변환은 6_등급판정기준.md 참조
    insfat, insfat_missing = numeric_score("INSFAT")
    insfat_class = missing_series()
    insfat_class.loc[insfat.isin([7, 8, 9])] = 0
    insfat_class.loc[insfat.eq(6)] = 1
    insfat_class.loc[insfat.isin([4, 5])] = 2
    insfat_class.loc[insfat.isin([2, 3])] = 3
    insfat_class.loc[insfat.eq(1)] = 4
    insfat_class.loc[insfat_missing] = "MISSING"
    result["INSFAT_GRADE_CLASS"] = insfat_class

    yuksak, yuksak_missing = numeric_score("YUKSAK")
    yuksak_class = pd.Series(4, index=result.index, dtype=object)
    yuksak_class.loc[yuksak.isin([3, 4, 5])] = 0
    yuksak_class.loc[yuksak.isin([2, 6])] = 1
    yuksak_class.loc[yuksak.eq(1)] = 2
    yuksak_class.loc[yuksak.eq(7)] = 3
    yuksak_class.loc[yuksak_missing] = "MISSING"
    result["YUKSAK_GRADE_CLASS"] = yuksak_class

    fatsak, fatsak_missing = numeric_score("FATSAK")
    fatsak_class = pd.Series(4, index=result.index, dtype=object)
    fatsak_class.loc[fatsak.isin([1, 2, 3, 4])] = 0
    fatsak_class.loc[fatsak.eq(5)] = 1
    fatsak_class.loc[fatsak.eq(6)] = 2
    fatsak_class.loc[fatsak.eq(7)] = 3
    fatsak_class.loc[fatsak_missing] = "MISSING"
    result["FATSAK_GRADE_CLASS"] = fatsak_class

    tissue, tissue_missing = numeric_score("TISSUE")
    tissue_class = missing_series()
    tissue_class.loc[tissue.eq(1)] = 0
    tissue_class.loc[tissue.eq(2)] = 1
    tissue_class.loc[tissue.eq(3)] = 2
    tissue_class.loc[tissue.eq(4)] = 3
    tissue_class.loc[tissue.eq(5)] = 4
    tissue_class.loc[tissue_missing] = "MISSING"
    result["TISSUE_GRADE_CLASS"] = tissue_class

    growth, growth_missing = numeric_score("GROWTH")
    growth_class = missing_series()
    growth_class.loc[growth.isin([1, 2, 3, 4, 5, 6, 7])] = 0
    growth_class.loc[growth.isin([8, 9])] = 1
    growth_class.loc[growth_missing] = "MISSING"
    result["GROWTH_STATUS_CLASS"] = growth_class

    return result


class DataBuilder:
    def __init__(
        self, dataset_type: str = "train", # train 또는 test 데이터셋 만들건지 결정
        min_birth_date: Optional[str] = None,
        main_data_path: Optional[Path] = None,
        slaughter_data_path: Path = SLAUGHTER_DATA_PATH,
        lineage_data_path: Path = LINEAGE_DATA_PATH,
        lineage_offspring_data_path: Path = LINEAGE_OFFSPRING_DATA_PATH,
        lineage_farm_offspring_data_path: Path = LINEAGE_FARM_OFFSPRING_DATA_PATH,
        area_data_path: Path = AREA_DATA_PATH,
        death_data_path: Path = DEATH_DATA_PATH,
        same_farm_candidates_path: Path = SAME_FARM_CANDIDATES_PATH,
        processed_data_dir: Path = PROCESSED_DATA_DIR,
    ) -> None:
        
        normalized_dataset_type = dataset_type.strip().lower()
        if normalized_dataset_type not in {"train", "test"}:
            raise ValueError("dataset_type must be either 'train' or 'test'")
        if normalized_dataset_type == "test" and min_birth_date is not None:
            raise ValueError("test data cannot be filtered by min_birth_date")

        self.dataset_type = normalized_dataset_type
        self.min_birth_date = (pd.Timestamp(min_birth_date) if min_birth_date is not None else (pd.Timestamp("2020-01-01") if normalized_dataset_type == "train" else None))
        default_main_path = (TRAIN_DATA_PATH if normalized_dataset_type == "train" else TEST_DATA_PATH)
        self.main_data_path = Path(main_data_path or default_main_path)
        self.slaughter_data_path = Path(slaughter_data_path)
        self.lineage_data_path = Path(lineage_data_path)
        self.lineage_offspring_data_path = Path(lineage_offspring_data_path)
        self.lineage_farm_offspring_data_path = Path(lineage_farm_offspring_data_path)
        self.area_data_path = Path(area_data_path)
        self.death_data_path = Path(death_data_path)
        self.same_farm_candidates_path = Path(same_farm_candidates_path)
        self.processed_data_dir = Path(processed_data_dir)
        self._source_row_count: Optional[int] = None
        self._source_cattle_order: Optional[pd.Series] = None

    @staticmethod
    def _date_to_korean_season(dates: pd.Series) -> pd.Series:
        month = dates.dt.month
        season = pd.Series("MISSING", index=dates.index, dtype=object)
        season.loc[month.isin([3, 4, 5])] = "봄"
        season.loc[month.isin([6, 7, 8])] = "여름"
        season.loc[month.isin([9, 10, 11])] = "가을"
        season.loc[month.isin([12, 1, 2])] = "겨울"
        
        return season

    @staticmethod
    def _parse_date_series(values: pd.Series) -> pd.Series:
        """
        문자열 -> datetime64 변환, 코드 내부 날짜 처리용
        """
        normalized = values.astype("string").str.strip()
        missing = normalized.isna() | normalized.isin(MISSING_TOKENS)
        normalized = normalized.mask(missing)

        compact_date = normalized.str.fullmatch(r"\d{8}", na=False)
        parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
        if compact_date.any():
            parsed.loc[compact_date] = pd.to_datetime(normalized.loc[compact_date], format="%Y%m%d", errors="coerce")

        unresolved = parsed.isna() & normalized.notna()
        separated_date = (normalized.str.fullmatch(r"\d{4}[./-]\d{1,2}[./-]\d{1,2}", na=False) & unresolved)
        if separated_date.any():
            standardized = normalized.loc[separated_date].str.replace(r"[./]", "-", regex=True)
            parsed.loc[separated_date] = pd.to_datetime(standardized, format="%Y-%m-%d", errors="coerce")

        unresolved = parsed.isna() & normalized.notna()
        if unresolved.any():
            parsed.loc[unresolved] = pd.to_datetime(normalized.loc[unresolved], errors="coerce")
        
        return parsed

    @staticmethod
    def _format_date_series(dates: pd.Series) -> pd.Series:
        """ 
        저장 날짜 포맷 통일
        20240105   -> 2024-01-05
        2024-01-05 -> 2024-01-05
        2024/01/05 -> 2024-01-05
        2024.1.5   -> 2024-01-05
        """
        parsed = pd.to_datetime(dates, errors="coerce")
        
        return parsed.dt.strftime("%Y-%m-%d").astype("string").fillna("MISSING")

    @staticmethod
    def _validate_data_row_ids(data: pd.DataFrame, context: str) -> None:
        if "DATA_ROW_ID" not in data.columns:
            raise RuntimeError("{} is missing DATA_ROW_ID".format(context))
        if data["DATA_ROW_ID"].isna().any():
            raise RuntimeError("{} contains missing DATA_ROW_ID".format(context))
        if data["DATA_ROW_ID"].duplicated().any():
            raise RuntimeError("{} contains duplicate DATA_ROW_ID".format(context))

    def _run_auxiliary_loader(self, loader, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        DATA_ROW_ID 가 정상인지 검증하는 코드
        """
        loader_name = getattr(loader, "__name__", str(loader))
        self._validate_data_row_ids(data, "{} input".format(loader_name))
        before_ids = data["DATA_ROW_ID"].astype("string").tolist()
        before_x = data[X_COLUMNS].astype("string").fillna("<NA>").copy()

        result = loader(data, *args, **kwargs)

        if len(result) != len(data):
            raise RuntimeError("{} changed row count: {} -> {}".format(loader_name, len(data), len(result)))
        self._validate_data_row_ids(result, "{} output".format(loader_name))
        if result["DATA_ROW_ID"].astype("string").tolist() != before_ids:
            raise RuntimeError("{} changed DATA_ROW_ID order".format(loader_name))
        after_x = result[X_COLUMNS].astype("string").fillna("<NA>")
        if not before_x.equals(after_x):
            raise RuntimeError("{} modified original X columns".format(loader_name))
        
        return result

    # ============================================================
    # main data loader. train 또는 test
    # ============================================================
    def main_data_loader(self) -> pd.DataFrame:
        df = pd.read_csv(self.main_data_path, encoding="utf-8-sig")
        loaded_count = len(df)
        row_prefix = self.dataset_type.upper()
        df.insert(0, "DATA_ROW_ID", ["{}_{:09d}".format(row_prefix, index) for index in range(loaded_count)]) # 각 행을 고유하게 식별하고, 처리 과정에서 중복, 누락, 순서 변경을 검증용
        if df["DATA_ROW_ID"].duplicated().any():
            raise RuntimeError("DATA_ROW_ID must be unique immediately after loading")

        self._source_row_count = loaded_count
        self._source_cattle_order = df["CATTLE_NO"].astype("string").copy()

        required_columns = ["DATA_ROW_ID"] + X_COLUMNS
        if self.dataset_type == "train":
            required_columns += Y_COLUMNS + DERIVED_COLUMNS
        
        missing_columns = sorted(set(required_columns) - set(df.columns))
        if missing_columns:
            raise ValueError("Missing columns in {}: {}".format(self.main_data_path, missing_columns))

        df = df[required_columns].copy()
        missing_filtered_count = 0
        if self.dataset_type == "train":
            target_columns = Y_COLUMNS + DERIVED_COLUMNS
            target_missing = df[target_columns].isna()
            target_missing |= df[target_columns].astype(str).apply(lambda column: column.str.strip().isin(MISSING_TOKENS))
            missing_row_mask = target_missing.any(axis=1)
            missing_filtered_count = int(missing_row_mask.sum())
            df = df.loc[~missing_row_mask].copy()

        birth_dates = self._parse_date_series(df["BIRTH_YMD"])
        abatt_dates = self._parse_date_series(df["ABATT_DATE"])
        judge_dates = self._parse_date_series(df["JUDGE_DATE"])

        df["BIRTH_YMD"] = self._format_date_series(birth_dates)
        df["ABATT_DATE"] = self._format_date_series(abatt_dates)
        df["JUDGE_DATE"] = self._format_date_series(judge_dates)
        df["BIRTH_DATE"] = self._format_date_series(birth_dates)

        df["BIRTH_DATE_DT"] = birth_dates
        df["ABATT_DATE_DT"] = abatt_dates
        df["JUDGE_DATE_DT"] = judge_dates
        df["ABATT_SEASON"] = self._date_to_korean_season(df["ABATT_DATE_DT"])
        df["BIRTH_SEASON"] = self._date_to_korean_season(df["BIRTH_DATE_DT"])

        year_filtered_count = 0
        # train 데이터 생성. 결측치 및 날짜 기준 필터링 적용
        if self.dataset_type == "train":
            # BIRTH_DATE_DT, ABATT_DATE_DT 결측값 있는지 검사. -> 날씨 및 과거 도축 정보 매핑을 위해 필수. BIRTH_DATE_DT는 최소 날짜를 만족해야함.
            year_filter_mask = (df["BIRTH_DATE_DT"].isna() | df["ABATT_DATE_DT"].isna() | (df["BIRTH_DATE_DT"] < self.min_birth_date))
            year_filtered_count = int(year_filter_mask.sum())
            df = df.loc[~year_filter_mask].copy()
            original_grade_scores = df[["INSFAT", "YUKSAK", "FATSAK", "TISSUE", "GROWTH"]].copy(deep=True)
            df = add_grade_class_columns(df)
            if not df[["INSFAT", "YUKSAK", "FATSAK", "TISSUE", "GROWTH"]].equals(original_grade_scores):
                raise RuntimeError("Grade class mapping modified original score columns")

        total_filtered_count = loaded_count - len(df)
        total_filtered_pct = (total_filtered_count / loaded_count * 100 if loaded_count else 0.0)

        print("=== Main data statistics ===")
        print("Dataset type: {}".format(self.dataset_type))
        print("Main data rows loaded: {:,}".format(loaded_count))
        print("Rows filtered by missing Y/derived values: {:,}".format(missing_filtered_count))
        print("Rows filtered by cattle birth year: {:,}".format(year_filtered_count))
        print("Total rows filtered: {:,} ({:.4f}%)".format(total_filtered_count, total_filtered_pct))

        result = df.reset_index(drop=True)
        if result["DATA_ROW_ID"].duplicated().any():
            raise RuntimeError("main_data_loader produced duplicate DATA_ROW_ID values")
        
        # test 데이터 생성. 필터링 없음. 원본 데이터 유지
        if self.dataset_type == "test":
            if len(result) != loaded_count:
                raise RuntimeError("test main_data_loader removed rows")
            expected_ids = ["TEST_{:09d}".format(index) for index in range(loaded_count)] # 데이터 순서, 중복, 누락 체크용
            
            if result["DATA_ROW_ID"].tolist() != expected_ids:
                raise RuntimeError("test DATA_ROW_ID order does not match source order")
            if not result["CATTLE_NO"].astype("string").reset_index(drop=True).equals(self._source_cattle_order.reset_index(drop=True)):
                raise RuntimeError("test CATTLE_NO order does not match source order")

        return result

    
    # ============================================================
    # 한우 혈통 보조 전처리 함수
    # ============================================================
    @staticmethod
    def _normalize_lineage_identifier(column: pd.Series) -> pd.Series:
        """ 
        혈통 CATTLE_NO 식별, 오류 CATTLE_NO 필터링
        """
        normalized = column.astype("string").str.strip()
        invalid_mask = (normalized.isna() | normalized.isin(MISSING_TOKENS) | normalized.isin(UNKNOWN_LINEAGE_CODES))
        
        return normalized.mask(invalid_mask)

    @staticmethod
    def _resolve_parent_match(rows: pd.DataFrame, parent_column: str, father_column: str, mother_column: str) -> dict:
        """ 
        부모 개체의 혈통 정보 역추적
        동일한 자식에 대해서 다른 부모가 2번 이상 등장할 경우 잘못된 값으로 판단하여 결측 처리함. (해쉬 함수 오류 가능성 존재)
        """
        resolved = {}
        for cattle_no, group in rows.groupby(parent_column, sort=False):
            father_values = group[father_column].dropna().unique()
            mother_values = group[mother_column].dropna().unique()
            conflict = len(father_values) > 1 or len(mother_values) > 1
            resolved[cattle_no] = {
                "father": father_values[0] if len(father_values) == 1 else "MISSING",
                "mother": mother_values[0] if len(mother_values) == 1 else "MISSING",
                "conflict": conflict,
            }
            
        return resolved

    # ============================================================
    # 한우 혈통 보조 데이터로더
    # ============================================================
    def aux_lineage_data_loader(self, main_data: pd.DataFrame) -> pd.DataFrame:
        """
        메인 데이터의 개체번호를 혈통 데이터와 매칭해 아버지와어머니 번호를 추가한다.
        직접 매칭되지 않으면 해당 개체가 부모로 기록된 정보를 이용해 혈통을 역추적한다. (할아버지, 할머니 정보를 부모로 사용함)
        부모 정보가 충돌하거나 확인되지 않으면 MISSING으로 처리한다.
        """
        
        if "CATTLE_NO" not in main_data.columns:
            raise ValueError("main_data is missing lineage join column: CATTLE_NO")

        before_count = len(main_data)
        lineage = pd.read_csv(self.lineage_data_path, usecols=LINEAGE_SOURCE_COLUMNS, dtype="string", encoding="utf-8-sig")
        missing_columns = sorted(set(LINEAGE_SOURCE_COLUMNS) - set(lineage.columns))
        
        if missing_columns:
            raise ValueError("Missing columns in {}: {}".format(self.lineage_data_path, missing_columns))
        if lineage["CATTLE_NO"].isna().any():
            raise ValueError("lineage.CATTLE_NO contains missing values")
        if lineage["CATTLE_NO"].duplicated().any():
            raise ValueError("lineage.CATTLE_NO must be unique")

        for column in LINEAGE_SOURCE_COLUMNS:
            lineage[column] = self._normalize_lineage_identifier(lineage[column])

        result = main_data.copy()
        main_ids = self._normalize_lineage_identifier(result["CATTLE_NO"])
        valid_main_ids = set(main_ids.dropna())
        records = {cattle_no: {column: "MISSING" for column in LINEAGE_VALUE_COLUMNS} for cattle_no in valid_main_ids}
        
        direct_rows = lineage.loc[lineage["CATTLE_NO"].isin(valid_main_ids)].set_index("CATTLE_NO")
        direct_mapping = {"LINEAGE_FATHER_NO": "FATHER_CATTLE_NO", "LINEAGE_MOTHER_NO": "MOTHER_ANIMAL_NO"}
        for cattle_no, row in direct_rows.iterrows():
            record = records[cattle_no]
            for output_column, source_column in direct_mapping.items():
                value = row[source_column]
                record[output_column] = value if pd.notna(value) else "MISSING"

        unresolved_ids = valid_main_ids - set(direct_rows.index)
        father_rows = lineage.loc[
            lineage["FATHER_CATTLE_NO"].isin(unresolved_ids),
            ["FATHER_CATTLE_NO", "F_GFATHER_CATTLE_NO", "F_GMOTHER_ANIMAL_NO"]]
        mother_rows = lineage.loc[
            lineage["MOTHER_ANIMAL_NO"].isin(unresolved_ids),
            ["MOTHER_ANIMAL_NO", "M_GFATHER_CATTLE_NO", "M_GMOTHER_ANIMAL_NO"]]
        
        father_matches = self._resolve_parent_match(father_rows, "FATHER_CATTLE_NO", "F_GFATHER_CATTLE_NO", "F_GMOTHER_ANIMAL_NO")
        mother_matches = self._resolve_parent_match(mother_rows, "MOTHER_ANIMAL_NO", "M_GFATHER_CATTLE_NO", "M_GMOTHER_ANIMAL_NO")

        for cattle_no in unresolved_ids:
            father_match = father_matches.get(cattle_no)
            mother_match = mother_matches.get(cattle_no)
            record = records[cattle_no]

            if father_match and mother_match:
                continue
            match = father_match or mother_match
            if match is None:
                continue
            if match["conflict"]:
                continue

            record["LINEAGE_FATHER_NO"] = match["father"]
            record["LINEAGE_MOTHER_NO"] = match["mother"]

        lineage_result = pd.DataFrame.from_dict(records, orient="index")
        lineage_result.index.name = "CATTLE_NO"
        for column in LINEAGE_OUTPUT_COLUMNS:
            if lineage_result.empty:
                result[column] = "MISSING"
            else:
                result[column] = main_ids.map(lineage_result[column]).fillna("MISSING")

        after_count = len(result)
        if after_count != before_count:
            raise RuntimeError("Lineage join changed row count: {} -> {}".format(before_count, after_count))
        
        print("\n=== Lineage auxiliary data statistics ===")
        direct_missing_count = int(main_ids.isna().sum()+ (~main_ids.dropna().isin(direct_rows.index)).sum())
        direct_missing_pct = (direct_missing_count / before_count * 100 if before_count else 0.0)
        
        print("CATTLE_NO MATCH MISSING: {:,} ({:.4f}%)".format(direct_missing_count, direct_missing_pct)) # train.csv에는 있지만 lineage.csv에는 CATTLE_NO가 없는 경우
        for column in LINEAGE_OUTPUT_COLUMNS:
            missing_count = int(result[column].eq("MISSING").sum())
            missing_pct = missing_count / before_count * 100 if before_count else 0.0
            print("{} MISSING: {:,} ({:.4f}%)".format(column, missing_count, missing_pct))

        return result

    # ============================================================
    # 한우 혈통 -> 자식 수 변환
    # ============================================================    
    def aux_lineage_resolve_offspring(self, main_data: pd.DataFrame) -> pd.DataFrame:
        required_main_columns = {"ABATT_DATE", "LINEAGE_FATHER_NO", "LINEAGE_MOTHER_NO"}
        
        missing_main_columns = sorted(required_main_columns - set(main_data.columns))
        if missing_main_columns:
            raise ValueError("main_data is missing lineage offspring columns: {}".format(missing_main_columns))

        before_count = len(main_data)
        original_order = main_data.index.to_series().reset_index(drop=True)
        result = main_data.copy()
        if "ABATT_DATE_DT" in result.columns:
            target_year = result["ABATT_DATE_DT"].dt.year.astype("Int64")
        else:
            target_year = self._parse_date_series(result["ABATT_DATE"]).dt.year.astype("Int64")

        father_ids = self._normalize_lineage_identifier(result["LINEAGE_FATHER_NO"]).mask(result["LINEAGE_FATHER_NO"].astype("string").str.strip().eq("MISSING"))
        mother_ids = self._normalize_lineage_identifier(result["LINEAGE_MOTHER_NO"]).mask(result["LINEAGE_MOTHER_NO"].astype("string").str.strip().eq("MISSING"))

        required_lookup_columns = {"LINEAGE_NO", "ROLE", "TARGET_YEAR", "PAST_OFFSPRING_COUNT"}
        target_father_ids = set(father_ids.dropna())
        target_mother_ids = set(mother_ids.dropna())
        target_years = set(target_year.dropna().astype(int))
        matching_chunks = []

        for chunk in pd.read_csv(
            self.lineage_offspring_data_path,
            usecols=list(required_lookup_columns),
            dtype={"LINEAGE_NO": "string", "ROLE": "string", "TARGET_YEAR": "int64", "PAST_OFFSPRING_COUNT": "int64"},
            encoding="utf-8-sig",
            chunksize=250_000,
        ):
            missing_lookup_columns = sorted(required_lookup_columns - set(chunk.columns))
            if missing_lookup_columns:
                raise ValueError("Missing columns in {}: {}".format(self.lineage_offspring_data_path, missing_lookup_columns))

            chunk["LINEAGE_NO"] = self._normalize_lineage_identifier(chunk["LINEAGE_NO"])
            chunk["ROLE"] = chunk["ROLE"].str.strip().str.upper()
            
            invalid_roles = set(chunk["ROLE"].dropna()) - {"FATHER", "MOTHER"}
            if invalid_roles:
                raise ValueError("Lineage offspring lookup contains invalid ROLE values: {}".format(sorted(invalid_roles)))

            needed = chunk["TARGET_YEAR"].isin(target_years) & ((chunk["ROLE"].eq("FATHER") & chunk["LINEAGE_NO"].isin(target_father_ids)) | (chunk["ROLE"].eq("MOTHER") & chunk["LINEAGE_NO"].isin(target_mother_ids)))
            if needed.any():
                matching_chunks.append(chunk.loc[needed].copy())

        if matching_chunks:
            lookup = pd.concat(matching_chunks, ignore_index=True)
        else:
            lookup = pd.DataFrame(columns=list(required_lookup_columns))

        if lookup.duplicated(["LINEAGE_NO", "ROLE", "TARGET_YEAR"]).any():
            raise ValueError("Lineage offspring lookup contains duplicate mapping keys")
        
        if not lookup.empty:
            invalid_counts = (lookup["PAST_OFFSPRING_COUNT"].isna() | lookup["PAST_OFFSPRING_COUNT"].lt(0))
            if invalid_counts.any():
                raise ValueError("Lineage offspring lookup contains invalid offspring counts")

        lookup_series = lookup.set_index(["LINEAGE_NO", "ROLE", "TARGET_YEAR"])["PAST_OFFSPRING_COUNT"]

        def map_offspring_count(parent_ids: pd.Series, role: str) -> pd.Series:
            output = pd.Series("MISSING", index=result.index, dtype=object)
            valid = parent_ids.notna() & target_year.notna()
            if not valid.any():
                return output

            keys = pd.MultiIndex.from_arrays(
                [
                    parent_ids.loc[valid].astype("string"),
                    pd.Series(role, index=parent_ids.loc[valid].index),
                    target_year.loc[valid].astype("int64"),
                ],
                names=["LINEAGE_NO", "ROLE", "TARGET_YEAR"],
            )
            mapped = lookup_series.reindex(keys).fillna(0).astype("int64")
            output.loc[valid] = mapped.to_numpy()
            return output

        result["FATHER_PAST_OFFSPRING_COUNT"] = map_offspring_count(father_ids, "FATHER")
        result["MOTHER_PAST_OFFSPRING_COUNT"] = map_offspring_count(mother_ids, "MOTHER")

        if len(result) != before_count:
            raise RuntimeError("Lineage offspring mapping changed row count: {} -> {}".format(before_count, len(result)))
        if not original_order.equals(result.index.to_series().reset_index(drop=True)):
            raise RuntimeError("Lineage offspring mapping changed main_data row order")

        print("\n=== Lineage offspring auxiliary data statistics ===")
        
        for column in LINEAGE_OFFSPRING_OUTPUT_COLUMNS:
            missing_count = int(result[column].eq("MISSING").sum())
            missing_pct = (missing_count / before_count * 100 if before_count else 0.0)
            print("{} MISSING: {:,} ({:.4f}%)".format(column, missing_count, missing_pct))

        return result
    
    # ============================================================
    # 한우 혈통 없는 경우 농가의 평균 엄마 아빠 자식수로 보간
    # ============================================================
    def impute_lineage_offspring_by_farm_year(self, main_data: pd.DataFrame) -> pd.DataFrame:
        required_main_columns = {"FARM_UNIQUE_NO", "ABATT_DATE", *LINEAGE_OFFSPRING_OUTPUT_COLUMNS}
        missing_main_columns = sorted(required_main_columns - set(main_data.columns))
        
        if missing_main_columns:
            raise ValueError("main_data is missing offspring imputation columns: {}".format(missing_main_columns))

        required_lookup_columns = {"FARM_UNIQUE_NO", "TARGET_YEAR", "FARM_YEAR_FATHER_OFFSPRING_MEAN", "FARM_YEAR_FATHER_SAMPLE_COUNT", "FARM_YEAR_MOTHER_OFFSPRING_MEAN", "FARM_YEAR_MOTHER_SAMPLE_COUNT"}
        lookup = pd.read_csv(
            self.lineage_farm_offspring_data_path,
            usecols=list(required_lookup_columns),
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
        missing_lookup_columns = sorted(required_lookup_columns - set(lookup.columns))
        if missing_lookup_columns:
            raise ValueError("Missing columns in {}: {}".format(self.lineage_farm_offspring_data_path, missing_lookup_columns))
        
        lookup["FARM_UNIQUE_NO"] = (lookup["FARM_UNIQUE_NO"].astype("string").str.strip())
        if lookup.duplicated(["FARM_UNIQUE_NO", "TARGET_YEAR"]).any():
            raise ValueError("Farm-year offspring imputation lookup contains duplicate keys")

        before_count = len(main_data)
        original_order = main_data.index.to_series().reset_index(drop=True)
        result = main_data.copy()
        if "ABATT_DATE_DT" in result.columns:
            target_year = result["ABATT_DATE_DT"].dt.year.astype("Int64")
        else:
            target_year = self._parse_date_series(result["ABATT_DATE"]).dt.year.astype("Int64")
            
        farm_ids = result["FARM_UNIQUE_NO"].astype("string").str.strip()
        invalid_farm = farm_ids.isna() | farm_ids.isin(MISSING_TOKENS)
        farm_ids = farm_ids.mask(invalid_farm)

        lookup_index = lookup.set_index(["FARM_UNIQUE_NO", "TARGET_YEAR"])
        valid_key = farm_ids.notna() & target_year.notna()
        mapped = pd.DataFrame(index=result.index)
        lookup_value_columns = ["FARM_YEAR_FATHER_OFFSPRING_MEAN", "FARM_YEAR_FATHER_SAMPLE_COUNT", "FARM_YEAR_MOTHER_OFFSPRING_MEAN", "FARM_YEAR_MOTHER_SAMPLE_COUNT"]
        
        for column in lookup_value_columns:
            mapped[column] = pd.NA
        
        # 같은 농가로 묶인 그룹은 평균 함께 처리
        if valid_key.any():
            keys = pd.MultiIndex.from_arrays([farm_ids.loc[valid_key], target_year.loc[valid_key].astype("int64")], names=["FARM_UNIQUE_NO", "TARGET_YEAR"])
            matched = lookup_index.reindex(keys)[lookup_value_columns]
            mapped.loc[valid_key, lookup_value_columns] = matched.to_numpy()

        imputation_specs = [
            ("FATHER_PAST_OFFSPRING_COUNT", "FARM_YEAR_FATHER_OFFSPRING_MEAN", "FARM_YEAR_FATHER_SAMPLE_COUNT", "FATHER_OFFSPRING_IMPUTED"),
            ("MOTHER_PAST_OFFSPRING_COUNT", "FARM_YEAR_MOTHER_OFFSPRING_MEAN", "FARM_YEAR_MOTHER_SAMPLE_COUNT", "MOTHER_OFFSPRING_IMPUTED"),
        ]
        
        print("\n=== Lineage offspring imputation statistics ===")

        for value_column, mean_column, sample_column, flag_column in imputation_specs:
            original_values = result[value_column].copy()
            original_missing = original_values.astype("string").eq("MISSING")
            mean_values = pd.to_numeric(mapped[mean_column], errors="coerce")
            sample_values = pd.to_numeric(mapped[sample_column], errors="coerce")
            impute_mask = (original_missing & mean_values.notna() & sample_values.ge(MIN_FARM_YEAR_IMPUTATION_SAMPLES))

            result[flag_column] = "NO"
            result.loc[impute_mask, value_column] = mean_values.loc[impute_mask].round(4)
            result.loc[impute_mask, flag_column] = "YES"

            changed_nonmissing = (~original_missing & result[value_column].astype("string").ne(original_values.astype("string")))
            if changed_nonmissing.any():
                raise RuntimeError("{} changed an original nonmissing value".format(value_column))
            
            final_numeric = pd.to_numeric(result[value_column], errors="coerce")
            if final_numeric.dropna().lt(0).any():
                raise RuntimeError("{} contains a negative value after imputation".format(value_column))

            original_missing_count = int(original_missing.sum())
            imputed_count = int(impute_mask.sum())
            final_missing_count = int(result[value_column].astype("string").eq("MISSING").sum())
            reduction_pct = (imputed_count / original_missing_count * 100 if original_missing_count else 0.0)
            
            print("{} imputed: {:,} ({:.4f}%)".format(value_column, imputed_count, reduction_pct))
            print("{} final MISSING: {:,} ({:.4f}%)".format(value_column, final_missing_count, final_missing_count / before_count * 100 if before_count else 0.0))

        if len(result) != before_count:
            raise RuntimeError("Offspring imputation changed row count: {} -> {}".format(before_count, len(result)))
        if not original_order.equals(result.index.to_series().reset_index(drop=True)):
            raise RuntimeError("Offspring imputation changed main_data row order")
        
        for flag_column in LINEAGE_OFFSPRING_IMPUTATION_COLUMNS:
            invalid_flags = set(result[flag_column]) - {"YES", "NO"}
            
            if invalid_flags:
                raise RuntimeError("{} contains invalid flags: {}".format(flag_column, sorted(invalid_flags)))

        return result

    @staticmethod
    def _normalize_numeric(column: pd.Series) -> pd.Series:
        normalized = column.astype("string").str.strip()
        normalized = normalized.mask(normalized.isna() | normalized.isin(MISSING_TOKENS))
        
        return pd.to_numeric(normalized, errors="coerce")

    # ============================================================
    # 농가 면적 보조 데이터로더
    # ============================================================
    def aux_area_data_loader(self, main_data: pd.DataFrame) -> pd.DataFrame:
        """ 
        농가별 평균 한우 사육 두수와 면적 데이터를 메인 데이터에 결합
        동일 농가 후보(HIGH/MEDIUM)는 사육 두수를 합산하고 공통 면적으로 밀도를 계산
        조회되지 않거나 계산할 수 없는 값은 MISSING으로 처리
        """
        if "FARM_UNIQUE_NO" not in main_data.columns:
            raise ValueError("main_data is missing area join column: FARM_UNIQUE_NO")
        
        before_count = len(main_data)
        original_order = main_data["FARM_UNIQUE_NO"].reset_index(drop=True).copy()

        area = pd.read_csv(self.area_data_path, dtype={"FARM_UNIQUE_NO": "string"}, encoding="utf-8-sig")
        required_area_columns = {"FARM_UNIQUE_NO", "AREA", "AVG"}
        missing_area_columns = sorted(required_area_columns - set(area.columns))
        if missing_area_columns:
            raise ValueError("Missing columns in {}: {}".format(self.area_data_path, missing_area_columns))

        area["FARM_UNIQUE_NO"] = area["FARM_UNIQUE_NO"].astype("string").str.strip()
        area["AREA"] = self._normalize_numeric(area["AREA"])
        area["AVG"] = self._normalize_numeric(area["AVG"])
        area.loc[area["AREA"].le(0), "AREA"] = pd.NA

        valid_area_counts = (area.dropna(subset=["AREA"]).groupby("FARM_UNIQUE_NO")["AREA"].nunique())
        inconsistent_area_ids = valid_area_counts[valid_area_counts.gt(1)].index
        
        if len(inconsistent_area_ids):
            raise ValueError("{} FARM_UNIQUE_NO values have multiple valid AREA values".format(len(inconsistent_area_ids)))

        farm_area = (area.groupby("FARM_UNIQUE_NO", as_index=False).agg(FARM_AVG_HANWOO_COUNT=("AVG", lambda values: values.sum(min_count=1)), FARM_AREA=("AREA", "first")))
        if farm_area["FARM_UNIQUE_NO"].duplicated().any():
            raise RuntimeError("Area preprocessing created duplicate FARM_UNIQUE_NO")

        candidates = pd.read_csv(
            self.same_farm_candidates_path,
            dtype={"CANDIDATE_GROUP_ID": "string", "CANDIDATE_STRENGTH": "string", "FARM_UNIQUE_NO": "string"},
            encoding="utf-8-sig",
        )
        required_candidate_columns = {"CANDIDATE_GROUP_ID", "CANDIDATE_STRENGTH", "FARM_UNIQUE_NO", "AREA"}
        missing_candidate_columns = sorted(required_candidate_columns - set(candidates.columns))
        if missing_candidate_columns:
            raise ValueError("Missing columns in {}: {}".format(self.same_farm_candidates_path, missing_candidate_columns))

        candidates["CANDIDATE_GROUP_ID"] = (candidates["CANDIDATE_GROUP_ID"].astype("string").str.strip())
        candidates["CANDIDATE_STRENGTH"] = (candidates["CANDIDATE_STRENGTH"].astype("string").str.strip().str.upper())
        candidates["FARM_UNIQUE_NO"] = (candidates["FARM_UNIQUE_NO"].astype("string").str.strip())
        candidates["AREA"] = self._normalize_numeric(candidates["AREA"])
        integrated_candidates = candidates.loc[candidates["CANDIDATE_STRENGTH"].isin({"HIGH", "MEDIUM"})].copy()

        group_count_by_farm = integrated_candidates.groupby("FARM_UNIQUE_NO")["CANDIDATE_GROUP_ID"].nunique()
        duplicated_group_farms = group_count_by_farm[group_count_by_farm.gt(1)]
        
        if len(duplicated_group_farms):
            raise ValueError("{} FARM_UNIQUE_NO values belong to multiple HIGH/MEDIUM groups".format(len(duplicated_group_farms)))

        candidate_area_counts = integrated_candidates.groupby("CANDIDATE_GROUP_ID")["AREA"].nunique(dropna=True)
        inconsistent_candidate_groups = candidate_area_counts[candidate_area_counts.gt(1)].index
        
        if len(inconsistent_candidate_groups):
            raise ValueError("{} HIGH/MEDIUM groups have multiple candidate AREA values".format(len(inconsistent_candidate_groups)))

        integrated_candidates = integrated_candidates.merge(farm_area, on="FARM_UNIQUE_NO", how="left", validate="many_to_one", indicator=True)
        # candidate_farms_absent_from_area = int(integrated_candidates["_merge"].eq("left_only").sum())
        integrated_candidates = integrated_candidates.drop(columns="_merge") # 검증용 코드

        candidate_source_area_mismatch = (integrated_candidates["AREA"].notna() & integrated_candidates["FARM_AREA"].notna() & integrated_candidates["AREA"].ne(integrated_candidates["FARM_AREA"]))
        if candidate_source_area_mismatch.any():
            raise ValueError("{} HIGH/MEDIUM candidate rows disagree with source AREA".format(int(candidate_source_area_mismatch.sum())))

        source_area_counts = integrated_candidates.groupby("CANDIDATE_GROUP_ID")["FARM_AREA"].nunique(dropna=True)
        
        inconsistent_source_groups = source_area_counts[source_area_counts.gt(1)].index
        if len(inconsistent_source_groups):
            raise ValueError("{} HIGH/MEDIUM groups have multiple source AREA values".format(len(inconsistent_source_groups)))

        group_area = (integrated_candidates.groupby("CANDIDATE_GROUP_ID")["FARM_AREA"].first())
        group_avg = integrated_candidates.groupby("CANDIDATE_GROUP_ID")["FARM_AVG_HANWOO_COUNT"].sum(min_count=1)
        group_density = group_avg / group_area
        group_density = group_density.mask(group_area.isna() | group_area.le(0))

        area_lookup = farm_area.copy()
        area_lookup["HANWOO_DENSITY"] = (area_lookup["FARM_AVG_HANWOO_COUNT"] / area_lookup["FARM_AREA"])
        area_lookup.loc[area_lookup["FARM_AREA"].isna() | area_lookup["FARM_AREA"].le(0), ["FARM_AREA", "HANWOO_DENSITY"]] = pd.NA
        area_lookup = area_lookup.set_index("FARM_UNIQUE_NO")

        candidate_group_map = integrated_candidates.set_index("FARM_UNIQUE_NO")["CANDIDATE_GROUP_ID"]
        candidate_strength_map = integrated_candidates.set_index("FARM_UNIQUE_NO")["CANDIDATE_STRENGTH"]
        
        for farm_no, group_id in candidate_group_map.items():
            area_lookup.loc[farm_no, "FARM_AVG_HANWOO_COUNT"] = group_avg.loc[group_id]
            area_lookup.loc[farm_no, "FARM_AREA"] = group_area.loc[group_id]
            area_lookup.loc[farm_no, "HANWOO_DENSITY"] = group_density.loc[group_id]

        if area_lookup.index.duplicated().any():
            raise RuntimeError("Area lookup contains duplicate FARM_UNIQUE_NO values")
        if area_lookup["HANWOO_DENSITY"].isin([float("inf"), float("-inf")]).any():
            raise RuntimeError("HANWOO_DENSITY contains an infinite value")

        result = main_data.copy()
        main_farms = result["FARM_UNIQUE_NO"].astype("string").str.strip()
        result["FARM_AVG_HANWOO_COUNT"] = main_farms.map(area_lookup["FARM_AVG_HANWOO_COUNT"])
        result["FARM_AREA"] = main_farms.map(area_lookup["FARM_AREA"])
        result["HANWOO_DENSITY"] = main_farms.map(area_lookup["HANWOO_DENSITY"])
        result[AREA_OUTPUT_COLUMNS] = result[AREA_OUTPUT_COLUMNS].where(result[AREA_OUTPUT_COLUMNS].notna(), "MISSING")

        after_count = len(result)
        if after_count != before_count:
            raise RuntimeError("Area join changed row count: {} -> {}".format(before_count, after_count))
        if not original_order.equals(result["FARM_UNIQUE_NO"].reset_index(drop=True)):
            raise RuntimeError("Area join changed the main_data row order")

        process_type = main_farms.map(candidate_strength_map).fillna("INDIVIDUAL")
        print("\n=== Area auxiliary data statistics ===")
        for column in AREA_OUTPUT_COLUMNS:
            missing_count = int(result[column].eq("MISSING").sum())
            missing_pct = missing_count / before_count * 100 if before_count else 0.0
            print("{} MISSING: {:,} ({:.4f}%)".format(column, missing_count, missing_pct))
            
        for process_name in ["HIGH", "MEDIUM", "INDIVIDUAL"]:
            process_count = int(process_type.eq(process_name).sum())
            process_pct = (process_count / before_count * 100 if before_count else 0.0)
            print("{} processed rows: {:,} ({:.4f}%)".format(process_name, process_count, process_pct))

        return result

    @staticmethod
    def _death_status(reason_count: object, record_ratio: object) -> str:
        if pd.notna(reason_count) and float(reason_count) > 0:
            return "YES"
        if (pd.notna(record_ratio) and float(record_ratio) >= DEATH_RECORD_RATIO_THRESHOLD):
            return "NO"
        
        return "MISSING"

    # ============================================================
    # 농가 사망 보조 데이터로더
    # ============================================================
    def aux_death_data_loader(self, main_data: pd.DataFrame) -> pd.DataFrame:
        if "FARM_UNIQUE_NO" not in main_data.columns:
            raise ValueError("main_data is missing death join column: FARM_UNIQUE_NO")

        before_count = len(main_data)
        original_order = main_data["FARM_UNIQUE_NO"].reset_index(drop=True).copy()
        required_death_columns = {"FARM_UNIQUE_NO", "AVG", "FARM_DEATH_REASON_RECORD_RATIO", *DEATH_REASON_COUNT_COLUMNS, *DEATH_YEAR_COLUMNS}
        death = pd.read_csv(self.death_data_path, dtype={"FARM_UNIQUE_NO": "string"}, encoding="utf-8-sig")
        missing_death_columns = sorted(required_death_columns - set(death.columns))
        if missing_death_columns:
            raise ValueError("Missing columns in {}: {}".format(self.death_data_path, missing_death_columns))

        death = death[list(required_death_columns)].copy()
        death["FARM_UNIQUE_NO"] = (death["FARM_UNIQUE_NO"].astype("string").str.strip())
        if death["FARM_UNIQUE_NO"].isna().any():
            raise ValueError("death summary FARM_UNIQUE_NO contains missing values")
        if death["FARM_UNIQUE_NO"].duplicated().any():
            raise ValueError("death summary FARM_UNIQUE_NO must be unique")

        numeric_columns = ["AVG", "FARM_DEATH_REASON_RECORD_RATIO", *DEATH_REASON_COUNT_COLUMNS, *DEATH_YEAR_COLUMNS]
        for column in numeric_columns:
            death[column] = self._normalize_numeric(death[column])

        record_ratio = death["FARM_DEATH_REASON_RECORD_RATIO"]
        invalid_ratio = record_ratio.notna() & ~record_ratio.between(0, 1)
        if invalid_ratio.any():
            raise ValueError("{} death summary record ratios are outside 0..1".format(int(invalid_ratio.sum())))

        death["_TOTAL_DEATH_COUNT"] = (death[DEATH_YEAR_COLUMNS].fillna(0).sum(axis=1))
        if death["_TOTAL_DEATH_COUNT"].le(0).any():
            raise ValueError("death summary contains a farm with no death records")
        death["_KNOWN_REASON_COUNT"] = (death["_TOTAL_DEATH_COUNT"] * death["FARM_DEATH_REASON_RECORD_RATIO"])

        candidates = pd.read_csv(
            self.same_farm_candidates_path,
            usecols=["CANDIDATE_GROUP_ID", "CANDIDATE_STRENGTH", "FARM_UNIQUE_NO"],
            dtype="string",
            encoding="utf-8-sig",
        )
        required_candidate_columns = {"CANDIDATE_GROUP_ID", "CANDIDATE_STRENGTH", "FARM_UNIQUE_NO"}
        missing_candidate_columns = sorted(required_candidate_columns - set(candidates.columns))
        if missing_candidate_columns:
            raise ValueError("Missing columns in {}: {}".format(self.same_farm_candidates_path, missing_candidate_columns))

        for column in required_candidate_columns:
            candidates[column] = candidates[column].astype("string").str.strip()
            
        candidates["CANDIDATE_STRENGTH"] = (candidates["CANDIDATE_STRENGTH"].str.upper())
        integrated_candidates = candidates.loc[candidates["CANDIDATE_STRENGTH"].isin({"HIGH", "MEDIUM"})].copy()

        group_count_by_farm = integrated_candidates.groupby("FARM_UNIQUE_NO")["CANDIDATE_GROUP_ID"].nunique()
        duplicated_group_farms = group_count_by_farm[group_count_by_farm.gt(1)]
        if len(duplicated_group_farms):
            raise ValueError("{} FARM_UNIQUE_NO values belong to multiple HIGH/MEDIUM groups".format(len(duplicated_group_farms)))

        candidate_death = integrated_candidates.merge(death, on="FARM_UNIQUE_NO", how="left", validate="one_to_one", indicator=True)
        # absent_candidate_count = int(candidate_death["_merge"].eq("left_only").sum())
        # print("HIGH/MEDIUM candidate farms absent from death summary: {:,}".format(absent_candidate_count))

        # HIGH/MEDIUM 동일 농가 그룹은 폐사 정보가 있는 구성원으로 그룹 통계를 계산하고, 폐사 정보가 없는 구성원도 같은 그룹 통계로 결측값을 보완함.
        group_members = candidate_death.loc[candidate_death["_merge"].eq("both")].drop(columns="_merge")
        candidate_group_lookup = integrated_candidates.drop_duplicates(subset=["FARM_UNIQUE_NO"])
        group_ids_by_farm = candidate_group_lookup.set_index("FARM_UNIQUE_NO")["CANDIDATE_GROUP_ID"]
        group_strength_by_farm = candidate_group_lookup.set_index("FARM_UNIQUE_NO")["CANDIDATE_STRENGTH"]

        group_aggregations = {
            "FARM_DEATH_AVG_COUNT": ("AVG", lambda values: values.sum(min_count=1)),
            "_GROUP_TOTAL_DEATH_COUNT": ("_TOTAL_DEATH_COUNT", "sum"),
            "_GROUP_KNOWN_REASON_COUNT": ("_KNOWN_REASON_COUNT", "sum"),
        }
        for count_column in DEATH_REASON_COUNT_COLUMNS:
            group_aggregations[count_column] = (count_column, lambda values: values.sum(min_count=1))
            
        grouped_death = group_members.groupby("CANDIDATE_GROUP_ID", as_index=True).agg(**group_aggregations)
        grouped_death["_GROUP_RECORD_RATIO"] = (grouped_death["_GROUP_KNOWN_REASON_COUNT"] / grouped_death["_GROUP_TOTAL_DEATH_COUNT"])
        if (grouped_death["_GROUP_RECORD_RATIO"].notna() & ~grouped_death["_GROUP_RECORD_RATIO"].between(0, 1)).any():
            raise RuntimeError("A grouped death record ratio is outside 0..1")

        death_lookup = death.set_index("FARM_UNIQUE_NO").copy()
        death_lookup["FARM_DEATH_AVG_COUNT"] = death_lookup["AVG"]
        for count_column, status_column in DEATH_REASON_COUNT_COLUMNS.items():
            death_lookup[status_column] = [
                self._death_status(count, ratio)
                for count, ratio in zip(death_lookup[count_column], death_lookup["FARM_DEATH_REASON_RECORD_RATIO"])
            ]

        for farm_no, group_id in group_ids_by_farm.items():
            if group_id not in grouped_death.index:
                continue
            group_row = grouped_death.loc[group_id]
            death_lookup.loc[farm_no, "FARM_DEATH_AVG_COUNT"] = group_row["FARM_DEATH_AVG_COUNT"]
            for count_column, status_column in DEATH_REASON_COUNT_COLUMNS.items():
                death_lookup.loc[farm_no, status_column] = self._death_status(group_row[count_column], group_row["_GROUP_RECORD_RATIO"])

        if death_lookup.index.duplicated().any():
            raise RuntimeError("Death lookup contains duplicate FARM_UNIQUE_NO values")
        valid_statuses = {"YES", "NO", "MISSING"}
        
        for status_column in DEATH_STATUS_COLUMNS:
            invalid_statuses = set(death_lookup[status_column].dropna()) - valid_statuses
            if invalid_statuses:
                raise RuntimeError("{} contains invalid values: {}".format(status_column, sorted(invalid_statuses)))

        result = main_data.copy()
        main_farms = result["FARM_UNIQUE_NO"].astype("string").str.strip()
        for column in DEATH_OUTPUT_COLUMNS:
            result[column] = main_farms.map(death_lookup[column])
        
        result[DEATH_OUTPUT_COLUMNS] = result[DEATH_OUTPUT_COLUMNS].where(result[DEATH_OUTPUT_COLUMNS].notna(), "MISSING")

        after_count = len(result)
        if after_count != before_count:
            raise RuntimeError("Death join changed row count: {} -> {}".format(before_count, after_count))
        
        if not original_order.equals(result["FARM_UNIQUE_NO"].reset_index(drop=True)):
            raise RuntimeError("Death join changed the main_data row order")
        
        for status_column in DEATH_STATUS_COLUMNS:
            invalid_statuses = set(result[status_column]) - valid_statuses
            if invalid_statuses:
                raise RuntimeError("{} contains invalid joined values: {}".format(status_column, sorted(invalid_statuses)))

        process_type = main_farms.map(group_strength_by_farm).fillna("INDIVIDUAL")
        death_data_available = main_farms.isin(death_lookup.index)
        print("\n=== Death auxiliary data statistics ===")
        avg_missing_count = int(result["FARM_DEATH_AVG_COUNT"].eq("MISSING").sum())
        avg_missing_pct = (avg_missing_count / before_count * 100 if before_count else 0.0)
        print("FARM_DEATH_AVG_COUNT MISSING: {:,} ({:.4f}%)".format(avg_missing_count, avg_missing_pct))
        
        for status_column in DEATH_STATUS_COLUMNS:
            for status in ["YES", "NO", "MISSING"]:
                status_count = int(result[status_column].eq(status).sum())
                status_pct = (status_count / before_count * 100 if before_count else 0.0)
                print("{} {}: {:,} ({:.4f}%)".format(status_column, status, status_count, status_pct))
                
        for process_name in ["HIGH", "MEDIUM", "INDIVIDUAL"]:
            process_count = int(process_type.eq(process_name).sum())
            process_pct = (process_count / before_count * 100 if before_count else 0.0)
            print("{} processed rows: {:,} ({:.4f}%)".format(process_name, process_count, process_pct))
        
        # absent_count = int((~death_data_available).sum())
        # absent_pct = absent_count / before_count * 100 if before_count else 0.0
        # print("Rows whose farm is absent from death summary: {:,} ({:.4f}%)".format(absent_count, absent_pct))

        # print("\n=== Death processing examples ===")
        # example_columns = ["FARM_UNIQUE_NO"] + DEATH_OUTPUT_COLUMNS
        # diagnostic_masks = {
        #     "YES STATUS": result[DEATH_STATUS_COLUMNS].eq("YES").any(axis=1),
        #     "NO STATUS": result[DEATH_STATUS_COLUMNS].eq("NO").any(axis=1),
        #     "MISSING STATUS": result[DEATH_STATUS_COLUMNS].eq("MISSING").any(axis=1),
        #     "HIGH/MEDIUM": process_type.isin({"HIGH", "MEDIUM"}),
        #     "NO DEATH SUMMARY": ~death_data_available,
        # }
        # for label, mask in diagnostic_masks.items():
        #     example_rows = result.loc[mask, example_columns]
        #     if example_rows.empty:
        #         continue
        #     print("\n[{}]".format(label))
        #     print(example_rows.head(1).to_string(index=False))

        return result

    @staticmethod
    def _last_grade_ratio_column(level: str) -> str:
        normalized = (level.replace("++", "PP").replace("+", "P").replace("등외", "OUT"))
        
        return "PAST_LAST_GRADE_{}_RATIO".format(normalized)

    @staticmethod
    def _prepare_slaughter_key_data(data: pd.DataFrame) -> pd.DataFrame:
        prepared = data.copy()
        if "ABATT_DATE_DT" in prepared.columns:
            prepared["__ABATT_DATE_DT"] = prepared["ABATT_DATE_DT"]
        else:
            prepared["__ABATT_DATE_DT"] = DataBuilder._parse_date_series(prepared["ABATT_DATE"])
            
        prepared["__TARGET_YEAR"] = prepared["__ABATT_DATE_DT"].dt.year.astype("Int64")
        prepared["__ABATT_SEASON"] = DataBuilder._date_to_korean_season(prepared["__ABATT_DATE_DT"])
        prepared["__AGE_NUM"] = pd.to_numeric(prepared["AGE"], errors="coerce").round().astype("Int64")
        prepared["sido"] = prepared["sido"].astype("string").str.strip()
        
        return prepared

    def _build_slaughter_stats_for_year(self, history: pd.DataFrame, target_year: int, history_years: int, age_tolerance_months: int, use_season: bool) -> pd.DataFrame:
        history_window = history.loc[history["__TARGET_YEAR"].between(target_year - history_years, target_year - 1)].copy()
        
        if history_window.empty:
            return pd.DataFrame()

        base_key_columns = ["sido"]
        target_key_columns = ["__TARGET_YEAR", "sido"]
        
        if use_season:
            base_key_columns.append("__ABATT_SEASON")
            target_key_columns.append("__ABATT_SEASON")
        base_key_columns.append("__AGE_NUM")
        target_key_columns.append("__AGE_NUM")

        numeric_columns = SLAUGHTER_CONTINUOUS_COLUMNS
        numeric_sum_columns = ["__{}_SUM".format(column) for column in numeric_columns]
        numeric_sumsq_columns = ["__{}_SUMSQ".format(column) for column in numeric_columns]
        numeric_count_columns = ["__{}_COUNT".format(column) for column in numeric_columns]

        exact = history_window[base_key_columns].drop_duplicates().copy()
        exact = exact.set_index(base_key_columns)

        sample_count = history_window.groupby(base_key_columns, observed=True).size()
        exact["__SAMPLE_COUNT"] = sample_count

        for column, sum_column, sumsq_column, count_column in zip(numeric_columns, numeric_sum_columns, numeric_sumsq_columns, numeric_count_columns):
            numeric = pd.to_numeric(history_window[column], errors="coerce")
            valid = numeric.notna()
            numeric_group = history_window.loc[valid, base_key_columns].copy()
            numeric_group["__VALUE"] = numeric.loc[valid].astype(float)
            exact[sum_column] = numeric_group.groupby(base_key_columns, observed=True)["__VALUE"].sum()
            exact[sumsq_column] = numeric_group.assign(__VALUE_SQ=numeric_group["__VALUE"] ** 2).groupby(base_key_columns, observed=True)["__VALUE_SQ"].sum() ###
            exact[count_column] = numeric_group.groupby(base_key_columns, observed=True)["__VALUE"].count()

        for level in SLAUGHTER_WGRADE_LEVELS:
            column_name = "__WGRADE_{}".format(level)
            exact[column_name] = history_window["WGRADE"].eq(level).groupby([history_window[column] for column in base_key_columns], observed=True).sum()

        for level in SLAUGHTER_LAST_GRADE_LEVELS:
            column_name = "__LAST_GRADE_{}".format(level.replace("++", "PP").replace("+", "P").replace("등외", "OUT"))
            exact[column_name] = history_window["LAST_GRADE"].eq(level).groupby([history_window[column] for column in base_key_columns], observed=True).sum()

        for source_column, prefix in SLAUGHTER_CLASS_FEATURES:
            values = history_window[source_column].astype("string").str.strip()
            for class_value in range(5):
                exact["__{}_{}".format(prefix, class_value)] = values.eq(str(class_value)).groupby([history_window[column] for column in base_key_columns], observed=True).sum()

        growth_values = history_window["GROWTH_STATUS_CLASS"].astype("string").str.strip()
        exact["__GROWTH_NORMAL"] = growth_values.eq("0").groupby([history_window[column] for column in base_key_columns], observed=True).sum()
        exact["__GROWTH_ABNORMAL"] = growth_values.eq("1").groupby([history_window[column] for column in base_key_columns], observed=True).sum()

        exact = exact.fillna(0).reset_index()

        year_counts = history_window[base_key_columns + ["__TARGET_YEAR"]].drop_duplicates()
        expanded_years = []
        expanded_metrics = []
        for delta in range(-age_tolerance_months, age_tolerance_months + 1):
            metric_part = exact.copy()
            metric_part["__AGE_NUM"] = metric_part["__AGE_NUM"] + delta
            metric_part["__TARGET_YEAR"] = target_year
            expanded_metrics.append(metric_part)

            year_part = year_counts.copy()
            year_part["__AGE_NUM"] = year_part["__AGE_NUM"] + delta
            year_part["__LOOKUP_TARGET_YEAR"] = target_year
            expanded_years.append(year_part)

        expanded = pd.concat(expanded_metrics, ignore_index=True)
        grouped = expanded.groupby(target_key_columns, observed=True).sum(numeric_only=True).reset_index()

        expanded_year_data = pd.concat(expanded_years, ignore_index=True)
        history_year_count = (expanded_year_data.groupby(["__LOOKUP_TARGET_YEAR", *[column for column in target_key_columns if column != "__TARGET_YEAR"]], observed=True)["__TARGET_YEAR"].nunique().rename("PAST_SLAUGHTER_HISTORY_YEARS").reset_index().rename(columns={"__LOOKUP_TARGET_YEAR": "__TARGET_YEAR"}))
        grouped = grouped.merge(history_year_count, on=target_key_columns, how="left")

        output = grouped[target_key_columns].copy()
        output["PAST_SLAUGHTER_SAMPLE_COUNT"] = grouped["__SAMPLE_COUNT"].astype("int64")
        output["PAST_SLAUGHTER_HISTORY_YEARS"] = grouped["PAST_SLAUGHTER_HISTORY_YEARS"].fillna(0).astype("int64")

        for column, sum_column, sumsq_column, count_column in zip(numeric_columns, numeric_sum_columns, numeric_sumsq_columns, numeric_count_columns):
            count = grouped[count_column].astype(float)
            total = grouped[sum_column].astype(float)
            sumsq = grouped[sumsq_column].astype(float)
            mean = total / count.where(count.ne(0))
            variance = (sumsq - (total ** 2 / count.where(count.ne(0)))) / (count - 1).where(count.gt(1))
            output["PAST_{}_MEAN".format(column)] = mean.fillna("MISSING")
            output["PAST_{}_VARIANCE".format(column)] = variance.fillna(0.0)

        sample_count = grouped["__SAMPLE_COUNT"].replace(0, pd.NA)
        for level in SLAUGHTER_WGRADE_LEVELS:
            output["PAST_WGRADE_{}_RATIO".format(level)] = (grouped["__WGRADE_{}".format(level)] / sample_count).fillna(0.0)

        for level in SLAUGHTER_LAST_GRADE_LEVELS:
            suffix = (level.replace("++", "PP").replace("+", "P").replace("등외", "OUT"))
            output["PAST_LAST_GRADE_{}_RATIO".format(suffix)] = (grouped["__LAST_GRADE_{}".format(suffix)] / sample_count).fillna(0.0)

        for _, prefix in SLAUGHTER_CLASS_FEATURES:
            for class_value in range(5):
                output["{}_{}_RATIO".format(prefix, class_value)] = (grouped["__{}_{}".format(prefix, class_value)] / sample_count).fillna(0.0)

        output["PAST_GROWTH_NORMAL_RATIO"] = (grouped["__GROWTH_NORMAL"] / sample_count).fillna(0.0)
        output["PAST_GROWTH_ABNORMAL_RATIO"] = (grouped["__GROWTH_ABNORMAL"] / sample_count).fillna(0.0)

        return output

    # ============================================================
    # 지역별 과거 한우 도축 평균 정보 데이터로더
    # ============================================================
    def aux_slaughter_data_loader(self, main_data: pd.DataFrame, history_years: int = 5, min_sample_count: int = 5, age_tolerance_months: int = 12, use_season: bool = True) -> pd.DataFrame:
        if history_years <= 0:
            raise ValueError("history_years must be positive")
        if min_sample_count <= 0:
            raise ValueError("min_sample_count must be positive")
        if age_tolerance_months < 0:
            raise ValueError("age_tolerance_months must be non-negative")

        required_main_columns = {"sido", "ABATT_DATE", "AGE"}
        missing_main_columns = sorted(required_main_columns - set(main_data.columns))
        if missing_main_columns:
            raise ValueError("main_data is missing slaughter join columns: {}".format(missing_main_columns))

        before_count = len(main_data)
        original_index = main_data.index.to_series().reset_index(drop=True)
        original_columns = main_data.columns.tolist()

        result = self._prepare_slaughter_key_data(main_data)
        if "ABATT_DATE_DT" in result.columns:
            result["__ABATT_DATE_DT"] = result["ABATT_DATE_DT"]
            result["__TARGET_YEAR"] = result["__ABATT_DATE_DT"].dt.year.astype("Int64")
        if "ABATT_SEASON" in result.columns:
            result["__ABATT_SEASON"] = result["ABATT_SEASON"]

        slaughter_columns = ["sido", "ABATT_DATE", "AGE", "WEIGHT", "BACKFAT", "REA", "WINDEX", "WGRADE", "INSFAT", "YUKSAK", "FATSAK", "TISSUE", "GROWTH", "LAST_GRADE", "SOURCE"]
        slaughter = pd.read_csv(self.slaughter_data_path, usecols=slaughter_columns, dtype="string", encoding="utf-8-sig")
        missing_slaughter_columns = sorted(set(slaughter_columns) - set(slaughter.columns))
        if missing_slaughter_columns:
            raise ValueError("Missing columns in {}: {}".format(self.slaughter_data_path, missing_slaughter_columns))

        slaughter = add_grade_class_columns(slaughter)
        slaughter = self._prepare_slaughter_key_data(slaughter)
        invalid_history = (
            slaughter["sido"].isna()
            | slaughter["sido"].isin(MISSING_TOKENS)
            | slaughter["__TARGET_YEAR"].isna()
            | slaughter["__ABATT_SEASON"].eq("MISSING")
            | slaughter["__AGE_NUM"].isna()
        )
        slaughter = slaughter.loc[~invalid_history].copy()

        stats_frames = []
        target_years = sorted(result["__TARGET_YEAR"].dropna().astype(int).unique())
        for target_year in target_years:
            stats = self._build_slaughter_stats_for_year(slaughter, target_year, history_years, age_tolerance_months, use_season)
            if not stats.empty:
                stats_frames.append(stats)

        key_columns = ["__TARGET_YEAR", "sido"]
        if use_season:
            key_columns.append("__ABATT_SEASON")
        key_columns.append("__AGE_NUM")

        if stats_frames:
            stats = pd.concat(stats_frames, ignore_index=True)
            if stats.duplicated(key_columns).any():
                raise RuntimeError("Slaughter stats contain duplicate join keys")
        else:
            stats = pd.DataFrame(columns=key_columns + SLAUGHTER_OUTPUT_COLUMNS)

        merge_columns = key_columns + [column for column in SLAUGHTER_OUTPUT_COLUMNS if column in stats.columns]
        result = result.merge(stats[merge_columns], on=key_columns, how="left", sort=False)

        result["PAST_SLAUGHTER_SAMPLE_COUNT"] = (
            pd.to_numeric(result["PAST_SLAUGHTER_SAMPLE_COUNT"], errors="coerce").fillna(0).astype("int64"))
        for column in ["PAST_SLAUGHTER_HISTORY_YEARS"] + SLAUGHTER_STAT_COLUMNS:
            if column in result.columns:
                result[column] = result[column].astype(object)

        available_mask = result["PAST_SLAUGHTER_SAMPLE_COUNT"].ge(min_sample_count)
        result["PAST_SLAUGHTER_AVAILABLE"] = "NO"
        result.loc[available_mask, "PAST_SLAUGHTER_AVAILABLE"] = "YES"
        result.loc[~available_mask, "PAST_SLAUGHTER_HISTORY_YEARS"] = "MISSING"
        for column in SLAUGHTER_STAT_COLUMNS:
            result.loc[~available_mask, column] = "MISSING"
            result.loc[available_mask, column] = result.loc[available_mask, column].fillna(0.0)

        for column in SLAUGHTER_OUTPUT_COLUMNS:
            if column not in result.columns:
                result[column] = "MISSING"

        result = result.drop(columns=["__ABATT_DATE_DT", "__TARGET_YEAR", "__ABATT_SEASON", "__AGE_NUM", ], errors="ignore")

        if len(result) != before_count:
            raise RuntimeError("Slaughter join changed row count: {} -> {}".format(before_count, len(result)))
        if not result.index.to_series().reset_index(drop=True).equals(original_index):
            raise RuntimeError("Slaughter join changed main row order")
        if result[original_columns].equals(main_data[original_columns]) is False:
            raise RuntimeError("Slaughter join modified original main_data columns")
        if result.loc[result["PAST_SLAUGHTER_SAMPLE_COUNT"].lt(min_sample_count), "PAST_SLAUGHTER_AVAILABLE"].ne("NO").any():
            raise RuntimeError("Unavailable slaughter rows have invalid availability")

        ratio_columns = [column for column in SLAUGHTER_STAT_COLUMNS if column.endswith("_RATIO")]
        available_ratios = result.loc[result["PAST_SLAUGHTER_AVAILABLE"].eq("YES"), ratio_columns]
        if not available_ratios.empty:
            numeric_ratios = available_ratios.apply(pd.to_numeric, errors="coerce")
            if numeric_ratios.lt(0).any().any() or numeric_ratios.gt(1).any().any():
                raise RuntimeError("Slaughter ratio columns are outside 0~1")

        print("\n=== Slaughter auxiliary data statistics ===")
        print("history_years: {}".format(history_years))
        print("min_sample_count: {}".format(min_sample_count))
        print("age_tolerance_months: {}".format(age_tolerance_months))
        print("use_season: {}".format(use_season))
        
        for status in ["YES", "NO"]:
            status_count = int(result["PAST_SLAUGHTER_AVAILABLE"].eq(status).sum())
            status_pct = status_count / before_count * 100 if before_count else 0.0
            print("PAST_SLAUGHTER_AVAILABLE {}: {:,} ({:.4f}%)".format(status, status_count, status_pct))

        # sample_counts = result["PAST_SLAUGHTER_SAMPLE_COUNT"]
        # print("PAST_SLAUGHTER_SAMPLE_COUNT:")
        # print("  min: {:,}".format(int(sample_counts.min()) if len(sample_counts) else 0))
        # print("  mean: {:.4f}".format(sample_counts.mean() if len(sample_counts) else 0.0))
        # print("  variance: {:.4f}".format(sample_counts.var() if len(sample_counts) > 1 else 0.0))
        # print("  max: {:,}".format(int(sample_counts.max()) if len(sample_counts) else 0))

        if "ABATT_DATE_DT" in main_data.columns:
            year_values = main_data["ABATT_DATE_DT"].dt.year
        else:
            year_values = self._parse_date_series(main_data["ABATT_DATE"]).dt.year
        for target_year, mask in year_values.groupby(year_values).groups.items():
            year_available = result.loc[mask, "PAST_SLAUGHTER_AVAILABLE"].eq("YES")
            year_pct = year_available.mean() * 100 if len(year_available) else 0.0
            print("TARGET_YEAR={} available: {:,}/{:,} ({:.4f}%)".format(int(target_year), int(year_available.sum()), len(year_available), year_pct))

        for column in SLAUGHTER_OUTPUT_COLUMNS:
            values = result[column].astype("string").str.strip()
            missing_count = int(values.eq("MISSING").sum())
            missing_pct = missing_count / before_count * 100 if before_count else 0.0
            print("{} MISSING: {:,} ({:.4f}%)".format(column, missing_count, missing_pct))

        return result

    def _ordered_output_columns(self, data: pd.DataFrame) -> list:
        columns = (["DATA_ROW_ID"] + X_COLUMNS + AUXILIARY_OUTPUT_COLUMNS)
        if self.dataset_type == "train":
            columns += Y_COLUMNS + GRADE_CLASS_COLUMNS + DERIVED_COLUMNS

        missing_columns = sorted(set(columns) - set(data.columns))
        if missing_columns:
            raise ValueError("Built data is missing output columns: {}".format(missing_columns))
        
        return list(dict.fromkeys(columns))

    def _validate_test_result(self, data: pd.DataFrame) -> None:
        if self.dataset_type != "test":
            return
        if self._source_row_count is None or self._source_cattle_order is None:
            raise RuntimeError("Test source metadata was not initialized")
        if len(data) != self._source_row_count:
            raise RuntimeError("Test row count changed: {} -> {}".format(self._source_row_count, len(data)))
        
        expected_ids = ["TEST_{:09d}".format(index) for index in range(self._source_row_count)]
        
        if data["DATA_ROW_ID"].astype("string").tolist() != expected_ids:
            raise RuntimeError("Test DATA_ROW_ID order does not match source order")
        if not data["CATTLE_NO"].astype("string").reset_index(drop=True).equals(self._source_cattle_order.reset_index(drop=True)):
            raise RuntimeError("Test CATTLE_NO order does not match source order")

    def build(self) -> pd.DataFrame:
        data = self.main_data_loader()
        data = self._run_auxiliary_loader(self.aux_lineage_data_loader, data)
        data = self._run_auxiliary_loader(self.aux_lineage_resolve_offspring, data)
        data = self._run_auxiliary_loader(self.impute_lineage_offspring_by_farm_year, data)
        data = self._run_auxiliary_loader(self.aux_area_data_loader, data)
        data = self._run_auxiliary_loader(self.aux_death_data_loader, data)
        data = self._run_auxiliary_loader(self.aux_slaughter_data_loader, data, history_years=5, min_sample_count=5, age_tolerance_months=12, use_season=True)
        
        if "weather" in data.columns:
            raise RuntimeError("Built tabular data must not contain weather lists")
        self._validate_test_result(data)
        ordered_columns = self._ordered_output_columns(data)
        result = data[ordered_columns].copy()
        self._validate_data_row_ids(result, "build result")
        self.print_tabular_missing_statistics(result)

        filtered_count = (self._source_row_count - len(result) if self._source_row_count is not None else 0)
        print("\n=== Build summary ===")
        print("dataset_type: {}".format(self.dataset_type))
        print("source rows: {:,}".format(self._source_row_count if self._source_row_count is not None else 0))
        print("final rows: {:,}".format(len(result)))
        print("filtered rows: {:,}".format(filtered_count))
        print("DATA_ROW_ID unique: {:,}".format(result["DATA_ROW_ID"].nunique(dropna=False)))
        print("DATA_ROW_ID duplicates: {:,}".format(int(result["DATA_ROW_ID"].duplicated().sum())))
        
        if self.dataset_type == "test":
            print("test source row count preserved: YES")
            print("test CATTLE_NO order preserved: YES")
        return result

    def save_built_data(self, data: pd.DataFrame, save_type: str = "parquet") -> Path:
        normalized_save_type = save_type.strip().lower()
        if normalized_save_type not in {"parquet", "csv"}:
            raise ValueError("save_type must be either 'parquet' or 'csv'")

        self._validate_data_row_ids(data, "save input")
        if self.dataset_type == "test":
            self._validate_test_result(data)
        ordered = data[self._ordered_output_columns(data)].copy()
        if ordered.columns[0] != "DATA_ROW_ID":
            raise RuntimeError("DATA_ROW_ID must be the first output column")

        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.processed_data_dir / "hanwoo_{}_merged.{}".format(self.dataset_type, normalized_save_type)

        if normalized_save_type == "parquet":
            try:
                import pyarrow  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("Parquet 저장을 위해 pyarrow가 필요합니다: pip install pyarrow") from exc

            # PyArrow 는 physical data type만 사용가능함. 근데 pandas는 MISSING을 object 타입으로 처리. 에러발생해서  문자열로 변환함
            parquet_data = ordered.copy()
            object_columns = parquet_data.select_dtypes(include=["object"]).columns
            for column in object_columns:
                parquet_data[column] = (parquet_data[column].astype("string").fillna("MISSING"))

            parquet_data.to_parquet(output_path, engine="pyarrow", compression="zstd", index=False)
            reloaded = pd.read_parquet(output_path, columns=["DATA_ROW_ID", "CATTLE_NO"], engine="pyarrow")
        else:
            ordered.to_csv(output_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
            reloaded = pd.read_csv(output_path, usecols=["DATA_ROW_ID", "CATTLE_NO"], dtype="string", encoding="utf-8-sig")

        if len(reloaded) != len(ordered):
            raise RuntimeError("Saved data row count does not match build result")
        if reloaded["DATA_ROW_ID"].astype("string").tolist() != ordered["DATA_ROW_ID"].astype("string").tolist():
            raise RuntimeError("Saved DATA_ROW_ID order does not match build result")
        if not reloaded["CATTLE_NO"].astype("string").reset_index(drop=True).equals(ordered["CATTLE_NO"].astype("string").reset_index(drop=True)):
            raise RuntimeError("Saved CATTLE_NO order does not match build result")

        print("Saved merged data:", output_path)
        return output_path

    def build_and_save(self, save_type: str = "parquet") -> Path:
        data = self.build()
        return self.save_built_data(data, save_type)

    def print_tabular_missing_statistics(self, data: pd.DataFrame) -> None:
        tabular_columns = (
            X_COLUMNS
            + SEASON_COLUMNS
            + LINEAGE_OUTPUT_COLUMNS
            + LINEAGE_OFFSPRING_OUTPUT_COLUMNS
            + LINEAGE_OFFSPRING_IMPUTATION_COLUMNS
            + AREA_OUTPUT_COLUMNS
            + DEATH_OUTPUT_COLUMNS
            + SLAUGHTER_OUTPUT_COLUMNS
        )
        if self.dataset_type == "train":
            tabular_columns += (Y_COLUMNS + GRADE_CLASS_COLUMNS + DERIVED_COLUMNS)
        missing_columns = sorted(set(tabular_columns) - set(data.columns))
        if missing_columns:
            raise ValueError("Data is missing tabular columns: {}".format(missing_columns))

        total_count = len(data)
        print("\n=== Tabular data missing statistics ===")
        print("Total rows: {:,}".format(total_count))
        for column in tabular_columns:
            values = data[column].astype("string").str.strip()
            missing_mask = (values.isna() | values.isin(MISSING_TOKENS) | values.eq("MISSING"))
            missing_count = int(missing_mask.sum())
            missing_pct = missing_count / total_count * 100 if total_count else 0.0
            print("{} MISSING: {:,} ({:.4f}%)".format(column, missing_count, missing_pct))


def main():
    parser = argparse.ArgumentParser(description="Build merged Hanwoo train or test tabular data.")
    parser.add_argument("--dataset-type", choices=["train", "test"], default="train")
    parser.add_argument("--save-type", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--min-birth-date", default="2008-01-01", help="Train-only minimum birth date. Train defaults to 2020-01-01. Test does not allow this option.")
    # parser.add_argument("--min-birth-date", default=None, help="Train-only minimum birth date. Train defaults to 2020-01-01. Test does not allow this option.")
    args = parser.parse_args()

    builder = DataBuilder(dataset_type=args.dataset_type, min_birth_date=args.min_birth_date)
    output_path = builder.build_and_save(save_type=args.save_type)
    print("Build completed:", output_path)


if __name__ == "__main__":
    main()
    
  
