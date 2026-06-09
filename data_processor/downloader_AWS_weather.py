#!/usr/bin/env python3
"""
Download daily AWS weather rows from KMA APIHub hourly statistics.
The script calls awsh.php 24 times per variable and aggregates the result into hanwoo_weather-compatible daily rows:

    stn,date,ta_max,rn_day,ta_min,rhm_avg,ws_davg

Example:
    STNS=$(awk '/^[0-9]+\\par$/ {gsub(/\\par/,""); print}' weather_stn.rtf | sort -n | paste -sd, -)
    
    python3 AWS_weather_downloader.py \
    --stn "$STNS" \
    --start-date 2014-12-01 \
    --end-date 2014-12-31 \
    --output aws_weather_20141201_20141231.xlsx \
    --timeout 120 \
    --retries 5 \
    --sleep 0.2
    
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import mean
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


import pandas as pd


API_URL = "https://apihub.kma.go.kr/api/typ01/url/awsh.php"
OUTPUT_COLUMNS = [
    "stn",
    "date",
    "ta_max",
    "rn_day",
    "ta_min",
    "rhm_avg",
    "ws_davg",
]
MISSING_VALUE = -99.0

# Optional hard-coded API key. Fill this if you do not want to pass --auth-key.
HARD_CODED_AUTH_KEY = ""

@dataclass(frozen=True)
class Config:
    auth_key: str
    timeout: int
    retries: int
    sleep_sec: float


def parse_date(value: str) -> date:
    normalized = value.replace(".", "-").replace("/", "-")
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(normalized, fmt).date()
        except ValueError:
            continue
    raise argparse.ArgumentTypeError("date must be like 2023-03-11, 2023.3.11, or 20230311")


def parse_stns(value: str) -> list[int]:
    stns = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        stns.append(int(token))
    if not stns:
        raise argparse.ArgumentTypeError("--stn must include at least one station number")
    return stns


def daterange(start: date, end: date):
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def hourly_timestamps(day: date) -> list[str]:
    # awsh.php hourly values summarize the previous 60 minutes.
    # 01:00..next-day 00:00 covers one calendar day as 24 one-hour windows.
    start = datetime.combine(day, datetime.min.time()) + timedelta(hours=1)
    return [(start + timedelta(hours=i)).strftime("%Y%m%d%H%M") for i in range(24)]


def request_text(var: str, tm: str, stn: int | None, config: Config) -> str:
    params = {
        "var": var,
        "tm": tm,
        "help": "0",
        "authKey": config.auth_key,
    }
    if stn is not None:
        params["stn"] = str(stn)
    url = f"{API_URL}?{urlencode(params)}"
    last_error: Exception | None = None

    for attempt in range(config.retries + 1):
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 hanwoo-weather-downloader/1.0",
                    "Accept": "text/plain,*/*",
                },
            )
            with urlopen(request, timeout=config.timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code} for {url}\n{body[:1000]}")
        except (URLError, TimeoutError) as exc:
            last_error = exc

        if attempt < config.retries:
            time.sleep(max(config.sleep_sec, 0.5) * (attempt + 1))

    stn_msg = "all" if stn is None else str(stn)
    raise RuntimeError(f"API request failed for var={var}, tm={tm}, stn={stn_msg}: {last_error}")


def split_data_line(line: str) -> list[str]:
    line = line.strip().rstrip("\\")
    if not line or line.startswith("#"):
        return []
    if "," in line:
        return [part.strip() for part in next(csv.reader([line]))]
    return line.split()


def to_float(value: str) -> float | None:
    value = value.strip()
    if value in {"", "-99", "-99.0"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_response_by_stn(text: str, tm: str, target_stns: set[int]) -> dict[int, list[float | None]]:
    rows: dict[int, list[float | None]] = {}
    for raw_line in text.splitlines():
        parts = split_data_line(raw_line)
        if len(parts) < 3:
            continue
        if parts[0] != tm:
            continue
        try:
            row_stn = int(parts[1])
        except ValueError:
            continue
        if row_stn in target_stns:
            rows[row_stn] = numeric_values_after_tm_stn(parts)
    return rows


def numeric_values_after_tm_stn(parts: list[str]) -> list[float | None]:
    values = []
    for token in parts[2:]:
        values.append(to_float(token))
    return values


def fetch_var_24h_all_stns(
    var: str,
    day: date,
    stns: list[int],
    config: Config,
) -> dict[int, dict[str, list[float | None]]]:
    target_stns = set(stns)
    result: dict[int, dict[str, list[float | None]]] = {
        stn: {} for stn in stns
    }

    for tm in hourly_timestamps(day):
        text = request_text(var, tm, None, config)
        rows_by_stn = parse_response_by_stn(text, tm, target_stns)
        for stn in stns:
            result[stn][tm] = rows_by_stn.get(stn, [])

        if config.sleep_sec > 0:
            time.sleep(config.sleep_sec)

    return result


def aggregate_temperature(values_by_tm: dict[str, list[float | None]]) -> tuple[float | None, float | None]:
    hourly_ta_max = []
    hourly_ta_min = []

    for values in values_by_tm.values():
        # TA response:
        # TM STN TA HMI TA_AVG QCM TA_MAX MI TA_MIN MI
        # numeric values after TM/STN:
        # [TA, HMI, TA_AVG, QCM, TA_MAX, MI, TA_MIN, MI]
        if len(values) >= 7:
            if values[4] is not None:
                hourly_ta_max.append(values[4])
            if values[6] is not None:
                hourly_ta_min.append(values[6])

    ta_max = max(hourly_ta_max) if hourly_ta_max else None
    ta_min = min(hourly_ta_min) if hourly_ta_min else None
    return ta_max, ta_min


def aggregate_rain(values_by_tm: dict[str, list[float | None]]) -> float | None:
    rn_day_values = []

    for values in values_by_tm.values():
        # RN response is expected to include RN_DAY as the daily rainfall up to
        # the observation time. In the common AWS hourly layout:
        # [RE_SUM, RE_QCM, RN_DAY, RN_DAY_MI, RN_HR1, RN_HR1_MI]
        if len(values) >= 3 and values[2] is not None:
            rn_day_values.append(values[2])

    if not rn_day_values:
        return None
    return max(rn_day_values)


def aggregate_humidity(values_by_tm: dict[str, list[float | None]]) -> float | None:
    hm_values = []

    for values in values_by_tm.values():
        # HM response:
        # TM STN HM HMI HM_AVG QCM HM_MAX MI HM_MIN MI
        # numeric values after TM/STN:
        # [HM, HMI, HM_AVG, QCM, HM_MAX, MI, HM_MIN, MI]
        if len(values) >= 1 and values[0] is not None:
            hm_values.append(values[0])

    return mean(hm_values) if hm_values else None


def aggregate_wind(values_by_tm: dict[str, list[float | None]]) -> float | None:
    ws_avg_values = []

    for values in values_by_tm.values():
        # WD response contains both 10-minute wind speed WS and 60-minute
        # minute-level average wind speed WS1_AVG. Prefer WS1_AVG when present.
        # Common numeric layout after TM/STN:
        # [WD, WS, WS_HMI, WD_MAX, WS_MAX, WS_MAX_MI, WS_QCM,
        #  WD1_MAX, WS1_MAX, WS1_QCM, WS1_AVG, ...]
        if len(values) >= 11 and values[10] is not None and values[10] >= 0:
            ws_avg_values.append(values[10])
        elif len(values) >= 2 and values[1] is not None and values[1] >= 0:
            ws_avg_values.append(values[1])

    if not ws_avg_values:
        return None
    return mean(ws_avg_values)


def rounded_or_missing(value: float | None) -> float:
    if value is None:
        return MISSING_VALUE
    return round(value, 1)


def format_output_date(day: date) -> str:
    return f"{day.year}.{day.month}.{day.day}"
    

def build_daily_row_from_values(
    day: date,
    stn: int,
    ta_values: dict[str, list[float | None]],
    rn_values: dict[str, list[float | None]],
    hm_values: dict[str, list[float | None]],
    wd_values: dict[str, list[float | None]],
) -> dict[str, object]:

    ta_max, ta_min = aggregate_temperature(ta_values)
    rn_day = aggregate_rain(rn_values)
    rhm_avg = aggregate_humidity(hm_values)
    ws_davg = aggregate_wind(wd_values)

    return {
        "stn": stn,
        "date": format_output_date(day),
        "ta_max": rounded_or_missing(ta_max),
        "rn_day": rounded_or_missing(rn_day),
        "ta_min": rounded_or_missing(ta_min),
        "rhm_avg": rounded_or_missing(rhm_avg),
        "ws_davg": rounded_or_missing(ws_davg),
    }


def build_daily_rows(day: date, stns: list[int], config: Config) -> list[dict[str, object]]:
    print(f"downloading TA 24 hourly records for all stations, date={day.isoformat()}")
    ta_values = fetch_var_24h_all_stns("TA", day, stns, config)
    print(f"downloading RN 24 hourly records for all stations, date={day.isoformat()}")
    rn_values = fetch_var_24h_all_stns("RN", day, stns, config)
    print(f"downloading HM 24 hourly records for all stations, date={day.isoformat()}")
    hm_values = fetch_var_24h_all_stns("HM", day, stns, config)
    print(f"downloading WD 24 hourly records for all stations, date={day.isoformat()}")
    wd_values = fetch_var_24h_all_stns("WD", day, stns, config)

    return [
        build_daily_row_from_values(
            day,
            stn,
            ta_values[stn],
            rn_values[stn],
            hm_values[stn],
            wd_values[stn],
        )
        for stn in stns
    ]


def default_output_path(stns: list[int], start: date, end: date) -> str:
    stn_label = str(stns[0]) if len(stns) == 1 else f"{len(stns)}stns"
    return f"kma_aws_daily_weather_{stn_label}_{start:%Y%m%d}_{end:%Y%m%d}.xlsx"


def partial_output_path(
    output: str | None,
    stns: list[int],
    start: date,
    requested_end: date,
    saved_end: date | None,
    failed_day: date,
) -> Path:
    failed_label = f"{failed_day:%Y%m%d}"
    if saved_end is None:
        saved_label = "no_success"
    else:
        saved_label = f"{saved_end:%Y%m%d}"

    if output is None:
        stn_label = str(stns[0]) if len(stns) == 1 else f"{len(stns)}stns"
        return Path(
            f"kma_aws_daily_weather_{stn_label}_{start:%Y%m%d}_{saved_label}"
            f"_failed_{failed_label}.xlsx"
        )

    output_path = Path(output)
    requested_label = f"{requested_end:%Y%m%d}"
    stem = output_path.stem
    if saved_end is not None and requested_label in stem:
        stem = stem.replace(requested_label, saved_label)
    else:
        stem = f"{stem}_through_{saved_label}"
    return output_path.with_name(f"{stem}_failed_{failed_label}{output_path.suffix or '.xlsx'}")


def save_rows(rows: list[dict[str, object]], output_path: Path) -> None:
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)
    print(f"\nsaved: {output_path}")
    print(f"rows: {len(df)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download daily AWS weather rows from 24 hourly KMA API calls.")
    parser.add_argument("--date", default=None, type=parse_date, help="Single date. Example: 2023-03-11 or 2023.3.11")
    parser.add_argument("--start-date", default=None, type=parse_date, help="Start date. Example: 2000-01-02")
    parser.add_argument("--end-date", default=None, type=parse_date, help="End date. Example: 2000-01-10")
    parser.add_argument("--stn", required=True, type=parse_stns, help="AWS station number(s), e.g. 741 or 920,741,661")
    parser.add_argument("--output", default=None, help="Optional .xlsx output path")
    parser.add_argument("--auth-key", default=None, help="KMA APIHub auth key")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.1, help="Seconds between API calls")
    args = parser.parse_args()

    auth_key = args.auth_key or HARD_CODED_AUTH_KEY or os.environ.get("KMA_AUTH_KEY")
    if not auth_key:
        print("Set HARD_CODED_AUTH_KEY in this script, pass --auth-key, or set KMA_AUTH_KEY.", file=sys.stderr)
        return 2

    if args.date is not None:
        start = args.date
        end = args.date
    else:
        if args.start_date is None or args.end_date is None:
            print("Pass either --date or both --start-date and --end-date.", file=sys.stderr)
            return 2
        start = args.start_date
        end = args.end_date

    if end < start:
        print("--end-date must be greater than or equal to --start-date.", file=sys.stderr)
        return 2

    config = Config(
        auth_key=auth_key,
        timeout=args.timeout,
        retries=args.retries,
        sleep_sec=args.sleep,
    )

    rows = []
    total = (end - start).days + 1
    done = 0
    last_success_day: date | None = None
    failed_day: date | None = None

    try:
        for day in daterange(start, end):
            done += 1
            print(f"\n[{done}/{total}] date={day.isoformat()}, stns={','.join(map(str, args.stn))}")
            day_rows = build_daily_rows(day, args.stn, config)
            rows.extend(day_rows)
            last_success_day = day
    except KeyboardInterrupt:
        failed_day = start + timedelta(days=max(done - 1, 0))
        print(f"\nInterrupted while downloading date={failed_day.isoformat()}.", file=sys.stderr)
    except Exception as exc:
        failed_day = start + timedelta(days=max(done - 1, 0))
        print(f"\nDownload failed while downloading date={failed_day.isoformat()}.", file=sys.stderr)
        print(f"Error: {exc}", file=sys.stderr)

    if failed_day is not None:
        if rows:
            output_path = partial_output_path(
                args.output,
                args.stn,
                start,
                end,
                last_success_day,
                failed_day,
            )
            save_rows(rows, output_path)
            if last_success_day is not None:
                print(f"last complete date saved: {last_success_day.isoformat()}")
            print(f"failed date: {failed_day.isoformat()}")
        else:
            print("No complete daily rows were downloaded, so no Excel file was saved.", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    print()
    print(df.to_string(index=False))

    output = args.output or default_output_path(args.stn, start, end)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)
    print(f"\nsaved: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
