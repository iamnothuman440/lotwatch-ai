"""
LotWatch AI - 통계 분석 로직 (analyzer.py)

화면(UI)과 관계없는 '계산'만 이 파일에 모았습니다. app.py가 아래 함수들을 불러 씁니다.

- read_table() → guess_coa_mapping() → prepare_coa() : COA 파일(CSV/Excel) 읽기 → 컬럼 자동 인식 → 표준 컬럼으로 변환·검증
- is_long_format() → prepare_long_coa() : 세로형(Long Format) COA를 표준 컬럼 표로 변환(pivot)·검증
- load_coa_csv()      : 위 세 단계를 한 번에 (샘플 데이터용)
- load_changes_csv()  : 4M 변경 이력 CSV 읽기 + 검증
- run_analysis()      : 규격 판정 → 추세 비교 → 변화점 탐지 → 시험방법 변경 → 4M 비교를 한 번에 실행
- run_before_after()  : 변경 전·후 표준 표를 합쳐(날짜순) run_analysis()로 분석 + compare_before_after()로 전후 평균 비교
"""

import re
import warnings

import numpy as np
import pandas as pd
from scipy import stats

# ── 기본 설정값 ─────────────────────────────────────────────
COA_COLUMNS = ["Lot", "Date", "Supplier", "Moisture", "Purity", "Test_Method"]  # 분석기에 전달되는 표준 컬럼
# 공급사마다 다른 COA 컬럼명 → 표준 컬럼 (대소문자·공백·괄호·_·-·.·% 차이는 무시하고 비교)
COLUMN_ALIASES = {
    "Lot": ["Lot", "Lot No.", "Lot Number", "LotNo", "Batch", "Batch No.", "Batch Number", "BatchNo"],
    "Date": ["Date", "Test Date", "Inspection Date", "InspectDate", "검사일", "시험일", "분석일"],
    # Manufacturer(제조사)는 공급사와 다를 수 있어 넣지 않습니다.
    "Supplier": ["Supplier", "Supplier Name", "Vendor", "VendorName", "공급업체", "공급사", "공급자", "납품업체"],
    # 세로형 COA의 Parameter 값(시험항목 이름)도 이 두 목록으로 판별합니다. LOD(Loss on Drying)는 수분과 다를 수 있어 넣지 않습니다.
    "Moisture": ["Moisture", "Moisture (%)", "Moisture Content", "Moisture Content (%)", "Water Content",
                 "Water Content (%)", "수분", "수분(%)", "수분 함량", "수분함량"],
    "Purity": ["Purity", "Purity (%)", "Assay", "순도", "순도(%)"],
    "Test_Method": ["Test_Method", "Test Method", "Method", "시험방법", "분석방법"],
    # 아래는 세로형(Long Format) COA 전용 컬럼
    "Parameter": ["Parameter", "Test Item", "시험항목", "검사항목"],
    "MeasuredValue": ["MeasuredValue", "Measured Value", "Result Value", "Test Result", "측정값", "측정 결과", "시험결과"],
    "LSL": ["LSL", "Lower Spec Limit", "규격 하한", "하한"],
    "USL": ["USL", "Upper Spec Limit", "규격 상한", "상한"],
}
WIDE_COLUMNS = ["Lot", "Date", "Supplier", "Moisture", "Purity"]  # 가로형 COA 필수 컬럼 (Test_Method는 있으면 사용)
LONG_COLUMNS = ["Lot", "Date", "Supplier", "Parameter", "MeasuredValue"]  # 세로형 COA 필수 컬럼
LONG_OPTIONAL = ["Test_Method", "LSL", "USL"]  # 세로형 COA에 있으면 쓰는 컬럼
NO_METHOD_NOTICE = "시험방법 정보가 입력 데이터에 없어 시험방법 변경 분석은 수행하지 않습니다."
PARTIAL_METHOD_NOTICE = "시험방법 정보가 일부 데이터에 없어 시험방법 변경 분석을 수행하지 않습니다."  # 변경 전·후 중 한쪽에만 있을 때
CHANGE_COLUMNS = ["Date", "Type", "Description"]  # 4M 변경 이력 CSV 필수 컬럼
QUALITY_ITEMS = ["Moisture", "Purity"]  # 분석할 품질 특성

RECOMMENDED_LOTS = 10  # 권장 최소 Lot 수
MIN_SEGMENT = 4  # 변화 전/후 구간에 각각 필요한 최소 Lot 수
P_VALUE_LIMIT = 0.01  # t-검정 유의수준: 우연히 생긴 차이일 확률이 1% 미만일 때만 인정
EFFECT_LIMIT = 1.5  # 평균 차이가 평소 변동폭(표준편차)의 1.5배 이상일 때만 인정
TREND_WINDOW = 10  # 추세 비교: 초기 10 Lot 평균 vs 최근 10 Lot 평균

# 상태 이름 (대시보드에서 색상과 연결됩니다)
OK = "정상"
WATCH = "주의"  # 변화 감지 + 근접한 4M 변경 이력 있음
CHECK = "확인 필요"  # 변화 감지 + 근접한 4M 변경 이력 없음
OOS = "규격 부적합"


class DataError(Exception):
    """사용자 화면에 그대로 보여줄 수 있는 데이터 오류"""


# ── CSV 읽기와 검증 ─────────────────────────────────────────
def _read_csv(file):
    """CSV를 글자(문자열)로 읽습니다. 엑셀에서 저장한 한글 CSV(CP949)도 읽을 수 있게 두 가지 인코딩을 시도합니다."""
    for encoding in ("utf-8-sig", "cp949"):
        try:
            if hasattr(file, "seek"):
                file.seek(0)  # 업로드 파일은 다시 읽기 전에 처음 위치로 되돌립니다.
            return pd.read_csv(file, dtype=str, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            raise DataError("CSV 파일이 비어 있습니다.")
        except pd.errors.ParserError:
            raise DataError("CSV 형식을 읽을 수 없습니다. 쉼표(,)로 구분된 CSV 파일인지 확인해주세요.")
    raise DataError("CSV 파일의 글자 인코딩을 읽을 수 없습니다. 엑셀에서 'CSV UTF-8' 형식으로 다시 저장해주세요.")


def _standardize_columns(df, expected, file_label):
    """컬럼 이름의 공백·대소문자 차이를 무시하고 표준 이름으로 맞춘 뒤, 빠진 컬럼이 있는지 확인합니다.
    예) ' moisture ' → 'Moisture', 'Test Method' → 'Test_Method'"""
    lookup = {name.lower(): name for name in expected}
    new_names = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_")
        if key in lookup:
            new_names[col] = lookup[key]
    df = df.rename(columns=new_names)

    missing = [name for name in expected if name not in df.columns]
    if missing:
        raise DataError(
            f"필수 컬럼이 없습니다. {file_label}에 {', '.join(missing)} 컬럼이 필요합니다.\n\n"
            f"(필수 컬럼: {', '.join(expected)})"
        )
    # 필요한 컬럼만 남기고, 모든 칸이 빈 줄은 지웁니다.
    return df[expected].dropna(how="all").copy()


def _row_numbers(mask, limit=5, labels=None):
    """문제가 있는 행 번호를 엑셀 기준(제목 줄 = 1행)으로 알려줍니다. labels가 있으면 행 설명도 붙입니다. 예) 23 [LOT021 / Moisture]"""
    rows = [str(i + 2) + (f" [{labels[i]}]" if labels is not None else "") for i in mask[mask].index[:limit]]
    more = " 등" if mask.sum() > limit else ""
    return ", ".join(rows) + more


def _clean_text(series):
    return series.fillna("").astype(str).str.strip()


def _parse_dates(series, file_label):
    text = _clean_text(series)
    dates = pd.to_datetime(text, errors="coerce", format="mixed")
    bad = dates.isna()
    if bad.any():
        raise DataError(f"{file_label}의 Date 형식을 확인해주세요. 예: 2026-01-05 (확인할 행: {_row_numbers(bad)})")
    return dates.dt.normalize()


def _parse_numbers(series, col, labels=None):
    text = _clean_text(series)
    blank = text == ""  # 'N/A', 'NA' 같은 글자도 파일을 읽을 때 빈 값으로 바뀝니다.
    if blank.any():
        raise DataError(f"{col} 컬럼에 빈 값(N/A 포함)이 있습니다 (확인할 행: {_row_numbers(blank, labels=labels)}). "
                        "값을 입력하거나 해당 행을 삭제해주세요.")
    numbers = pd.to_numeric(text, errors="coerce")
    bad = ~np.isfinite(numbers)
    if bad.any():
        row = bad[bad].index[0]
        where = f" [{labels[row]}]" if labels is not None else ""
        raise DataError(f"수치형 데이터가 아닌 값이 발견되었습니다: {col} 컬럼 {row + 2}행{where}의 '{text[row]}' (숫자만 입력해주세요)")
    return numbers.astype(float)


def read_table(file):
    """CSV 또는 Excel(.xlsx, 첫 번째 시트)을 글자(문자열) 표로 읽습니다."""
    if str(getattr(file, "name", file)).lower().endswith(".xlsx"):
        try:
            return pd.read_excel(file, sheet_name=0, dtype=str)
        except Exception:
            raise DataError("Excel 파일을 읽을 수 없습니다. 첫 번째 시트 1행에 컬럼 제목이 있는 .xlsx 파일인지 확인해주세요.")
    return _read_csv(file)


def _norm(name):
    """컬럼명 비교용: 소문자로 바꾸고 글자·숫자가 아닌 것(공백, 괄호, _, -, ., % 등)은 모두 지웁니다."""
    return re.sub(r"[\W_]+", "", str(name).lower())


def guess_coa_mapping(columns):
    """원본 컬럼명을 COLUMN_ALIASES와 비교해 {표준 컬럼: 원본 컬럼}을 돌려줍니다. (못 찾은 컬럼은 빠짐)"""
    mapping = {}
    for std, aliases in COLUMN_ALIASES.items():
        names = {_norm(a) for a in aliases}
        match = next((c for c in columns if _norm(c) in names), None)
        if match is not None:
            mapping[std] = match
    return mapping


def load_coa_csv(file):
    """COA 파일을 읽고 형식(Wide/Long)과 컬럼을 자동 인식해 표준 표로 돌려줍니다. (샘플 데이터처럼 확인 화면이 필요 없을 때)"""
    raw = read_table(file)
    mapping = guess_coa_mapping(raw.columns)
    return prepare_long_coa(raw, mapping)[0] if is_long_format(mapping) else prepare_coa(raw, mapping)


def prepare_coa(raw, mapping):
    """{표준 컬럼: 원본 컬럼} 매핑대로 표준 6개 컬럼 표를 만들고, 값을 검증한 뒤 날짜순으로 정렬해서 돌려줍니다.
    Test_Method 컬럼이 없으면 빈 값(미기재)으로 채웁니다. → 시험방법 변경 분석만 건너뜀"""
    missing = [c for c in WIDE_COLUMNS if c not in mapping]
    if missing:
        raise DataError(f"다음 필수 컬럼을 인식하지 못했습니다: {', '.join(missing)}")
    if mapping["Moisture"] == mapping["Purity"]:
        raise DataError(f"Moisture와 Purity에 같은 원본 컬럼({mapping['Moisture']})을 연결할 수 없습니다. "
                        "한 컬럼에 여러 시험항목의 값이 있는 세로형 COA라면 입력 형식을 Long Format으로 선택해주세요.")
    df = pd.DataFrame({c: raw[mapping[c]] for c in COA_COLUMNS if c in mapping}).dropna(how="all")
    if df.empty:
        raise DataError("파일에 데이터가 없습니다. 제목 줄 아래에 Lot 데이터를 입력해주세요.")

    df["Lot"] = _clean_text(df["Lot"])
    if (df["Lot"] == "").any():
        raise DataError(f"Lot 값이 비어 있는 행이 있습니다 (확인할 행: {_row_numbers(df['Lot'] == '')}).")
    df["Date"] = _parse_dates(df["Date"], "COA 파일")
    for col in QUALITY_ITEMS:
        df[col] = _parse_numbers(df[col], col)
    df["Supplier"] = _clean_text(df["Supplier"])
    df["Test_Method"] = _clean_text(df["Test_Method"]) if "Test_Method" in df else ""

    # 같은 날짜는 파일에 적힌 순서를 유지합니다(stable).
    return df.sort_values("Date", kind="stable").reset_index(drop=True)


def is_long_format(mapping):
    """시험항목(Parameter) 컬럼과 측정값(MeasuredValue) 컬럼이 모두 있으면 세로형(Long Format) COA로 봅니다."""
    return "Parameter" in mapping and "MeasuredValue" in mapping


def prepare_long_coa(raw, mapping):
    """세로형 COA(1행 = Lot 1개의 시험항목 1개)를 prepare_coa()와 같은 표준 표(1행 = Lot 1개)로 바꿉니다.
    Parameter가 Moisture인 행의 MeasuredValue → Moisture, Purity인 행 → Purity (그 밖의 시험항목은 제외)
    반환: (표준 표, 파일의 규격 {품질 특성: (LSL, USL)}, 제외한 시험항목 목록)"""
    missing = [c for c in LONG_COLUMNS if c not in mapping]
    if missing:
        raise DataError(f"다음 필수 컬럼을 인식하지 못했습니다: {', '.join(missing)}")
    df = pd.DataFrame({c: raw[mapping[c]] for c in LONG_COLUMNS + LONG_OPTIONAL if c in mapping}).dropna(how="all")
    if df.empty:
        raise DataError("파일에 데이터가 없습니다. 제목 줄 아래에 Lot 데이터를 입력해주세요.")

    # 1) 시험항목 이름 → Moisture / Purity (대소문자·공백·괄호 차이 무시)
    param = _clean_text(df["Parameter"])
    if (param == "").any():
        raise DataError(f"Parameter 값이 비어 있는 행이 있습니다 (확인할 행: {_row_numbers(param == '')}).")
    lookup = {_norm(a): item for item in QUALITY_ITEMS for a in COLUMN_ALIASES[item]}
    df["Parameter"] = param.map(lambda p: lookup.get(_norm(p)))
    ignored = sorted(set(param[df["Parameter"].isna()]))
    for item in QUALITY_ITEMS:
        if not (df["Parameter"] == item).any():
            raise DataError(f"{item} 데이터가 없습니다. Parameter 값에서 {item} 항목을 찾지 못했습니다. "
                            f"(파일의 시험항목: {', '.join(sorted(set(param)))})")
    df = df[df["Parameter"].notna()].copy()  # 분석 대상(Moisture, Purity) 행만 검증·변환합니다.

    # 2) 값 검증 (행 번호는 원본 파일 기준)
    for col in ("Lot", "Date", "Supplier"):
        df[col] = _clean_text(df[col])
        if (df[col] == "").any():
            raise DataError(f"{col} 값이 비어 있는 행이 있습니다 (확인할 행: {_row_numbers(df[col] == '')}).")
    df["Date"] = _parse_dates(df["Date"], "COA 파일")
    df["MeasuredValue"] = _parse_numbers(df["MeasuredValue"], "MeasuredValue", df["Lot"] + " / " + param.loc[df.index])
    dup = df.duplicated(["Lot", "Parameter"], keep=False)
    if dup.any():  # 어느 값을 쓸지 임의로 고르지 않습니다.
        groups = list(df[dup].groupby(["Lot", "Parameter"], sort=False))
        detail = "; ".join(f"{lot} / {item}: {', '.join(str(i + 2) for i in g.index)}행" for (lot, item), g in groups[:5])
        raise DataError(f"동일 Lot의 동일 시험항목이 여러 건 존재합니다. 데이터를 확인해주세요. ({detail}{' 등' if len(groups) > 5 else ''})")
    per_lot = df.groupby("Lot", sort=False)[["Date", "Supplier"]].nunique()
    mixed = per_lot.index[(per_lot > 1).any(axis=1)]
    if len(mixed):
        raise DataError(f"같은 Lot인데 행마다 Date 또는 Supplier가 다릅니다: {', '.join(mixed[:5])}. "
                        "한 Lot의 행에는 같은 날짜와 공급사를 입력해주세요.")

    # 3) Lot별 1행으로 변환 (Lot 순서는 파일에 처음 나온 순서)
    out = df.groupby("Lot", sort=False)[["Date", "Supplier"]].first()
    out = out.join(df.pivot(index="Lot", columns="Parameter", values="MeasuredValue"))
    for item in QUALITY_ITEMS:
        lacking = out.index[out[item].isna()]
        if len(lacking):
            raise DataError(f"{item} 측정값이 없는 Lot이 있습니다: {', '.join(lacking[:5])}{' 등' if len(lacking) > 5 else ''}. "
                            "각 Lot에 Moisture와 Purity 행이 모두 있어야 합니다.")
    if "Test_Method" in df:  # 시험항목별 시험방법을 Moisture → Purity 순서로 합칩니다. 예) KF / HPLC
        df["Test_Method"] = _clean_text(df["Test_Method"])
        out["Test_Method"] = (df.sort_values("Parameter", kind="stable").groupby("Lot")["Test_Method"]
                              .agg(lambda s: " / ".join(dict.fromkeys(filter(None, s)))))
    else:
        out["Test_Method"] = ""  # 시험방법 정보 없음 (기존 분석기에서 빈 값 = 미기재)

    # 4) 파일의 규격(LSL/USL): 시험항목별로 값이 하나일 때만 참고값으로 돌려줍니다.
    specs = {}
    for item in QUALITY_ITEMS:
        rows = df[df["Parameter"] == item]
        limits = [pd.to_numeric(_clean_text(rows[c]), errors="coerce").dropna().unique() if c in rows else []
                  for c in ("LSL", "USL")]
        spec = tuple(float(v[0]) if len(v) == 1 else None for v in limits)
        if spec != (None, None):
            specs[item] = spec

    out = out.reset_index()[COA_COLUMNS]
    return out.sort_values("Date", kind="stable").reset_index(drop=True), specs, ignored


def load_changes_csv(file):
    """4M 변경 이력 CSV(Date, Type, Description)를 읽고 검증합니다."""
    df = _standardize_columns(_read_csv(file), CHANGE_COLUMNS, "4M 변경 이력 CSV")
    df["Date"] = _parse_dates(df["Date"], "4M 변경 이력 CSV")
    df["Type"] = _clean_text(df["Type"])
    df["Description"] = _clean_text(df["Description"])
    return df.sort_values("Date", kind="stable").reset_index(drop=True)


# ── 개별 분석 함수 ──────────────────────────────────────────
def check_specs(df, specs):
    """각 Lot이 규격(최소~최대, 경계값 포함) 안에 있는지 판정합니다: PASS / OUT OF SPEC"""
    data = df.copy()
    all_pass = pd.Series(True, index=data.index)
    for item in QUALITY_ITEMS:
        low, high = specs[item]
        passed = data[item].between(low, high)
        data[f"{item}_판정"] = np.where(passed, "PASS", "OUT OF SPEC")
        all_pass &= passed
    data["종합판정"] = np.where(all_pass, "PASS", "OUT OF SPEC")
    return data


def compare_recent(values, window=TREND_WINDOW):
    """초기 N Lot 평균(기준)과 최근 N Lot 평균, 전체 평균을 비교합니다."""
    w = max(1, min(window, len(values) // 2))
    first_mean = values[:w].mean()
    recent_mean = values[-w:].mean()
    diff = recent_mean - first_mean
    return {
        "window": w,
        "overall_mean": values.mean(),
        "first_mean": first_mean,
        "recent_mean": recent_mean,
        "diff": diff,
        "diff_pct": diff / abs(first_mean) * 100 if first_mean != 0 else None,
    }


def detect_change_point(values):
    """평균이 달라지는 지점(변화점)을 1개 찾고, 그 변화가 의미 있는지 판단합니다.

    1) 데이터를 '앞 구간'과 '뒤 구간'으로 나누는 모든 위치를 하나씩 시험합니다.
    2) 두 구간을 각자의 평균으로 설명했을 때 남는 오차(제곱합)가 가장 작은 위치를 변화점 후보로 고릅니다.
       (ruptures 라이브러리의 L2 비용 변화점 탐지와 같은 원리를, 설치 없이 이해하기 쉽게 직접 구현했습니다.)
    3) 아래 두 조건을 모두 만족하면 '변화 감지'로 판단합니다.
       - 앞뒤 평균 차이가 평소 변동폭(구간 안의 표준편차)의 1.5배 이상
       - Welch t-검정 p-value < 0.01
    데이터가 너무 적으면(8 Lot 미만) None을 돌려줍니다.
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < 2 * MIN_SEGMENT:
        return None

    best_k, best_cost = None, np.inf
    for k in range(MIN_SEGMENT, n - MIN_SEGMENT + 1):
        before, after = x[:k], x[k:]
        cost = ((before - before.mean()) ** 2).sum() + ((after - after.mean()) ** 2).sum()
        if cost < best_cost:
            best_k, best_cost = k, cost

    before, after = x[:best_k], x[best_k:]
    shift = after.mean() - before.mean()
    noise = np.sqrt(best_cost / (n - 2))  # 평소 변동폭 (두 구간의 합동 표준편차)

    if noise < 1e-9:  # 두 구간 모두 값이 일정한 경우 (예: 99.5가 계속되다가 99.4가 계속됨)
        effect = np.inf if abs(shift) > 1e-9 else 0.0
        p_value = 0.0 if abs(shift) > 1e-9 else 1.0
    else:
        effect = abs(shift) / noise
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # 값이 거의 같을 때 나오는 계산 경고는 숨깁니다.
            p_value = stats.ttest_ind(before, after, equal_var=False).pvalue

    return {
        "index": best_k,  # 변화 후 첫 Lot의 위치(0부터 셈)
        "before_mean": before.mean(),
        "after_mean": after.mean(),
        "shift": shift,
        "shift_pct": shift / abs(before.mean()) * 100 if before.mean() != 0 else None,
        "effect": effect,
        "p_value": p_value,
        "detected": bool(p_value < P_VALUE_LIMIT and effect >= EFFECT_LIMIT),
    }


def detect_method_changes(data):
    """시간순으로 Test_Method를 비교해서 바뀐 시점을 찾습니다. (빈 값은 건너뜁니다)"""
    events = []
    previous = None
    for i, method in enumerate(data["Test_Method"]):
        if method == "":
            continue
        if previous is not None and method != previous:
            events.append({
                "index": i,
                "lot": data["Lot"].iloc[i],
                "date": data["Date"].iloc[i],
                "from": previous,
                "to": method,
            })
        previous = method
    return events


def compare_with_4m(event_date, changes, window_days):
    """변화 시점 전후 window_days일 안에 4M 변경 이력이 있는지 확인합니다.
    days: 4M 날짜 - 변화 날짜 (음수 = 변화보다 먼저, 양수 = 변화보다 나중)
    records: 모든 4M 기록 (AI 리포트에 기록마다 범위 안/밖 판정을 붙여 전달하기 위해 함께 돌려줌)"""
    if changes is None:
        return {"has_data": False, "records": [], "nearby": [], "nearest": None}
    records = [
        {
            "date": row["Date"],
            "type": row["Type"],
            "description": row["Description"],
            "days": int((row["Date"] - event_date).days),
        }
        for _, row in changes.iterrows()
    ]
    nearby = [r for r in records if abs(r["days"]) <= window_days]
    nearest = min(records, key=lambda r: abs(r["days"])) if records else None
    return {"has_data": True, "records": records, "nearby": nearby, "nearest": nearest}


def describe_4m(four_m, subject):
    """4M 비교 결과를 문장으로 바꿉니다. subject 예: '품질 변화가', '시험방법 변경이'"""
    if not four_m["has_data"]:
        return "4M 변경 이력 데이터가 없어 비교하지 못했습니다. 공급사에 해당 기간의 변경 여부를 확인해 주세요."
    if four_m["nearby"]:
        return "변화 시점과 근접한 4M 변경 이력이 있습니다. 변경 내용과 품질 영향을 확인해 주세요."
    return (f"{subject} 감지되었으나 해당 기간 내 등록된 4M 변경 이력이 확인되지 않았습니다. "
            "공급사에 변경 여부 및 원인 확인을 권고합니다.")


def days_text(days):
    """4M 날짜와 변화 시점의 간격을 글로 표현합니다."""
    if days == 0:
        return "변화 시점과 같은 날"
    return f"변화 시점 {abs(days)}일 {'후' if days > 0 else '전'}"


# ── 전체 분석 ──────────────────────────────────────────────
def run_analysis(coa, specs, changes=None, window_days=30, method_notice=None):
    """모든 분석을 실행하고 대시보드와 AI 리포트에 필요한 결과를 dict로 돌려줍니다.

    coa     : load_coa_csv()가 돌려준 DataFrame
    specs   : {"Moisture": (최소, 최대), "Purity": (최소, 최대)}
    changes : load_changes_csv()가 돌려준 DataFrame 또는 None(4M 데이터 없음)
    method_notice : 시험방법 변경 분석을 하지 않을 이유(안내 문구). None이면 Test_Method 값으로 판단합니다.
    """
    if len(coa) < 2:
        raise DataError("분석할 데이터가 충분하지 않습니다. 최소 10개 이상의 Lot을 권장합니다.")

    data = check_specs(coa, specs)
    n = len(data)
    notices = []
    if n < RECOMMENDED_LOTS:
        notices.append(f"분석할 데이터가 충분하지 않습니다. 최소 {RECOMMENDED_LOTS}개 이상의 Lot을 권장합니다. (현재 {n}개)")
    if n < 2 * MIN_SEGMENT:
        notices.append(f"Lot 수가 적어 변화점 분석은 건너뛰었습니다. (최소 {2 * MIN_SEGMENT}개 필요)")

    events = []  # 감지된 모든 변화(품질 특성 변화 + 시험방법 변경)

    # 1) 품질 특성별 분석: 규격 판정, 추세 비교, 변화점 탐지, 4M 비교
    items = {}
    for item in QUALITY_ITEMS:
        values = data[item].to_numpy()
        oos_mask = data[f"{item}_판정"] == "OUT OF SPEC"
        change = detect_change_point(values)
        four_m = None

        if change is None:
            message = "Lot 수가 적어 변화점 분석을 하지 못했습니다."
        elif change["detected"]:
            k = change["index"]
            change.update({
                "lot": data["Lot"].iloc[k],
                "date": data["Date"].iloc[k],
                "prev_date": data["Date"].iloc[k - 1],
                "before_lots": (data["Lot"].iloc[0], data["Lot"].iloc[k - 1]),
                "after_lots": (data["Lot"].iloc[k], data["Lot"].iloc[-1]),
            })
            direction = "상승" if change["shift"] > 0 else "하락"
            message = f"Lot {change['lot']}부터 평균이 {direction}하는 변화가 감지되었습니다."
            four_m = compare_with_4m(change["date"], changes, window_days)
            events.append({
                "kind": "품질 변화",
                "item": item,
                "title": f"{item} 평균 {direction}",
                "lot": change["lot"],
                "date": change["date"],
                "four_m": four_m,
                "four_m_message": describe_4m(four_m, "품질 변화가"),
            })
        else:
            message = "뚜렷한 평균 변화가 감지되지 않았습니다."

        detected = bool(change and change["detected"])
        if oos_mask.any():
            status = OOS
            spec_state = f"규격 부적합 {int(oos_mask.sum())} Lot"
        elif detected:
            status = WATCH if four_m["nearby"] else CHECK
            spec_state = "규격 이내 / 변화 확인 권고"
        else:
            status = OK
            spec_state = "규격 이내 / 변화 없음"

        items[item] = {
            "spec": specs[item],
            "oos_count": int(oos_mask.sum()),
            "oos_lots": data.loc[oos_mask, "Lot"].tolist(),
            "trend": compare_recent(values),
            "change": change,
            "detected": detected,
            "four_m": four_m,
            "status": status,
            "spec_state": spec_state,
            "message": message,
        }

    # 2) 시험방법 변경 탐지 + 4M 비교
    if method_notice is None and (data["Test_Method"] == "").all():
        method_notice = NO_METHOD_NOTICE
    method_changes = [] if method_notice else detect_method_changes(data)
    if method_notice:
        notices.append(method_notice)
    for ev in method_changes:
        ev["four_m"] = compare_with_4m(ev["date"], changes, window_days)
        ev["status"] = WATCH if ev["four_m"]["nearby"] else CHECK
        events.append({
            "kind": "시험방법 변경",
            "item": "Test_Method",
            "title": f"시험방법 {ev['from']} → {ev['to']}",
            "lot": ev["lot"],
            "date": ev["date"],
            "four_m": ev["four_m"],
            "four_m_message": describe_4m(ev["four_m"], "시험방법 변경이"),
        })
    events.sort(key=lambda e: e["date"])

    # 3) KPI와 종합 상태
    oos_lots = int((data["종합판정"] == "OUT OF SPEC").sum())
    quality_changes = sum(1 for e in events if e["kind"] == "품질 변화")  # 평균이 달라진 품질 특성 수
    method_change_count = len(method_changes)  # 시험방법 변경 횟수
    changes_detected = len(events)  # = quality_changes + method_change_count
    check_recommended = sum(1 for e in events if not e["four_m"]["nearby"])
    detected_text = " 및 ".join(  # 예: "품질 변화 1건 및 시험방법 변경 1건"
        f"{label} {count}건"
        for label, count in (("품질 변화", quality_changes), ("시험방법 변경", method_change_count))
        if count
    )

    if oos_lots > 0:
        overall = OOS
        message = f"규격 부적합 Lot이 {oos_lots}개 있습니다. 해당 Lot을 우선 확인해 주세요."
        if changes_detected:
            message += f" 이와 별도로 {detected_text}이 감지되었습니다."
    elif check_recommended > 0:
        overall = CHECK
        message = f"모든 Lot이 규격 이내이지만, 확인이 필요한 변화가 {check_recommended}건 감지되었습니다."
    elif changes_detected > 0:
        overall = WATCH
        message = f"모든 Lot이 규격 이내입니다. 감지된 {detected_text}은 근접한 4M 변경 이력이 있으니 변경 내용과 품질 영향을 확인해 주세요."
    else:
        overall = OK
        message = "모든 Lot이 규격 이내이며, 뚜렷한 품질 변화가 감지되지 않았습니다."

    return {
        "data": data,
        "total_lots": n,
        "period": (data["Date"].iloc[0], data["Date"].iloc[-1]),
        "suppliers": sorted(s for s in data["Supplier"].unique() if s),
        "specs": specs,
        "window_days": window_days,
        "changes_4m": changes,
        "items": items,
        "method_changes": method_changes,
        "has_method": method_notice is None,  # False = 시험방법 변경 분석을 하지 않음 (이유는 method_notice)
        "method_notice": method_notice,
        "events": events,
        "oos_lots": oos_lots,
        "changes_detected": changes_detected,
        "quality_changes": quality_changes,
        "method_change_count": method_change_count,
        "check_recommended": check_recommended,
        "overall_status": overall,
        "overall_message": message,
        "notices": notices,
    }


# ── 변경 전·후 COA 비교 ─────────────────────────────────────
def compare_before_after(before, after):
    """사용자가 지정한 변경 전·후 표준 표를 단순 비교합니다. (변화 감지 판정이 아닌 평균 비교)
    Lot 수·기간·시험방법, 품질 특성별 평균·변화량·변화율, 두 파일에 모두 있는 Lot, 날짜 역전 여부"""
    def summary(df):
        methods = list(dict.fromkeys(m for m in df["Test_Method"] if m))  # 빈 값 제외, 처음 나온 순서
        return {"lots": len(df), "period": (df["Date"].min(), df["Date"].max()), "methods": methods}

    items = {}
    for item in QUALITY_ITEMS:
        b, a = before[item].mean(), after[item].mean()
        items[item] = {"before_mean": b, "after_mean": a, "diff": a - b,
                       "diff_pct": (a - b) / abs(b) * 100 if b != 0 else None}  # 변경 전 평균이 0이면 변화율 없음
    b, a = summary(before), summary(after)
    return {
        "before": b,
        "after": a,
        "combined": {"lots": b["lots"] + a["lots"],
                     "period": (min(b["period"][0], a["period"][0]), max(b["period"][1], a["period"][1]))},
        "items": items,
        "duplicate_lots": sorted(set(before["Lot"]) & set(after["Lot"])),  # 경고만 하고 두 행 모두 분석에 사용
        "dates_reversed": bool(a["period"][0] < b["period"][1]),  # 변경 후의 일부 날짜가 변경 전보다 빠름
        "partial_method": bool(b["methods"]) != bool(a["methods"]),  # 시험방법 정보가 한쪽 파일에만 있음
    }


def run_before_after(before, after, specs, changes=None, window_days=30):
    """변경 전·후 표준 표를 합쳐 날짜순으로 정렬한 뒤(같은 날짜는 변경 전 먼저) 기존 run_analysis()로 분석합니다.
    전후 구분은 사용자가 지정한 그대로 쓰고, 분석기에는 표준 6개 컬럼만 넘깁니다. 전후 비교는 result["comparison"]에 붙입니다."""
    comparison = compare_before_after(before, after)
    combined = (pd.concat([before.assign(Change_Period="Before"), after.assign(Change_Period="After")], ignore_index=True)
                .sort_values("Date", kind="stable").reset_index(drop=True))
    # 시험방법 정보가 한쪽 파일에만 있으면 전후 시험방법을 비교할 수 없으므로 시험방법 변경 분석은 하지 않습니다.
    notice = PARTIAL_METHOD_NOTICE if comparison["partial_method"] else None
    result = run_analysis(combined[COA_COLUMNS], specs, changes, window_days, notice)
    comparison["periods"] = combined["Change_Period"].tolist()  # result["data"]와 같은 행 순서
    result["comparison"] = comparison
    return result
