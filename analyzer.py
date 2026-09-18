"""
LotWatch AI - 통계 분석 로직 (analyzer.py)

화면(UI)과 관계없는 '계산'만 이 파일에 모았습니다. app.py가 아래 함수들을 불러 씁니다.

- read_table() → guess_coa_mapping() → prepare_coa() : COA 파일(CSV/Excel) 읽기 → 컬럼 자동 인식 → 표준 컬럼으로 변환·검증
- load_coa_csv()      : 위 세 단계를 한 번에 (샘플 데이터용)
- load_changes_csv()  : 4M 변경 이력 CSV 읽기 + 검증
- run_analysis()      : 규격 판정 → 추세 비교 → 변화점 탐지 → 시험방법 변경 → 4M 비교를 한 번에 실행
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
    "Lot": ["Lot", "Lot No.", "Lot Number", "Batch", "Batch No.", "Batch Number"],
    "Date": ["Date", "Test Date", "Inspection Date", "검사일", "시험일", "분석일"],
    "Supplier": ["Supplier", "Supplier Name", "Vendor", "공급업체", "공급사"],
    "Moisture": ["Moisture", "Moisture (%)", "Water Content", "수분", "수분(%)"],
    "Purity": ["Purity", "Purity (%)", "Assay", "순도", "순도(%)"],
    "Test_Method": ["Test_Method", "Test Method", "Method", "시험방법", "분석방법"],
}
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


def _row_numbers(mask, limit=5):
    """문제가 있는 행 번호를 엑셀 기준(제목 줄 = 1행)으로 알려줍니다."""
    rows = [str(i + 2) for i in mask[mask].index[:limit]]
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


def _parse_numbers(series, col):
    text = _clean_text(series)
    blank = text == ""
    if blank.any():
        raise DataError(f"{col} 컬럼에 빈 값이 있습니다 (확인할 행: {_row_numbers(blank)}). 값을 입력하거나 해당 행을 삭제해주세요.")
    numbers = pd.to_numeric(text, errors="coerce")
    bad = ~np.isfinite(numbers)
    if bad.any():
        row = bad[bad].index[0]
        raise DataError(f"수치형 데이터가 아닌 값이 발견되었습니다: {col} 컬럼 {row + 2}행의 '{text[row]}' (숫자만 입력해주세요)")
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
    """COA 파일을 읽고 컬럼을 자동 인식해 검증합니다. (샘플 데이터처럼 확인 화면이 필요 없을 때)"""
    raw = read_table(file)
    return prepare_coa(raw, guess_coa_mapping(raw.columns))


def prepare_coa(raw, mapping):
    """{표준 컬럼: 원본 컬럼} 매핑대로 표준 6개 컬럼 표를 만들고, 값을 검증한 뒤 날짜순으로 정렬해서 돌려줍니다."""
    missing = [c for c in COA_COLUMNS if c not in mapping]
    if missing:
        raise DataError(f"다음 필수 컬럼을 인식하지 못했습니다: {', '.join(missing)}")
    df = pd.DataFrame({c: raw[mapping[c]] for c in COA_COLUMNS}).dropna(how="all")
    if df.empty:
        raise DataError("파일에 데이터가 없습니다. 제목 줄 아래에 Lot 데이터를 입력해주세요.")

    df["Lot"] = _clean_text(df["Lot"])
    if (df["Lot"] == "").any():
        raise DataError(f"Lot 값이 비어 있는 행이 있습니다 (확인할 행: {_row_numbers(df['Lot'] == '')}).")
    df["Date"] = _parse_dates(df["Date"], "COA 파일")
    for col in QUALITY_ITEMS:
        df[col] = _parse_numbers(df[col], col)
    df["Supplier"] = _clean_text(df["Supplier"])
    df["Test_Method"] = _clean_text(df["Test_Method"])

    # 같은 날짜는 파일에 적힌 순서를 유지합니다(stable).
    return df.sort_values("Date", kind="stable").reset_index(drop=True)


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
    days: 4M 날짜 - 변화 날짜 (음수 = 변화보다 먼저, 양수 = 변화보다 나중)"""
    if changes is None:
        return {"has_data": False, "nearby": [], "nearest": None}
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
    return {"has_data": True, "nearby": nearby, "nearest": nearest}


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
def run_analysis(coa, specs, changes=None, window_days=30):
    """모든 분석을 실행하고 대시보드와 AI 리포트에 필요한 결과를 dict로 돌려줍니다.

    coa     : load_coa_csv()가 돌려준 DataFrame
    specs   : {"Moisture": (최소, 최대), "Purity": (최소, 최대)}
    changes : load_changes_csv()가 돌려준 DataFrame 또는 None(4M 데이터 없음)
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
    method_changes = detect_method_changes(data)
    if (data["Test_Method"] == "").all():
        notices.append("Test_Method 값이 모두 비어 있어 시험방법 변경 분석은 건너뛰었습니다.")
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
