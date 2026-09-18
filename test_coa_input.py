"""COA 입력 단계 테스트 (컬럼 자동 인식 · CSV/Excel) — 실행: python test_coa_input.py"""
import io

import pandas as pd

import analyzer as az

SPECS = {"Moisture": (0.0, 0.5), "Purity": (99.0, 100.0)}
SAMPLE = pd.read_csv("sample_data/coa_sample.csv", dtype=str, encoding="utf-8-sig")
CHANGES = az.load_changes_csv("sample_data/changes_sample.csv")
COMPANY_NAMES = {"Lot": "Batch No.", "Date": "검사일", "Supplier": "공급업체",
                 "Moisture": "수분(%)", "Purity": "순도(%)", "Test_Method": "시험방법"}


def as_file(df, name):
    """DataFrame을 업로드 파일처럼(CSV 또는 xlsx) 만듭니다."""
    f = io.BytesIO()
    if name.endswith(".xlsx"):
        df.to_excel(f, index=False)
    else:
        f.write(df.to_csv(index=False).encode("utf-8-sig"))
    f.seek(0)
    f.name = name
    return f


def analyze(file):
    """앱과 같은 순서: 읽기 → 컬럼 자동 인식 → 표준 컬럼 변환·검증 → 기존 분석기"""
    raw = az.read_table(file)
    return az.run_analysis(az.prepare_coa(raw, az.guess_coa_mapping(raw.columns)), SPECS, CHANGES, 30)


def expect_error(file, text):
    try:
        analyze(file)
    except az.DataError as e:
        assert text in str(e), str(e)
        return str(e)
    raise AssertionError("오류가 나야 합니다")


# A. 표준 컬럼 CSV → 정상 분석
a = analyze(as_file(SAMPLE, "a.csv"))
assert (a["total_lots"], a["oos_lots"], a["quality_changes"], a["method_change_count"], a["check_recommended"]) == (40, 0, 1, 1, 2)
assert a["items"]["Moisture"]["change"]["lot"] == "021"
print("A. 표준 CSV: 정상 분석 (40 Lot, 품질 변화 1, 시험방법 변경 1, 확인 권고 2)")

# B. 회사 컬럼명(Batch No., 검사일, ...) → 자동 매핑 후 A와 완전히 같은 결과 (CSV와 Excel 모두)
company = SAMPLE.rename(columns=COMPANY_NAMES)
assert az.guess_coa_mapping(company.columns) == COMPANY_NAMES
for name in ("b.csv", "b.xlsx"):
    b = analyze(as_file(company, name))
    pd.testing.assert_frame_equal(b["data"], a["data"])
    assert b["events"] == a["events"]
print("B. 회사 컬럼명 CSV·Excel: 자동 매핑 후 A와 동일한 분석 결과")

# 대소문자·공백·괄호·_·- 차이 무시, 관계없는 컬럼은 무시
messy = [" BATCH_NO ", "Inspection-Date", "vendor", "Water Content", "ASSAY (%)", "METHOD", "Remarks"]
assert az.guess_coa_mapping(messy) == {"Lot": " BATCH_NO ", "Date": "Inspection-Date", "Supplier": "vendor",
                                       "Moisture": "Water Content", "Purity": "ASSAY (%)", "Test_Method": "METHOD"}

# C. 필수 컬럼이 없는 CSV → 어떤 컬럼인지 정확히 알려주는 오류 (앱은 이 오류를 화면에 표시)
msg = expect_error(as_file(company.drop(columns=["수분(%)", "순도(%)"]), "c.csv"), "")
assert msg == "다음 필수 컬럼을 인식하지 못했습니다: Moisture, Purity", msg
print(f"C. 필수 컬럼 누락: '{msg}'")

# 값 변환 오류는 문제 행(엑셀 기준 행 번호)을 알려줌
bad = company.copy()
bad.loc[4, "수분(%)"] = "abc"
print("   값 오류:", expect_error(as_file(bad, "d.csv"), "Moisture 컬럼 6행의 'abc'"))
bad = company.copy()
bad.loc[2, "검사일"] = "어제"
print("   날짜 오류:", expect_error(as_file(bad, "e.xlsx"), "확인할 행: 4"))

print("모든 테스트 통과")
