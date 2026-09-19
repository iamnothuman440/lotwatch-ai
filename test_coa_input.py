"""COA 입력 단계 테스트 (컬럼 자동 인식 · CSV/Excel · 세로형 Long Format) — 실행: python test_coa_input.py"""
import io

import json

import pandas as pd

import ai_report
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


def expect_error(file, text, run=analyze):
    try:
        run(file)
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

# 12. 가로형 회귀: 세로형으로 오인하지 않음 (PASS/FAIL용 'Test Result' 컬럼이 있어도), 같은 컬럼을 Moisture·Purity에 연결하면 오류
assert not az.is_long_format(az.guess_coa_mapping([*SAMPLE.columns, "Test Result"]))
expect_error(as_file(SAMPLE, "w.csv"), "같은 원본 컬럼(Moisture)",
             lambda f: az.prepare_coa(az.read_table(f), {**az.guess_coa_mapping(SAMPLE.columns), "Purity": "Moisture"}))
blank_method = analyze(as_file(SAMPLE.assign(Test_Method=""), "m.csv"))
assert not blank_method["has_method"] and az.NO_METHOD_NOTICE in blank_method["notices"]
print("12. 가로형 회귀: A·B·C 그대로 통과, 세로형으로 오인하지 않음")

# ── 세로형(Long Format) COA: 1행 = Lot 1개의 시험항목 1개 ─────────────
LONG = pd.read_csv("sample_data/coa_long_sample.csv", dtype=str, encoding="utf-8-sig")


def long_analyze(file):
    """앱과 같은 순서: 읽기 → 컬럼 자동 인식 → 세로형 인식 → 표준 표로 변환(pivot)·검증 → 기존 분석기"""
    raw = az.read_table(file)
    mapping = az.guess_coa_mapping(raw.columns)
    assert az.is_long_format(mapping), mapping
    df, specs, ignored = az.prepare_long_coa(raw, mapping)
    return df, specs, ignored, az.run_analysis(df, SPECS, CHANGES, 30)


def long_error(df, text):
    return expect_error(as_file(df, "bad.csv"), text, long_analyze)


# 1. 정상 Long Format CSV
df1, specs1, ignored1, r1 = long_analyze(as_file(LONG, "l.csv"))
assert (r1["total_lots"], r1["oos_lots"], r1["quality_changes"], r1["method_change_count"], r1["check_recommended"]) == (40, 0, 1, 0, 1)
assert r1["items"]["Moisture"]["change"]["lot"] == "LOT021" and ignored1 == []
print("1. 세로형 CSV: 정상 분석 (40 Lot, 품질 변화 1 · Lot LOT021부터)")

# 2. 정상 Long Format XLSX (엑셀처럼 숫자·날짜 셀로 저장)
xlsx = LONG.assign(InspectDate=pd.to_datetime(LONG["InspectDate"]),
                   **{c: LONG[c].astype(float) for c in ("LSL", "USL", "MeasuredValue")})
df2, specs2, _, r2 = long_analyze(as_file(xlsx, "l.xlsx"))
pd.testing.assert_frame_equal(df2, df1)
assert specs2 == specs1 and r2["events"] == r1["events"]
print("2. 세로형 XLSX: CSV와 같은 변환·분석 결과")

# 3. Moisture/Purity pivot: 가로형 샘플과 값·날짜·순서가 같고, 변화점 분석 결과도 같음 (행 순서를 섞어도 동일)
pd.testing.assert_frame_equal(df1.drop(columns=["Lot", "Test_Method"]), a["data"][["Date", "Supplier", "Moisture", "Purity"]])
assert df1["Lot"].tolist() == ("LOT" + a["data"]["Lot"]).tolist()
for item in az.QUALITY_ITEMS:
    assert r1["items"][item]["change"]["index"] == a["items"][item]["change"]["index"]
    assert r1["items"][item]["trend"] == a["items"][item]["trend"]
pd.testing.assert_frame_equal(long_analyze(as_file(LONG.sample(frac=1, random_state=0), "s.csv"))[0], df1)
print("3. pivot: 가로형 샘플과 동일한 Moisture/Purity 값, 행 순서를 섞어도 같은 결과")

# 4. Parameter 이름 alias 인식 (LOD · Loss on Drying · 기타 항목은 자동 매핑하지 않고 제외)
aliased = LONG.copy()
aliased["Parameter"] = [({"Moisture": ["수분", "Water Content (%)", "moisture content", "수분 함량"],
                          "Purity": ["Assay", "순도(%)", "PURITY (%)", "순도"]}[p])[i // 2 % 4] for i, p in enumerate(LONG["Parameter"])]
extra = LONG[LONG["Parameter"] == "Moisture"].assign(Parameter="LOD", MeasuredValue="0.2")
extra2 = extra.assign(Parameter="Color", MeasuredValue="White")  # 숫자가 아니어도 분석 대상이 아니면 오류 아님
df4, _, ignored4, _ = long_analyze(as_file(pd.concat([aliased, extra, extra2]), "p.csv"))
pd.testing.assert_frame_equal(df4, df1)
assert ignored4 == ["Color", "LOD"]
print("   LOD만 있는 경우:", long_error(LONG.replace({"Parameter": {"Moisture": "Loss on Drying"}}), "Moisture 데이터가 없습니다"))
print("4. Parameter alias: 수분·Water Content (%)·Assay·순도(%) 등 인식, LOD·Color는 제외")

# 5. LotNo/InspectDate/VendorName 및 다른 이름 alias 인식 (Manufacturer·제조사는 Supplier로 매핑하지 않음)
assert {k: v for k, v in az.guess_coa_mapping(LONG.columns).items() if k in ("Lot", "Date", "Supplier")} == \
    {"Lot": "LotNo", "Date": "InspectDate", "Supplier": "VendorName"}
renamed = LONG.rename(columns={"LotNo": "Batch Number", "InspectDate": "검사일", "VendorName": "납품업체",
                               "Parameter": "시험항목", "MeasuredValue": "측정값"})
pd.testing.assert_frame_equal(long_analyze(as_file(renamed, "r.csv"))[0], df1)
assert az.guess_coa_mapping(["BatchNo", "공급자"]) == {"Lot": "BatchNo", "Supplier": "공급자"}
assert "Supplier" not in az.guess_coa_mapping(["Manufacturer", "Manufacturer Name", "제조사"])
print("5. 컬럼 alias: LotNo·InspectDate·VendorName·Batch Number·검사일·납품업체 인식, Manufacturer 제외")

# 6. Test_Method 없음 → 오류 아님, 빈 값(미기재) + 시험방법 변경 분석 안 함 / 있으면 기존처럼 분석
assert (df1["Test_Method"] == "").all() and not r1["has_method"] and az.NO_METHOD_NOTICE in r1["notices"]
with_method = LONG.assign(TestMethod=["A" if int(lot[3:]) < 35 else "B" for lot in LONG["LotNo"]])
r6 = long_analyze(as_file(with_method, "t.csv"))[3]
assert r6["has_method"] and [(e["lot"], e["from"], e["to"]) for e in r6["method_changes"]] == [("LOT035", "A", "B")]
per_item = LONG.assign(TestMethod=LONG["Parameter"].map({"Moisture": "KF", "Purity": "GC"}))
assert set(long_analyze(as_file(per_item.iloc[::-1], "k.csv"))[0]["Test_Method"]) == {"KF / GC"}
print("6. Test_Method 없음: 오류 없이 분석, 안내 문구 표시 / 있으면 LOT035 A → B 감지")

# 7. LSL/USL: 시험항목별로 값이 하나면 참고 규격으로 추출, Lot마다 다르면 추출하지 않음(None), 컬럼이 없으면 {}
assert specs1 == {"Moisture": (0.0, 0.5), "Purity": (99.0, 100.0)}
varied = LONG.copy()
varied.loc[0, "USL"] = "0.45"
assert long_analyze(as_file(varied, "v.csv"))[1] == {"Moisture": (0.0, None), "Purity": (99.0, 100.0)}
assert long_analyze(as_file(LONG.drop(columns=["LSL", "USL"]), "n.csv"))[1] == {}
print("7. LSL/USL:", specs1)

# 8. 동일 Lot + Parameter 중복 → 임의로 고르지 않고 오류 (Lot/시험항목/행 번호 안내)
dup = pd.concat([LONG, LONG.iloc[[0]].assign(MeasuredValue="0.15")], ignore_index=True)
print("8. 중복:", long_error(dup, "동일 Lot의 동일 시험항목이 여러 건 존재합니다. 데이터를 확인해주세요. (LOT001 / Moisture: 2, 82행)"))

# 9. MeasuredValue 누락 (빈 칸, N/A)
for missing in ("", "N/A"):
    m = LONG.copy()
    m.loc[40, "MeasuredValue"] = missing
    msg = long_error(m, "MeasuredValue 컬럼에 빈 값(N/A 포함)이 있습니다 (확인할 행: 42 [LOT021 / Moisture])")
print("9. 측정값 누락:", msg)

# 10. 잘못된 날짜
m = LONG.copy()
m.loc[5, "InspectDate"] = "2026-13-45"
print("10. 날짜 오류:", long_error(m, "Date 형식을 확인해주세요. 예: 2026-01-05 (확인할 행: 7)"))

# 11. 숫자로 바꿀 수 없는 측정값
m = LONG.copy()
m.loc[40, "MeasuredValue"] = "abc"
print("11. 숫자 오류:", long_error(m, "MeasuredValue 컬럼 42행 [LOT021 / Moisture]의 'abc'"))

# 그 밖의 누락·불일치: Lot/Supplier/Parameter 빈 값, 한 Lot에 Purity 없음, 같은 Lot인데 날짜가 다름
for col, row, text in (("LotNo", 3, "Lot 값이 비어 있는 행이 있습니다 (확인할 행: 5)"),
                       ("VendorName", 3, "Supplier 값이 비어 있는 행이 있습니다 (확인할 행: 5)"),
                       ("InspectDate", 3, "Date 값이 비어 있는 행이 있습니다 (확인할 행: 5)"),
                       ("Parameter", 3, "Parameter 값이 비어 있는 행이 있습니다 (확인할 행: 5)")):
    m = LONG.copy()
    m.loc[row, col] = ""
    long_error(m, text)
print("   Purity 누락:", long_error(LONG.drop(index=9), "Purity 측정값이 없는 Lot이 있습니다: LOT005"))
m = LONG.copy()
m.loc[5, "InspectDate"] = "2026-01-20"
print("   날짜 불일치:", long_error(m, "같은 Lot인데 행마다 Date 또는 Supplier가 다릅니다: LOT003"))
print("   필수 컬럼 누락:", long_error(LONG.drop(columns=["VendorName"]), "다음 필수 컬럼을 인식하지 못했습니다: Supplier"))

# 13. 가로형인데 Test_Method 컬럼이 없음 → 오류 없이 분석, 시험방법 변경 분석만 건너뜀 (CSV·Excel)
no_method = company.drop(columns=["시험방법"])
for name in ("nm.csv", "nm.xlsx"):
    r13 = analyze(as_file(no_method, name))
    pd.testing.assert_frame_equal(r13["data"].drop(columns="Test_Method"), a["data"].drop(columns="Test_Method"))
    assert (r13["data"]["Test_Method"] == "").all() and not r13["has_method"] and az.NO_METHOD_NOTICE in r13["notices"]
    assert (r13["quality_changes"], r13["method_change_count"], r13["check_recommended"]) == (1, 0, 1)
    for item in az.QUALITY_ITEMS:  # 품질 특성 분석 결과는 Test_Method가 있을 때와 같음
        assert all(r13["items"][item][k] == a["items"][item][k] for k in ("trend", "change", "status", "four_m"))
    assert r13["events"] == [e for e in a["events"] if e["kind"] == "품질 변화"]
print("13. 가로형 Test_Method 없음: 오류 없이 분석, 품질 분석 결과는 동일, 시험방법 변경 분석만 건너뜀")
print("    필수 컬럼 안내:", expect_error(as_file(company.drop(columns=["수분(%)", "시험방법"]), "nm2.csv"),
                                   "다음 필수 컬럼을 인식하지 못했습니다: Moisture"))

# 14. AI 리포트: 시험방법 정보가 없으면 '변경 없음'이 아니라 '분석하지 않음'으로 전달 (가로형·세로형)
for r in (r13, r1):
    summary = ai_report.build_summary(r)
    assert summary["시험방법_변경"] == az.NO_METHOD_NOTICE
    assert summary["시험방법_변경_건수"] == "분석하지 않음"
    assert "변경 없음" not in json.dumps(summary, ensure_ascii=False)
    report = ai_report.build_template_report(r)
    assert f"- 시험방법: {az.NO_METHOD_NOTICE}" in report and "시험방법 변경 0건" not in report
# 시험방법 정보가 있고 변경이 없을 때만 '변경 없음' / 변경이 있으면 기존처럼 건수 전달
same_method = ai_report.build_summary(analyze(as_file(SAMPLE.assign(Test_Method="A"), "sm.csv")))
assert (same_method["시험방법_변경"], same_method["시험방법_변경_건수"]) == ("변경 없음", 0)
assert ai_report.build_summary(a)["시험방법_변경_건수"] == 1
print("14. AI 요약·기본 리포트: 시험방법 정보 없음 → '분석하지 않음' 전달, '변경 없음'으로 쓰지 않음")

print("모든 테스트 통과")
