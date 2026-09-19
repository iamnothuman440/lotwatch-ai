"""변경 전·후 COA 비교 테스트 — 실행: python test_before_after.py

모든 데이터는 sample_data/의 가상(synthetic) 테스트 데이터입니다.
변경 전 = Lot 1~20, 변경 후 = Lot 21~40으로 나눠, 합쳐서 분석한 결과가 기존 단일 COA 분석과 같은지 확인합니다.
"""
import io
import json

import pandas as pd

import ai_report
import analyzer as az

SPECS = {"Moisture": (0.0, 0.5), "Purity": (99.0, 100.0)}
CHANGES = az.load_changes_csv("sample_data/changes_sample.csv")
WIDE = pd.read_csv("sample_data/coa_sample.csv", dtype=str, encoding="utf-8-sig")  # Lot 001~040, 시험방법 A→B(035)
LONG = pd.read_csv("sample_data/coa_long_sample.csv", dtype=str, encoding="utf-8-sig")  # 같은 값, LOT001~040, 시험방법 없음
SINGLE = az.run_analysis(az.load_coa_csv("sample_data/coa_sample.csv"), SPECS, CHANGES, 30)  # 기존 단일 COA 분석 (기준)


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


def load(df, name):
    """파일 읽기 → 형식 인식(Wide/Long) → 표준 형식 (파일마다 따로)"""
    return az.load_coa_csv(as_file(df, name))


def kpi(r):
    return r["total_lots"], r["oos_lots"], r["quality_changes"], r["method_change_count"], r["check_recommended"]


def same_quality_analysis(r):
    """품질 특성 분석(추세·변화점·상태·4M 비교)이 기존 단일 COA 분석과 같은지"""
    for item in az.QUALITY_ITEMS:
        for k in ("trend", "status", "oos_count"):
            assert r["items"][item][k] == SINGLE["items"][item][k], (item, k)
        c, s = r["items"][item]["change"], SINGLE["items"][item]["change"]
        assert all(c[k] == s[k] for k in ("index", "before_mean", "after_mean", "detected")), item
        assert (r["items"][item]["four_m"] or {}).get("nearest") == (SINGLE["items"][item]["four_m"] or {}).get("nearest")


def run(before, after, window=30):
    return az.run_before_after(before, after, SPECS, CHANGES, window)


WB, WA = load(WIDE.iloc[:20], "before.csv"), load(WIDE.iloc[20:], "after.csv")  # Wide: Lot, Date, Supplier, ...
LB, LA = load(LONG.iloc[:40], "before.csv"), load(LONG.iloc[40:], "after.csv")  # Long: LotNo, InspectDate, Parameter, ...

# 1~4. 형식 조합 (컬럼명·형식이 달라도 각각 표준화한 뒤 합쳐서 기존 분석기로 분석)
r1 = run(WB, WA)
assert kpi(r1) == kpi(SINGLE) == (40, 0, 1, 1, 2) and r1["events"] == SINGLE["events"]
same_quality_analysis(r1)
print("1. Wide → Wide: 기존 단일 분석과 동일 (40 Lot, 품질 변화 1, 시험방법 변경 1, 확인 권고 2)")

r2 = run(load(WIDE.iloc[:20], "before.xlsx"), LA)  # 변경 전 Excel(Wide) + 변경 후 CSV(Long)
assert r2["total_lots"] == 40 and r2["items"]["Moisture"]["change"]["lot"] == "LOT021"
same_quality_analysis(r2)
print("2. Wide(xlsx) → Long(csv): 품질 분석 동일, 변화 시작 LOT021")

r3 = run(LB, WA)
assert r3["total_lots"] == 40 and r3["items"]["Moisture"]["change"]["lot"] == "021"
same_quality_analysis(r3)
print("3. Long → Wide: 품질 분석 동일, 변화 시작 021")

r4 = run(LB, LA)
assert r4["total_lots"] == 40 and r4["items"]["Moisture"]["change"]["lot"] == "LOT021"
same_quality_analysis(r4)
print("4. Long → Long: 품질 분석 동일, 변화 시작 LOT021")

# 5~6. 결합 + 날짜순 정렬 (분석기에는 표준 6개 컬럼만, 전후 구분은 행 순서대로 따로 보관)
assert list(r1["data"].columns[:6]) == az.COA_COLUMNS and "Change_Period" not in r1["data"]
assert r1["data"]["Lot"].tolist() == WIDE["Lot"].tolist()
assert r1["comparison"]["periods"] == ["Before"] * 20 + ["After"] * 20
print("5. 결합: 변경 전 20 + 변경 후 20 = 40행, 분석기에는 표준 컬럼만 전달")
shuffled = run(load(WIDE.iloc[:20].sample(frac=1, random_state=1), "b.csv"), load(WIDE.iloc[20:].sample(frac=1, random_state=2), "a.csv"))
assert shuffled["data"]["Date"].is_monotonic_increasing and kpi(shuffled) == kpi(SINGLE)
same_day = run(WB.iloc[[-1]].assign(Lot="B-last"), WB.iloc[[-1]].assign(Lot="A-first"))  # 같은 날짜: 변경 전이 먼저
assert same_day["data"]["Lot"].tolist() == ["B-last", "A-first"] and same_day["comparison"]["periods"] == ["Before", "After"]
print("6. 날짜순 정렬: 행 순서를 섞어도 날짜순으로 합쳐지고 결과 동일 (같은 날짜는 변경 전 먼저)")

# 7~9. 전후 평균 · 변화량 · 변화율 (단순 평균 비교)
m = r1["comparison"]["items"]["Moisture"]
assert (round(m["before_mean"], 4), round(m["after_mean"], 4)) == (0.1435, 0.2685)
print(f"7. 전후 평균: Moisture {m['before_mean']:.4f} → {m['after_mean']:.4f}")
assert round(m["diff"], 4) == 0.125
print(f"8. 변화량: {m['diff']:+.4f}")
assert round(m["diff_pct"], 1) == 87.1 and m["diff_pct"] == m["diff"] / m["before_mean"] * 100
print(f"9. 변화율: {m['diff_pct']:+.1f}%")

# 10. 변경 전 평균이 0이면 변화율을 계산하지 않음 (0으로 나누지 않음)
zero = run(WB.assign(Moisture=0.0), WA)
z = zero["comparison"]["items"]["Moisture"]
assert z["diff_pct"] is None and round(z["diff"], 4) == 0.2685
assert ai_report.build_summary(zero)["변경_전후_비교"]["품질특성별_평균_비교"]["Moisture"]["변화율_퍼센트"] == "계산 불가 (변경 전 평균 0)"
assert "변화율 계산 불가" in ai_report.build_template_report(zero)
print("10. 변경 전 평균 0: 변화율 '계산 불가' (오류 없음)")

# 11. 두 파일에 같은 Lot → 경고용 목록, 두 행 모두 분석에 유지 (자동으로 합치지 않음)
dup = run(WB, load(WIDE.iloc[19:], "a.csv"))  # Lot 020이 양쪽에 있음
assert dup["comparison"]["duplicate_lots"] == ["020"] and dup["total_lots"] == 41
assert (dup["data"]["Lot"] == "020").sum() == 2
assert r1["comparison"]["duplicate_lots"] == []
print("11. 중복 Lot: ['020'] 경고, 두 행 모두 유지 (41 Lot)")

# 12. 변경 후 날짜가 변경 전보다 빠르면 경고만 하고, 사용자가 지정한 전후 구분은 그대로 유지
rev = run(WA, WB)  # 사용자가 Lot 21~40을 '변경 전'으로 지정
assert rev["comparison"]["dates_reversed"] and not r1["comparison"]["dates_reversed"]
assert rev["comparison"]["periods"] == ["After"] * 20 + ["Before"] * 20  # 날짜순이지만 구분은 뒤집지 않음
assert rev["comparison"]["before"]["period"][0] == WA["Date"].min()
print("12. 날짜 역전: 경고, 전후 구분은 사용자 지정 그대로")

# 13~16. Test_Method
assert r1["has_method"] and [(e["lot"], e["from"], e["to"]) for e in r1["method_changes"]] == [("035", "A", "B")]
ab = run(WB.assign(Test_Method="A"), WA.assign(Test_Method="B"))  # 변경 전 A / 변경 후 B
assert [(e["lot"], e["from"], e["to"]) for e in ab["method_changes"]] == [("021", "A", "B")]
print("13. 양쪽 모두 있음: 시험방법 변경 분석 수행 (변경 전 A · 변경 후 B → Lot 021부터 A → B 감지)")
for name, r in (("14. 변경 전만 있음", run(WB, LA)), ("15. 변경 후만 있음", run(LB, WA))):
    assert not r["has_method"] and r["method_changes"] == [] and r["method_change_count"] == 0
    assert r["method_notice"] == az.PARTIAL_METHOD_NOTICE and az.PARTIAL_METHOD_NOTICE in r["notices"]
    assert (r["data"]["Test_Method"] != "").sum() == 20  # 시험방법이 있는 쪽의 값은 그대로 유지
    summary = ai_report.build_summary(r)
    assert summary["시험방법_변경_건수"] == "분석하지 않음" and summary["시험방법_변경"] == az.PARTIAL_METHOD_NOTICE
    assert "변경 없음" not in json.dumps(summary, ensure_ascii=False)
    print(f"{name}: 미분석 + '{az.PARTIAL_METHOD_NOTICE}' ('변경 없음' 아님)")
assert not r4["has_method"] and r4["method_notice"] == az.NO_METHOD_NOTICE and r4["method_change_count"] == 0
print(f"16. 양쪽 모두 없음: 미분석 + '{az.NO_METHOD_NOTICE}'")

# 17~18. 기존 단일 COA 분석은 그대로 (전후 비교 정보 없음)
assert "comparison" not in SINGLE and SINGLE["has_method"] and SINGLE["method_notice"] is None
print("17. 기존 Wide 단일 COA: 40/0/1/1/2 그대로")
single_long = az.run_analysis(az.load_coa_csv("sample_data/coa_long_sample.csv"), SPECS, CHANGES, 30)
assert kpi(single_long) == (40, 0, 1, 0, 1) and single_long["method_notice"] == az.NO_METHOD_NOTICE
assert "comparison" not in single_long
print("18. 기존 Long 단일 COA: 자동 인식 후 40/0/1/미분석/1 그대로")

# 19. 4M 비교는 기존과 같음 (날짜 근접성만, 전후 파일 구분과 무관)
assert [e["four_m"] for e in r1["events"]] == [e["four_m"] for e in SINGLE["events"]]
wide60 = run(WB, WA, 60)
assert [len(e["four_m"]["nearby"]) for e in wide60["events"]] == [2, 1] and wide60["check_recommended"] == 0
print("19. 4M 비교: 기존과 동일 (30일: 범위 밖, 60일: 근접 기록 2·1건)")

# 20. AI 리포트: 단일 모드 요약은 그대로, 전후 모드는 전후 정보를 추가로 전달 (Key 없이 기본 리포트로 확인)
assert "변경_전후_비교" not in ai_report.build_summary(SINGLE)
ba = ai_report.build_summary(r2)["변경_전후_비교"]
assert ba["변경_전"] == {"Lot수": 20, "기간": "2025-12-15 ~ 2026-04-27", "시험방법": "A"}
assert ba["변경_후"] == {"Lot수": 20, "기간": "2026-05-04 ~ 2026-09-14", "시험방법": "정보 없음"}
assert ba["품질특성별_평균_비교"]["Moisture"] == {"변경_전_평균": 0.1435, "변경_후_평균": 0.2685, "변화량": 0.125, "변화율_퍼센트": 87.1}
assert "4M_비교" in ai_report.build_summary(r2)["품질특성별_결과"]["Moisture"]  # 4M 근접성도 함께 전달
text, reason = ai_report.generate_report(r2, None)
assert reason and ai_report.FALLBACK_NOTICE in text and "Moisture 변경 전·후 평균: 0.144 → 0.269 (+0.125, +87.1%)" in text
for rule in ("공급업체가 원료를 변경해서 Moisture가 상승했다",
             "변경 후 Moisture 평균이 상승했으며, 해당 기간의 4M 변경 이력과 시간적 관계를 확인할 필요가 있습니다",
             "4M 변경 이력이 없다는 것은 '등록된 기록이 없다'는 뜻일 뿐"):
    assert rule in ai_report.SYSTEM_PROMPT
print("20. AI 리포트: 단일 요약 그대로, 전후 Lot 수·기간·평균·변화량·변화율·시험방법·4M 전달, 단정 금지 규칙 포함")

# 예시 파일(변경 전 Wide · 변경 후 Long)을 합치면 기존 단일 샘플과 같은 결과
b, a = az.load_coa_csv("sample_data/coa_before_sample.csv"), az.load_coa_csv("sample_data/coa_after_sample.csv")
for path, long in (("sample_data/coa_before_sample.csv", False), ("sample_data/coa_after_sample.csv", True)):
    assert az.is_long_format(az.guess_coa_mapping(az.read_table(path).columns)) == long  # app.py의 형식 표시와 일치
sample = run(b, a)
assert kpi(sample) == kpi(SINGLE) and [(e["lot"], e["title"]) for e in sample["events"]] == [
    ("LOT021", "Moisture 평균 상승"), ("LOT035", "시험방법 A → B")]
print("예시 파일: 변경 전 Wide + 변경 후 Long → 기존 단일 샘플과 같은 결과")

print("모든 테스트 통과")
