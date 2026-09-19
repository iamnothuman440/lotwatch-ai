"""AI 리포트의 4M 시간관계 테스트 — 실행: python test_ai_4m.py

Python(analyzer)이 판정한 4M 시간관계(범위 안/밖/등록 이력 없음)가 Gemini 입력에 그대로 들어가는지 확인합니다.
기준 데이터: sample_data/ 가상 샘플 — Moisture 변화 Lot 021 (2026-05-04), 시험방법 A → B Lot 035 (2026-08-10)
"""
import io

import analyzer as az
import ai_report

SPECS = {"Moisture": (0.0, 0.5), "Purity": (99.0, 100.0)}
COA = az.load_coa_csv("sample_data/coa_sample.csv")


def changes(*rows):
    """4M 변경 이력 CSV를 만들어 앱과 같은 함수로 읽습니다. rows: (날짜, 유형, 내용)"""
    text = "Date,Type,Description\n" + "".join(f"{d},{t},{desc}\n" for d, t, desc in rows)
    f = io.BytesIO(text.encode("utf-8-sig"))
    f.name = "4m.csv"
    return az.load_changes_csv(f)


def moisture_4m(four_m_changes, window=30):
    """Gemini에 보내는 요약에서 Moisture 변화(2026-05-04)의 4M 비교 부분"""
    result = az.run_analysis(COA, SPECS, four_m_changes, window)
    return ai_report.build_summary(result)["품질특성별_결과"]["Moisture"]["4M_비교"], result


def records(relation):
    return {r["date"]: (r["temporal_relation"], r["within_window"], r["days_from_change"]) for r in relation["four_m_records"]}


# Test 1 — 범위 안: 2026-05-01은 변화 시점 3일 전 → within_range
rel, _ = moisture_4m(changes(("2026-05-01", "Equipment", "설비 교체")))
assert rel["change_date"] == "2026-05-04" and rel["comparison_window"] == "2026-04-04 ~ 2026-06-03 (변화 시점 전후 30일)"
assert rel["temporal_relation"] == "within_range" and records(rel) == {"2026-05-01": ("within_range", True, -3)}
assert rel["four_m_records"][0]["설명"] == "변화 시점 3일 전 (변화 이전 기록, 비교 범위 안)" and rel["nearest_4m"] == "2026-05-01"
print("Test 1 범위 안: 2026-05-01 → within_range (변화 시점 3일 전)")

# Test 2 — 범위 밖: 2026-07-01은 변화 시점 58일 후 → out_of_range, 변화 자체는 no_record
out = changes(("2026-07-01", "Material", "원료 공급처 변경"))
rel, result = moisture_4m(out)
assert records(rel) == {"2026-07-01": ("out_of_range", False, 58)}
assert rel["temporal_relation"] == "no_record" and "가장 가까운 등록 이력은 비교 범위 밖" in rel["판정"]
assert rel["four_m_records"][0]["설명"] == "변화 시점 58일 후 (변화 이후 기록, 비교 범위 밖)"
summary = ai_report.build_summary(result)
assert "2026-07-01" not in summary["4M_변경_이력"]  # 판정 없는 원본 날짜 목록은 보내지 않음 (건수만)
report = ai_report.build_template_report(result)
line = next(l for l in report.splitlines() if "2026-07-01" in l and "Moisture" in l)
assert "비교 범위 밖" in line and not any(w in report for w in ("관련 가능성", "관련성", "원인일 수", "관련 변경"))
for rule in ('temporal_relation이 "out_of_range"(비교 범위 밖)인 기록은 품질 변화와 관련된 변경으로 표현하거나 관련 가능성·원인 가능성이 있다고 제시하지 마세요',
             "가장 가까운 등록 이력은 비교 범위 밖에 있습니다.",
             "다시 계산하거나 시간적 관련성을 스스로 판단하지 마세요"):
    assert rule in ai_report.SYSTEM_PROMPT, rule
assert '"관련 가능성을 확인할 필요가 있습니다" 정도로 쓰세요' not in ai_report.SYSTEM_PROMPT  # 범위와 무관하게 관련 가능성을 권하던 옛 규칙 삭제
print(f"Test 2 범위 밖: 2026-07-01 → out_of_range (58일 후) · 기본 리포트: '{line.strip()}'")

# Test 3 — 등록 이력 없음: 범위 안 기록 없음 → no_record / 4M 기록 0건과 4M 파일 없음은 구분
rel, _ = moisture_4m(az.load_changes_csv("sample_data/changes_sample.csv"))
assert rel["temporal_relation"] == "no_record" and rel["registered_4m_count"] == 2
assert rel["판정"] == "해당 비교 범위 내 등록된 4M 변경 이력 없음 (가장 가까운 등록 이력은 비교 범위 밖)"
rel, _ = moisture_4m(changes())
assert rel["temporal_relation"] == "no_record" and rel["registered_4m_count"] == 0 and rel["four_m_records"] == []
assert rel["판정"] == "등록된 4M 변경 이력이 전혀 없음"
rel, _ = moisture_4m(None)
assert rel["temporal_relation"] == "no_4m_data" and "four_m_records" not in rel
assert "해당 비교 범위 내 등록된 4M 변경 이력이 없습니다" in ai_report.SYSTEM_PROMPT
assert "실제로 변경이 없었다는 뜻이 아닙니다" in ai_report.SYSTEM_PROMPT  # 기존 주의사항 유지
print("Test 3 등록 이력 없음: no_record (범위 밖 기록만 있음 / 기록 0건 구분), 4M 파일 없음은 no_4m_data")

# Test 4 — 범위 안 + 범위 밖 혼합: 기록마다 따로 판정
rel, result = moisture_4m(changes(("2026-05-01", "Equipment", "설비 교체"), ("2026-07-01", "Material", "원료 공급처 변경")))
assert rel["temporal_relation"] == "within_range" and rel["nearest_4m"] == "2026-05-01"
assert records(rel) == {"2026-05-01": ("within_range", True, -3), "2026-07-01": ("out_of_range", False, 58)}
assert result["items"]["Moisture"]["status"] == az.WATCH  # Python 판정(근접 이력 있음 → 주의)은 기존 그대로
print("Test 4 혼합: 2026-05-01 within_range / 2026-07-01 out_of_range 를 기록별로 구분")

# 경계값: 판정은 analyzer의 '전후 N일 이내(N일 포함)'를 그대로 따름 (AI 입력에서 다시 계산하지 않음)
assert records(moisture_4m(out, 58)[0])["2026-07-01"][0] == "within_range"
assert records(moisture_4m(out, 57)[0])["2026-07-01"][0] == "out_of_range"
print("경계값: 58일 차이는 비교 기간 58일이면 범위 안, 57일이면 범위 밖 (analyzer와 동일)")

# Test 5 — 기존 결과 유지: 샘플 분석·4M 판정·기본 리포트 문구, 변경 전·후 비교 모드의 4M 판정
sample = az.run_analysis(COA, SPECS, az.load_changes_csv("sample_data/changes_sample.csv"), 30)
assert (sample["total_lots"], sample["oos_lots"], sample["quality_changes"], sample["method_change_count"], sample["check_recommended"]) == (40, 0, 1, 1, 2)
assert [(e["lot"], len(e["four_m"]["nearby"]), e["four_m"]["nearest"]["days"]) for e in sample["events"]] == [("021", 0, -49), ("035", 0, -40)]
report = ai_report.build_template_report(sample)
assert "- Moisture 평균 상승의 가장 가까운 4M 기록: 2026-03-16 Equipment · 정기 설비 점검 (변화 시점 49일 전, 비교 범위 밖)" in report
method_rel = ai_report.build_summary(sample)["시험방법_변경"][0]["4M_비교"]
assert method_rel["temporal_relation"] == "no_record" and records(method_rel)["2026-07-01"] == ("out_of_range", False, -40)
before, after = az.load_coa_csv("sample_data/coa_before_sample.csv"), az.load_coa_csv("sample_data/coa_after_sample.csv")
compare = az.run_before_after(before, after, SPECS, az.load_changes_csv("sample_data/changes_sample.csv"), 30)
single_rel = ai_report.build_summary(sample)["품질특성별_결과"]["Moisture"]["4M_비교"]
assert ai_report.build_summary(compare)["품질특성별_결과"]["Moisture"]["4M_비교"] == single_rel
print("Test 5 기존 결과 유지: 샘플 40/0/1/1/2, 4M 판정(-49일·-40일, 범위 밖) 동일, 변경 전·후 모드도 같은 4M 판정")

print("모든 테스트 통과")
