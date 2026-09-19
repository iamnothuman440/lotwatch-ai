"""
LotWatch AI - AI 리포트 생성 (ai_report.py)

analyzer.py가 계산한 통계 결과를 Google Gemini API에 보내서,
QC 담당자가 읽기 쉬운 한국어 분석 리포트를 만듭니다.

- API Key는 st.secrets["GEMINI_API_KEY"] (.streamlit/secrets.toml 또는 Streamlit Cloud Secrets)에서만 읽습니다.
- API Key가 없거나, 호출이 실패하거나, 무료 사용 한도를 넘으면
  통계 분석 결과로 만든 '기본 요약 리포트'를 대신 돌려줍니다. (앱은 멈추지 않습니다)
"""

import json
import math
from datetime import timedelta

import streamlit as st
from google import genai
from google.genai import errors, types

import analyzer

# 사용할 Gemini 모델: 앞 모델이 한도 초과·일시 장애 등으로 실패하면 다음 모델을 시도합니다.
# (2026-09-19 실제 호출로 확인: 3.5 Flash-Lite는 빠르고 무료 사용량이 많음, 3.6 Flash는 예비용)
MODELS = ("gemini-3.5-flash-lite", "gemini-3.6-flash")
TIMEOUT_MS = 30_000  # 모델 하나당 응답을 최대 30초까지 기다립니다.
PLACEHOLDER_KEY = "여기에_내_Gemini_API_Key_입력"  # secrets.toml의 안내 문구 (실제 Key가 아님)

DISCLAIMER = "본 결과는 데이터 변화에 대한 확인 권고이며, 공급사의 변경이나 부적합을 확정하는 판단이 아닙니다."
FALLBACK_NOTICE = "AI 연결 실패로 기본 요약 리포트를 표시합니다."

SYSTEM_PROMPT = """당신은 제조업 품질관리(QC) 담당자를 돕는 품질 데이터 분석 보조자입니다.
LotWatch AI가 계산한 공급사 COA/수입검사 데이터의 통계 분석 결과를 받아, QC 담당자가 바로 이해하고 조치할 수 있는 한국어 리포트를 작성합니다.

작성 원칙
- 분석 결과에 있는 숫자와 사실만 사용하세요. 결과에 없는 원인을 추측해서 사실처럼 쓰지 마세요.
- Lot 번호, 날짜, 시험방법 이름(A, B 등), 4M 기록 내용은 분석 결과에 적힌 그대로 쓰고 수식어를 덧붙이지 마세요.
- 이 리포트는 '확인 권고'입니다. 공급사의 변경, 데이터 조작, 부정행위, 불량을 확정하는 표현은 절대 쓰지 마세요.
  (금지 예: "공급사가 몰래 변경했습니다", "공급사가 데이터를 조작했습니다", "불량입니다")
- 대신 "변화가 감지되었습니다", "확인이 필요합니다", "공급사 변경 가능성을 확인해 주세요", "해당 기간 내 등록된 4M 변경 이력이 확인되지 않았습니다"처럼 표현하세요.
- 4M 변경 이력이 없다는 것은 '등록된 기록이 없다'는 뜻일 뿐, 실제로 변경이 없었다는 뜻이 아닙니다.
- 규격 부적합(규격을 벗어남)과 규격 이내의 변화(평소와 달라짐)를 구분해서 설명하세요.
- 품질 변화(품질 특성 평균의 통계적 변화)와 시험방법 변경은 성격이 다르므로 구분해서 설명하세요.
- 시험방법_변경_건수가 '분석하지 않음'이면 시험방법 정보가 없거나 일부 데이터에만 있어 분석하지 않았다는 뜻입니다. "시험방법 변경 없음"으로 쓰지 말고, '시험방법_변경'에 적힌 이유대로 시험방법 변경 분석을 하지 않았다고 쓰세요.
- '변경_전후_비교'가 있으면 사용자가 변경 전·후로 지정한 두 COA를 합쳐 분석한 결과입니다.
  - 변경 전·후 평균, 변화량, 변화율은 적힌 그대로 인용하세요. 단순 평균 비교이므로, 통계적으로 의미 있는 변화인지는 품질특성별_결과의 '변화_감지'를 따르세요.
  - 파일이 변경 전·후로 나뉘어 있다는 사실이나 4M 기록만으로 변화의 원인을 단정하지 마세요. 금지 예: "공급업체가 원료를 변경해서 Moisture가 상승했다" / 허용 예: "변경 후 Moisture 평균이 상승했으며, 해당 기간의 4M 변경 이력과 시간적 관계를 확인할 필요가 있습니다"
  - 두 파일에 모두 있는 Lot이나 변경 전보다 빠른 변경 후 날짜가 있으면, 권장 확인사항에 데이터 확인을 포함하세요.
- 4M 변경 이력과 각 변화의 시간적 관계는 LotWatch AI(Python)가 이미 판정했고, 이 판정이 최종 기준입니다. 각 변화의 '4M_비교'에 적힌 temporal_relation과 '판정'을 그대로 따르고, 4M 날짜와 변화 시점의 차이나 비교 범위를 다시 계산하거나 시간적 관련성을 스스로 판단하지 마세요.
  - temporal_relation이 "within_range"(비교 범위 안)인 기록만 시간적으로 근접한 기록으로 설명하세요. 예: "품질 변화 시점과 가까운 4M 변경 이력이 등록되어 있습니다." 변경 내용과 품질 영향 확인을 권할 수는 있지만, 시간적 근접성은 인과관계를 뜻하지 않으므로 원인으로 단정하지 마세요. 변화 이후 기록(days_from_change가 양수)이면 해당 품질 변화의 원인으로 볼 수 없다는 점도 밝히세요.
  - temporal_relation이 "out_of_range"(비교 범위 밖)인 기록은 품질 변화와 관련된 변경으로 표현하거나 관련 가능성·원인 가능성이 있다고 제시하지 마세요. 권장 확인사항에도 이 기록과 품질 변화의 관련성 확인을 넣지 마세요. 필요할 때만 "가장 가까운 등록 이력은 비교 범위 밖에 있습니다." 정도로 쓰세요.
  - 변화의 temporal_relation이 "no_record"이면 "해당 비교 범위 내 등록된 4M 변경 이력이 없습니다"라고 쓰세요. 비교 범위 밖에만 기록이 있는 경우(관련 4M 변경 없음)와 등록된 4M 변경 이력이 전혀 없는 경우는 '판정'에 적힌 대로 구분하고, "no_4m_data"이면 4M 변경 이력이 제공되지 않아 비교하지 않았다고 쓰세요.
  - '변화 시점 N일 전/후' 표현은 '설명'에 적힌 그대로 옮겨 쓰고 전/후 방향을 바꾸지 마세요. 여러 변화를 한 문장에 묶지 말고 변화마다 따로 쓰세요.
  - 이 원칙은 '데이터 근거'와 '권장 확인사항'을 포함한 모든 섹션에 똑같이 적용하세요.
- 문장은 간결하게, 전문 용어는 짧게 풀어서 쓰세요.

출력 형식 (마크다운, 아래 네 제목을 그대로 사용)
#### 1. 핵심 발견
2~3문장 요약
#### 2. 데이터 근거
Lot 번호, 날짜, 평균값 등 수치를 bullet로 제시
#### 3. 확인이 필요한 이유
#### 4. 권장 확인사항
공급사 문의 또는 내부 점검 항목 3~5개를 bullet로 제시

면책 문구는 앱이 따로 표시하므로 쓰지 마세요."""


def get_api_key():
    """Gemini API Key를 st.secrets["GEMINI_API_KEY"]에서 읽습니다. 설정되지 않았으면 None."""
    try:
        key = str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:  # secrets.toml 파일이나 GEMINI_API_KEY 항목이 없을 때
        return None
    if not key or key == PLACEHOLDER_KEY:
        return None
    return key


def _num(value):
    return None if value is None else round(float(value), 4)


def _day(date):
    return date.strftime("%Y-%m-%d")


def _four_m_relation(four_m, event_date, window_days):
    """변화 시점과 4M 변경 이력의 시간적 관계를 AI에게 '판정 결과'로 넘깁니다. (AI가 날짜를 다시 계산하지 않도록)
    범위 안/밖은 analyzer.compare_with_4m()이 계산한 nearby를 그대로 옮깁니다.
    temporal_relation — 기록마다: within_range(비교 범위 안) / out_of_range(비교 범위 밖)
                      — 변화마다: within_range(범위 안 기록 있음) / no_record(범위 안 등록 이력 없음) / no_4m_data(4M 파일 없음)"""
    window = timedelta(days=window_days)
    relation = {
        "change_date": _day(event_date),
        "comparison_window": f"{_day(event_date - window)} ~ {_day(event_date + window)} (변화 시점 전후 {window_days}일)",
    }
    if not four_m["has_data"]:
        return {**relation, "temporal_relation": "no_4m_data", "판정": "4M 변경 이력 데이터가 제공되지 않아 비교하지 않음"}

    def record(r):
        within = r in four_m["nearby"]
        when = "변화 이후 기록" if r["days"] > 0 else "변화 이전 기록" if r["days"] < 0 else "변화와 같은 날 기록"
        return {
            "date": _day(r["date"]),
            "category": r["type"],
            "description": r["description"],
            "days_from_change": r["days"],
            "within_window": within,
            "temporal_relation": "within_range" if within else "out_of_range",
            "설명": f"{analyzer.days_text(r['days'])} ({when}, {'비교 범위 안' if within else '비교 범위 밖'})",
        }

    if four_m["nearby"]:
        verdict = "비교 범위 안에 등록된 4M 변경 이력이 있음 (시간적으로 가까울 뿐, 원인이라는 뜻이 아님)"
    elif four_m["records"]:
        verdict = "해당 비교 범위 내 등록된 4M 변경 이력 없음 (가장 가까운 등록 이력은 비교 범위 밖)"
    else:
        verdict = "등록된 4M 변경 이력이 전혀 없음"
    # ponytail: 4M 기록 전체를 변화마다 전달 — 기록이 수백 건이면 가까운 순 N건으로 줄일 것
    return {
        **relation,
        "temporal_relation": "within_range" if four_m["nearby"] else "no_record",
        "판정": verdict,
        "registered_4m_count": len(four_m["records"]),
        "four_m_records": [record(r) for r in four_m["records"]],
        "nearest_4m": _day(four_m["nearest"]["date"]) if four_m["nearest"] else None,
    }


def build_summary(result):
    """AI에게 보낼 분석 결과 요약(숫자와 사실만)을 만듭니다."""
    window = result["window_days"]
    items = {}
    for name, info in result["items"].items():
        trend = info["trend"]
        w = trend["window"]
        entry = {
            "규격": f"{info['spec'][0]} ~ {info['spec'][1]}",
            "규격_부적합_Lot수": info["oos_count"],
            "전체_평균": _num(trend["overall_mean"]),
            f"초기_{w}Lot_평균": _num(trend["first_mean"]),
            f"최근_{w}Lot_평균": _num(trend["recent_mean"]),
            "변화_감지": info["detected"],
            "상태": info["status"],
            "요약": info["message"],
        }
        if info["detected"]:
            c = info["change"]
            entry["변화점"] = {
                "변화_시작_Lot": c["lot"],
                "변화_시작일": _day(c["date"]),
                "변화_전_평균": _num(c["before_mean"]),
                "변화_전_구간": f"Lot {c['before_lots'][0]}~{c['before_lots'][1]}",
                "변화_후_평균": _num(c["after_mean"]),
                "변화_후_구간": f"Lot {c['after_lots'][0]}~{c['after_lots'][1]}",
                "평균_변화율_퍼센트": None if c["shift_pct"] is None else round(float(c["shift_pct"]), 1),
                "평소_변동폭_대비_배수": round(float(c["effect"]), 1) if math.isfinite(c["effect"]) else "매우 큼",
                "t검정_p값": "< 0.001" if c["p_value"] < 0.001 else round(float(c["p_value"]), 3),
            }
            entry["4M_비교"] = _four_m_relation(info["four_m"], c["date"], window)
        items[name] = entry

    changes = result["changes_4m"]
    start, end = result["period"]
    summary = {
        "공급사": ", ".join(result["suppliers"]) or "미기재",
        "분석_기간": f"{_day(start)} ~ {_day(end)}",
        "총_Lot": result["total_lots"],
        "규격_부적합_Lot수": result["oos_lots"],
        "품질_변화_감지_건수": result["quality_changes"],
        "시험방법_변경_건수": result["method_change_count"] if result["has_method"] else "분석하지 않음",
        "확인_권고_건수": result["check_recommended"],
        "종합_상태": result["overall_status"],
        "품질특성별_결과": items,
        "시험방법_변경": [
            {
                "변경": f"{e['from']} → {e['to']}",
                "시작_Lot": e["lot"],
                "시작일": _day(e["date"]),
                "4M_비교": _four_m_relation(e["four_m"], e["date"], window),
            }
            for e in result["method_changes"]
        ] or ("변경 없음" if result["has_method"] else result["method_notice"]),
        # 원본 날짜 목록 대신 건수만: 각 기록의 시간적 관계는 변화별 '4M_비교'에 판정과 함께 들어 있음
        "4M_변경_이력": f"{len(changes)}건 등록 (변화별 시간적 관계는 각 '4M_비교'의 temporal_relation 참고)"
                        if changes is not None else "제공되지 않음",
        "참고사항": result["notices"],
    }
    if result.get("comparison"):  # 변경 전·후 COA 비교 모드
        summary["변경_전후_비교"] = _before_after_summary(result["comparison"])
    return summary


def _before_after_summary(c):
    """변경 전·후 비교 모드에서 AI에게 추가로 알려줄 사실 (전후 구분은 사용자가 지정한 그대로)"""
    def period(p):
        return {"Lot수": p["lots"], "기간": f"{_day(p['period'][0])} ~ {_day(p['period'][1])}",
                "시험방법": ", ".join(p["methods"]) or "정보 없음"}

    return {
        "변경_전": period(c["before"]),
        "변경_후": period(c["after"]),
        "품질특성별_평균_비교": {
            item: {
                "변경_전_평균": _num(v["before_mean"]),
                "변경_후_평균": _num(v["after_mean"]),
                "변화량": _num(v["diff"]),
                "변화율_퍼센트": "계산 불가 (변경 전 평균 0)" if v["diff_pct"] is None else round(float(v["diff_pct"]), 1),
            }
            for item, v in c["items"].items()
        },
        "두_파일에_모두_있는_Lot": c["duplicate_lots"] or "없음",
        "변경_후_날짜가_변경_전보다_빠른_데이터": "있음" if c["dates_reversed"] else "없음",
    }


@st.cache_data(show_spinner=False, max_entries=50)
def _ask_gemini(summary, _api_key):
    """Gemini에 리포트를 요청합니다. 실패하면 예외를 던집니다.
    같은 분석 결과는 한 번만 요청하고 결과를 저장해 두어 무료 사용량을 아낍니다. (실패한 호출은 저장되지 않음)
    인자 이름이 밑줄(_)로 시작하면 Streamlit이 저장 기준에 쓰지 않으므로, Key는 어디에도 저장되지 않습니다."""
    client = genai.Client(api_key=_api_key, http_options=types.HttpOptions(timeout=TIMEOUT_MS))
    prompt = (
        "다음은 LotWatch AI가 공급사 COA/수입검사 데이터를 통계 분석한 결과입니다.\n"
        "이 결과를 바탕으로 QC 담당자를 위한 품질 분석 리포트를 작성해 주세요.\n\n"
        f"<analysis_result>\n{summary}\n</analysis_result>"
    )
    last_error = None
    for model in MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),  # 도구 호출 기능은 사용하지 않음
                ),
            )
        except Exception as error:
            # 모델 없음 · 한도 초과 · 서버 과부하 · 시간 초과 → 다음 모델 시도 / 그 외(Key 오류 등) → 바로 중단
            retry = getattr(error, "code", None) in (404, 429, 500, 503, 504) or "Timeout" in type(error).__name__
            if not retry:
                raise
            last_error = error
            continue
        text = (response.text or "").strip()
        if text:
            return text
        last_error = ValueError("Gemini 응답이 비어 있습니다.")
    raise last_error


def _error_message(error):
    """실패 원인을 사용자에게 보여줄 쉬운 문장으로 바꿉니다. (원래 오류 문구와 API Key는 보여주지 않음)"""
    code = getattr(error, "code", None)
    detail = str(getattr(error, "message", "") or "")
    if code == 429:
        return "Gemini 무료 사용 한도를 초과했습니다. 잠시 후(또는 내일) 다시 시도해 주세요."
    if code in (401, 403) or (code == 400 and "API key" in detail):
        return "Gemini API Key가 올바르지 않거나 사용 권한이 없습니다. secrets.toml의 GEMINI_API_KEY를 확인해 주세요."
    if code == 404:
        return "사용하려는 Gemini 모델을 찾을 수 없습니다. ai_report.py의 MODELS를 확인해 주세요."
    if isinstance(code, int) and code >= 500:
        return "Gemini 서버에 일시적인 문제가 있습니다. 잠시 후 다시 시도해 주세요."
    if isinstance(error, errors.APIError):
        return f"Gemini API 요청이 처리되지 않았습니다 (오류 코드 {code}). 잠시 후 다시 시도해 주세요."
    if "Timeout" in type(error).__name__:
        return "Gemini 응답 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요."
    return "Gemini에 연결하지 못했습니다. 인터넷 연결을 확인하고 잠시 후 다시 시도해 주세요."


def build_template_report(result):
    """AI를 쓸 수 없을 때 보여줄 기본 요약 리포트입니다. 통계 분석 결과만으로 AI 리포트와 같은 4단 구성으로 작성합니다."""
    window = result["window_days"]
    lines = [
        f"> ⚠️ **{FALLBACK_NOTICE}**",
        "> 아래 내용은 AI가 아닌 통계 분석 결과로 자동 작성된 요약입니다.",
        "",
        "#### 1. 핵심 발견",
        f"- 총 {result['total_lots']} Lot 중 규격 부적합 {result['oos_lots']} Lot, 품질 변화 {result['quality_changes']}건"
        + (f", 시험방법 변경 {result['method_change_count']}건" if result["has_method"] else "") + "이 감지되었습니다.",
        f"- 종합 상태: {result['overall_status']} — {result['overall_message']}",
        "",
        "#### 2. 데이터 근거",
    ]
    ba = result.get("comparison")
    if ba:  # 변경 전·후 비교 모드: 파일별 Lot 수·기간과 전후 평균
        for key, label in (("before", "변경 전"), ("after", "변경 후")):
            lines.append(f"- {label} COA: {ba[key]['lots']} Lot ({_day(ba[key]['period'][0])} ~ {_day(ba[key]['period'][1])})")
        for name, v in ba["items"].items():
            pct = "변화율 계산 불가" if v["diff_pct"] is None else f"{v['diff_pct']:+.1f}%"
            lines.append(f"- {name} 변경 전·후 평균: {v['before_mean']:.3f} → {v['after_mean']:.3f} ({v['diff']:+.3f}, {pct})")
    for name, info in result["items"].items():
        spec = f"규격 {info['spec'][0]:.2f} ~ {info['spec'][1]:.2f}"
        if info["detected"]:
            c = info["change"]
            lines.append(f"- {name}: Lot {c['lot']}({_day(c['date'])})부터 평균 {c['before_mean']:.3f} → "
                         f"{c['after_mean']:.3f} ({c['shift']:+.3f}), {spec}")
        else:
            t = info["trend"]
            lines.append(f"- {name}: 뚜렷한 평균 변화 없음 (초기 {t['window']} Lot 평균 {t['first_mean']:.3f} → "
                         f"최근 {t['window']} Lot 평균 {t['recent_mean']:.3f}), {spec}")
        if info["oos_count"]:
            more = " 등" if info["oos_count"] > 10 else ""
            lines.append(f"- {name} 규격 부적합 Lot: {', '.join(info['oos_lots'][:10])}{more}")
    for ev in result["method_changes"]:
        lines.append(f"- 시험방법: {ev['from']} → {ev['to']} (Lot {ev['lot']}부터, {_day(ev['date'])})")
    if not result["has_method"]:
        lines.append(f"- 시험방법: {result['method_notice']}")
    for e in result["events"]:
        four_m = e["four_m"]
        for r in four_m["nearby"]:  # 비교 범위 안의 기록: 기존 표시 그대로
            lines.append(f"- {e['title']}의 변화 시점 전후 {window}일 내 4M 기록: {_day(r['date'])} {r['type']}({r['description']}), "
                         f"{analyzer.days_text(r['days'])}")
        if not four_m["nearby"] and four_m["nearest"]:  # 범위 안에 없을 때만, 범위 밖 기록임을 명시
            r = four_m["nearest"]
            lines.append(f"- {e['title']}의 가장 가까운 4M 기록: {_day(r['date'])} {r['type']} · {r['description']} "
                         f"({analyzer.days_text(r['days'])}, 비교 범위 밖)")

    lines += ["", "#### 3. 확인이 필요한 이유"]
    reasons = [f"- 규격을 벗어난 Lot이 {result['oos_lots']}개 있습니다."] if result["oos_lots"] else []
    reasons += [f"- {e['title']} (Lot {e['lot']}부터): {e['four_m_message']}" for e in result["events"]]
    lines += reasons or ["- 현재 추가 확인이 필요한 변화는 감지되지 않았습니다."]

    lines += ["", "#### 4. 권장 확인사항"]
    for name, info in result["items"].items():
        if info["detected"]:
            lines.append(f"- 공급사에 {name} 변화 시점(Lot {info['change']['lot']}) 전후의 원료·공정·설비 변경 여부와 원인을 확인해 주세요.")
    for ev in result["method_changes"]:
        lines.append(f"- 시험방법 변경({ev['from']} → {ev['to']})의 사유와 이전 방법과의 비교(동등성) 자료를 공급사에 요청해 주세요.")
    if result["oos_lots"]:
        lines.append("- 규격 부적합 Lot의 재시험 여부와 사용 가능 여부를 검토해 주세요.")
    if ba and ba["duplicate_lots"]:
        lines.append(f"- 변경 전·후 파일에 모두 있는 Lot({', '.join(ba['duplicate_lots'][:10])})이 중복 포함되었는지 확인해 주세요.")
    if ba and ba["dates_reversed"]:
        lines.append("- 변경 후 데이터의 일부 날짜가 변경 전 데이터보다 빠르므로, 입력한 전후 구분을 확인해 주세요.")
    lines.append("- 이후 입고되는 Lot도 같은 기준으로 계속 모니터링해 주세요.")
    return "\n".join(lines).replace("~", r"\~")  # '~'가 취소선으로 보이지 않게 처리


def generate_report(result, api_key):
    """AI 리포트를 만듭니다. 어떤 경우에도 오류로 멈추지 않고 리포트를 돌려줍니다.
    반환값: (리포트 텍스트, 안내 메시지)
      - Gemini 성공: (AI 리포트, None)
      - Key 없음 · 호출 실패 · 무료 한도 초과: (기본 요약 리포트, 실패 사유)"""
    if not api_key:
        return build_template_report(result), "Gemini API Key가 설정되지 않았습니다."
    try:
        summary = json.dumps(build_summary(result), ensure_ascii=False, indent=2)
        # '~'가 한 줄에 두 번 나오면 화면에서 취소선으로 보이므로 일반 글자로 표시되게 바꿉니다.
        return _ask_gemini(summary, api_key).replace("~", r"\~"), None
    except Exception as error:
        # 터미널에는 오류 종류와 코드만 남깁니다. (API Key와 원래 오류 문구는 기록하지 않음)
        print(f"[LotWatch] Gemini 호출 실패: {type(error).__name__} {getattr(error, 'code', '') or ''}")
        return build_template_report(result), _error_message(error)
