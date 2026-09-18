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
- 변화 시점과 4M 변경 이력의 날짜 차이를 근거로 시간적 관계를 설명하세요.
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


def _four_m_relation(four_m, window_days):
    """변화 시점과 4M 변경 이력의 시간적 관계를 정리합니다."""
    if not four_m["has_data"]:
        return "4M 변경 이력 데이터가 제공되지 않음"

    def describe(r):
        return f"{_day(r['date'])} {r['type']} ({r['description']}), {analyzer.days_text(r['days'])}"

    relation = {
        "근접_판단_기준": f"변화 시점 전후 {window_days}일 이내",
        "근접한_4M_변경": [describe(r) for r in four_m["nearby"]] or "해당 기간 내 등록된 이력 없음",
    }
    if four_m["nearest"]:
        relation["가장_가까운_4M_변경"] = describe(four_m["nearest"])
    return relation


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
            entry["4M_비교"] = _four_m_relation(info["four_m"], window)
        items[name] = entry

    changes = result["changes_4m"]
    start, end = result["period"]
    return {
        "공급사": ", ".join(result["suppliers"]) or "미기재",
        "분석_기간": f"{_day(start)} ~ {_day(end)}",
        "총_Lot": result["total_lots"],
        "규격_부적합_Lot수": result["oos_lots"],
        "품질_변화_감지_건수": result["quality_changes"],
        "시험방법_변경_건수": result["method_change_count"],
        "확인_권고_건수": result["check_recommended"],
        "종합_상태": result["overall_status"],
        "품질특성별_결과": items,
        "시험방법_변경": [
            {
                "변경": f"{e['from']} → {e['to']}",
                "시작_Lot": e["lot"],
                "시작일": _day(e["date"]),
                "4M_비교": _four_m_relation(e["four_m"], window),
            }
            for e in result["method_changes"]
        ] or "변경 없음",
        "4M_변경_이력": [
            {"날짜": _day(row.Date), "유형": row.Type, "내용": row.Description}
            for row in changes.itertuples()
        ] if changes is not None else "제공되지 않음",
        "참고사항": result["notices"],
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
        f"- 총 {result['total_lots']} Lot 중 규격 부적합 {result['oos_lots']} Lot, 품질 변화 {result['quality_changes']}건, "
        f"시험방법 변경 {result['method_change_count']}건이 감지되었습니다.",
        f"- 종합 상태: {result['overall_status']} — {result['overall_message']}",
        "",
        "#### 2. 데이터 근거",
    ]
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
