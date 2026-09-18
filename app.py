"""
LotWatch AI - Streamlit 대시보드 (app.py)

실행 방법:  python -m streamlit run app.py

화면 순서: ① 데이터 선택 → ② 규격 입력 → 분석 결과
          (KPI → 품질 특성 추세 그래프 → 시험방법 변화 → 4M 비교 → AI 품질 분석 → Lot별 상세 데이터)
"""

import traceback
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ai_report
import analyzer

BASE_DIR = Path(__file__).parent
SAMPLE_COA = BASE_DIR / "sample_data" / "coa_sample.csv"
SAMPLE_CHANGES = BASE_DIR / "sample_data" / "changes_sample.csv"

# 상태 표시: 색상은 상태를 구분할 때만 쓰고, 항상 아이콘 + 글자와 함께 보여줍니다.
STATUS_ICON = {analyzer.OK: "🟢", analyzer.WATCH: "🟡", analyzer.CHECK: "🟠", analyzer.OOS: "🔴"}
STATUS_COLOR = {analyzer.OK: "green", analyzer.WATCH: "yellow", analyzer.CHECK: "orange", analyzer.OOS: "red"}
STATUS_BOX = {analyzer.OK: st.success, analyzer.WATCH: st.warning, analyzer.CHECK: st.warning, analyzer.OOS: st.error}

# 그래프 색상
DATA_COLOR = "#2a78d6"  # 측정값
SPEC_COLOR = "#d03b3b"  # 규격 한계선, 규격 부적합 Lot
CHANGE_COLOR = "#ec835a"  # 변화 시점, 변화 후 평균
GUIDE_COLOR = "#898781"  # 변화 전 평균, 4M 변경
METHOD_COLOR = "#52514e"  # 시험방법 변경

# 규격 입력 기본값: (최소값, 최대값, 입력 간격)
DEFAULT_SPECS = {"Moisture": (0.0, 0.50, 0.01), "Purity": (99.0, 100.0, 0.1)}

st.set_page_config(page_title="LotWatch AI", page_icon="🔍", layout="wide")

for key in ("result", "source", "ai_text", "ai_error"):
    st.session_state.setdefault(key, None)


# ── 작은 도우미 함수 ───────────────────────────────────────
def fmt(value):
    return f"{value:.3f}"


def fmt_pct(value):
    """변화율 표시 (예: +87.1%). 계산할 수 없으면 빈 문자열"""
    if value is None:
        return ""
    return "0%" if abs(value) < 0.05 else f"{value:+.1f}%"


def fmt_date(date):
    return date.strftime("%Y-%m-%d")


def fmt_spec(spec):
    return f"{spec[0]:.2f} ~ {spec[1]:.2f}"


def badge(status):
    """상태 배지 (예: 🟠 확인 필요)"""
    return f":{STATUS_COLOR[status]}-badge[{STATUS_ICON[status]} {status}]"


def four_m_record_text(r):
    return f"{fmt_date(r['date'])} {r['type']} · {r['description']} ({analyzer.days_text(r['days'])})"


# ── 데이터 불러오기와 분석 실행 ─────────────────────────────
def map_coa_columns(coa_file, raw):
    """COA 컬럼 자동 인식 결과를 보여주고, 인식하지 못한 컬럼은 사용자가 직접 고르게 합니다. {표준: 원본}을 돌려줌"""
    mapping = analyzer.guess_coa_mapping(raw.columns)
    missing = [c for c in analyzer.COA_COLUMNS if c not in mapping]
    if missing:
        notice = st.empty()  # 선택 결과에 따라 안내 문구를 바꾸기 위한 자리
        st.caption("아래에서 각 항목에 해당하는 원본 컬럼을 직접 선택해 주세요.")
        for col in missing:
            choice = st.selectbox(f"{col} 컬럼으로 사용할 원본 컬럼", list(raw.columns), index=None,
                                  placeholder="원본 컬럼 선택", key=f"map_{col}_{coa_file.file_id}")
            if choice is not None:
                mapping[col] = choice
        still_missing = [c for c in missing if c not in mapping]
        if still_missing:
            notice.error(f"다음 필수 컬럼을 인식하지 못했습니다: {', '.join(still_missing)}")
        else:
            notice.info(f"자동으로 인식하지 못한 컬럼({', '.join(missing)})을 직접 선택한 원본 컬럼으로 연결했습니다.")
    st.markdown("**COA 컬럼을 다음과 같이 인식했습니다.**")
    st.table(pd.DataFrame({"원본 컬럼": [str(mapping.get(c, "(인식하지 못함)")) for c in analyzer.COA_COLUMNS],
                           "LotWatch 컬럼": analyzer.COA_COLUMNS}), hide_index=True)
    return mapping


def read_uploaded_files(coa_file, changes_file):
    """업로드한 파일을 바로 검사해서 결과를 알려줍니다. 문제가 없으면 DataFrame을 돌려줍니다."""
    coa_df = changes_df = None
    if coa_file is not None:
        try:
            raw = analyzer.read_table(coa_file)
            mapping = map_coa_columns(coa_file, raw)
            if len(mapping) == len(analyzer.COA_COLUMNS):  # 필수 컬럼을 모두 연결했을 때만 값 검증
                coa_df = analyzer.prepare_coa(raw, mapping)
                suppliers = sorted(s for s in coa_df["Supplier"].unique() if s)
                if len(suppliers) > 1:
                    chosen = st.selectbox(
                        "분석할 공급사 선택",
                        suppliers,
                        help="여러 공급사의 데이터가 섞여 있으면 변화를 정확히 볼 수 없어 공급사별로 분석합니다.",
                    )
                    coa_df = coa_df[coa_df["Supplier"] == chosen].reset_index(drop=True)
                st.success(f"COA 데이터 {len(coa_df)} Lot을 확인했습니다.")
        except analyzer.DataError as e:
            st.error(str(e))
            coa_df = None
    if changes_file is not None:
        try:
            changes_df = analyzer.load_changes_csv(changes_file)
            st.success(f"4M 변경 이력 {len(changes_df)}건을 확인했습니다.")
        except analyzer.DataError as e:
            st.error(str(e))
            changes_df = None
    return coa_df, changes_df


def analyze(coa_df, changes_df, source, specs, window_days):
    """분석을 실행하고 결과를 session_state에 저장합니다. (버튼을 다시 눌러도 결과가 유지되도록)"""
    for item, (low, high) in specs.items():
        if low >= high:
            st.error(f"{item} 규격의 최소값은 최대값보다 작아야 합니다.")
            return
    try:
        result = analyzer.run_analysis(coa_df, specs, changes_df, window_days)
    except analyzer.DataError as e:
        st.error(str(e))
        return
    except Exception:
        traceback.print_exc()  # 자세한 오류 내용은 터미널에만 출력합니다.
        st.error("분석 중 예상하지 못한 문제가 발생했습니다. CSV 내용을 확인한 뒤 다시 시도해주세요.")
        return
    st.session_state.result = result
    st.session_state.source = source
    st.session_state.ai_text = st.session_state.ai_error = None
    st.toast("분석이 완료되었습니다. 아래에서 결과를 확인하세요.")


# ── 그래프 ────────────────────────────────────────────────
def add_event_line(fig, x, dates, label, color, dash, width, position):
    """세로선 + 짧은 설명을 그립니다. 글자가 겹치지 않도록 위치를 나눕니다.
    position: "above"(그래프 위, 변화 시점) / "top"(그래프 안쪽 위, 시험방법 변경) / "bottom"(그래프 안쪽 아래, 4M 변경)"""
    fig.add_vline(x=x, line=dict(color=color, width=width, dash=dash))
    span = dates.iloc[-1] - dates.iloc[0]
    near_right_edge = span.days > 0 and (x - dates.iloc[0]) / span > 0.75  # 오른쪽 끝이면 글자를 선 왼쪽에
    fig.add_annotation(
        x=x, y=0 if position == "bottom" else 1, yref="paper",
        yanchor="top" if position == "top" else "bottom",
        text=label, showarrow=False, font=dict(size=12 if position == "above" else 11),
        xanchor="right" if near_right_edge else "left", xshift=-4 if near_right_edge else 4,
    )


def make_trend_chart(result, item):
    """품질 특성의 시간별 추세 그래프: 측정값, 규격 상/하한, 변화점, 구간 평균, 시험방법 변경, 4M 변경"""
    data = result["data"]
    info = result["items"][item]
    low, high = info["spec"]
    dates = data["Date"]
    fig = go.Figure()

    # 1) 측정값 (규격 부적합 Lot은 빨간 점)
    is_oos = data[f"{item}_판정"] == "OUT OF SPEC"
    fig.add_trace(go.Scatter(
        x=dates, y=data[item], mode="lines+markers", name=item,
        line=dict(color=DATA_COLOR, width=2),
        marker=dict(size=8, color=[SPEC_COLOR if oos else DATA_COLOR for oos in is_oos],
                    line=dict(color="white", width=1.5)),
        customdata=data[["Lot", "Test_Method", f"{item}_판정"]].to_numpy(),
        hovertemplate=(
            "<b>Lot %{customdata[0]}</b> · %{x|%Y-%m-%d}<br>"
            + item + ": %{y}<br>시험방법: %{customdata[1]}<br>판정: %{customdata[2]}<extra></extra>"
        ),
    ))

    # 2) 규격 상한 / 하한
    fig.add_hline(y=high, line=dict(color=SPEC_COLOR, width=1.5, dash="dash"),
                  annotation_text=f"규격 상한 {high:g}", annotation_position="top left")
    fig.add_hline(y=low, line=dict(color=SPEC_COLOR, width=1.5, dash="dash"),
                  annotation_text=f"규격 하한 {low:g}", annotation_position="bottom left")

    # 3) 변화 시점(두 Lot 사이) + 변화 전/후 평균선
    if info["detected"]:
        c = info["change"]
        x_change = c["prev_date"] + (c["date"] - c["prev_date"]) / 2
        add_event_line(fig, x_change, dates, f"변화 시점 · Lot {c['lot']}", CHANGE_COLOR, "solid", 2, "above")
        fig.add_shape(type="line", x0=dates.iloc[0], x1=c["prev_date"], y0=c["before_mean"], y1=c["before_mean"],
                      line=dict(color=GUIDE_COLOR, width=2))
        fig.add_shape(type="line", x0=c["date"], x1=dates.iloc[-1], y0=c["after_mean"], y1=c["after_mean"],
                      line=dict(color=CHANGE_COLOR, width=2))

    # 4) 시험방법 변경
    for ev in result["method_changes"]:
        x_method = dates.iloc[ev["index"] - 1] + (ev["date"] - dates.iloc[ev["index"] - 1]) / 2
        add_event_line(fig, x_method, dates, f"시험방법 {ev['from']}→{ev['to']}", METHOD_COLOR, "dot", 1.5, "top")

    # 5) 4M 변경 (분석 기간 안에 있는 것만)
    if result["changes_4m"] is not None:
        for row in result["changes_4m"].itertuples():
            if dates.iloc[0] <= row.Date <= dates.iloc[-1]:
                add_event_line(fig, row.Date, dates, f"4M · {row.Type}", GUIDE_COLOR, "dot", 1, "bottom")

    # y축은 측정값과 규격 한계가 모두 보이도록
    y_min = min(data[item].min(), low)
    y_max = max(data[item].max(), high)
    pad = (y_max - y_min) * 0.12 or 1.0
    fig.update_layout(
        height=340,
        margin=dict(l=8, r=8, t=30, b=8),
        showlegend=False,
        hovermode="x",
        xaxis=dict(showgrid=False, showspikes=True, spikemode="across", spikethickness=1,
                   spikecolor="#c3c2b7", spikedash="solid"),
        yaxis=dict(title=dict(text=item), range=[y_min - pad, y_max + pad], zeroline=False),
    )
    return fig


# ── 결과 화면 구성 ─────────────────────────────────────────
def show_item_detail(info):
    """품질 특성 1개의 변화 탐지 결과 (그래프 오른쪽 설명)"""
    st.markdown(f"**{info['message']}**")
    c = info["change"]
    lines = []
    if info["detected"]:
        col1, col2 = st.columns(2)
        col1.metric("변화 전 평균", fmt(c["before_mean"]), help=f"Lot {c['before_lots'][0]} ~ {c['before_lots'][1]}")
        delta = f"{c['shift']:+.3f}" + (f" ({fmt_pct(c['shift_pct'])})" if c["shift_pct"] is not None else "")
        col2.metric("변화 후 평균", fmt(c["after_mean"]), delta=delta, delta_color="off",
                    help=f"Lot {c['after_lots'][0]} ~ {c['after_lots'][1]}")
        effect = f"{c['effect']:.1f}배" if c["effect"] != float("inf") else "매우 큼"
        p_text = "< 0.001" if c["p_value"] < 0.001 else f"= {c['p_value']:.3f}"
        four_m = info["four_m"]
        if not four_m["has_data"]:
            four_m_text = "4M 변경 이력 데이터 없음"
        elif four_m["nearby"]:
            four_m_text = "근접한 4M 변경 이력 있음 (" + ", ".join(f"{fmt_date(r['date'])} {r['type']}" for r in four_m["nearby"]) + ")"
        else:
            four_m_text = "변화 시점 전후 등록된 4M 변경 이력 없음"
        lines += [
            f"- **변화 시점:** Lot {c['lot']} ({fmt_date(c['date'])})",
            f"- **변화 크기:** 평소 변동폭의 {effect} (p {p_text})",
        ]
    lines += [f"- **규격:** {fmt_spec(info['spec'])}", f"- **상태:** {info['spec_state']}"]
    if info["oos_count"]:
        shown = ", ".join(info["oos_lots"][:10]) + (" 등" if info["oos_count"] > 10 else "")
        lines.append(f"- **규격 부적합 Lot:** {shown}")
    if info["detected"]:
        lines.append(f"- **4M 비교:** {four_m_text}")
    st.markdown("\n".join(lines))

    t = info["trend"]
    pct = f", {fmt_pct(t['diff_pct'])}" if t["diff_pct"] is not None else ""
    st.caption(
        f"추세 비교 · 초기 {t['window']} Lot 평균 {fmt(t['first_mean'])} → 최근 {t['window']} Lot 평균 "
        f"{fmt(t['recent_mean'])} ({t['diff']:+.3f}{pct}) · 전체 평균 {fmt(t['overall_mean'])}"
    )


def show_results(result):
    start, end = result["period"]
    st.subheader("분석 결과")
    st.caption(
        f"분석 대상: {st.session_state.source} · 공급사: {', '.join(result['suppliers']) or '미기재'} · "
        f"기간: {fmt_date(start)} ~ {fmt_date(end)} · 4M 비교 기간: 변화 시점 전후 {result['window_days']}일"
    )
    for notice in result["notices"]:
        st.warning(notice)

    # KPI: 왼쪽은 규격 판정, 오른쪽은 변화 감지 (휴대폰에서는 두 줄로 나뉨)
    spec_kpi, change_kpi = st.columns([2, 3])
    with spec_kpi:
        with st.container(horizontal=True):
            st.metric("총 Lot", f"{result['total_lots']}개", border=True)
            st.metric("규격 부적합", f"{result['oos_lots']}개", border=True, help="규격(최소~최대)을 벗어난 Lot 수")
    with change_kpi:
        with st.container(horizontal=True):
            st.metric("품질 변화 감지", f"{result['quality_changes']}건", border=True,
                      help="평균이 통계적으로 달라진 품질 특성 수 (Moisture, Purity 각각 최대 1건)")
            st.metric("시험방법 변경", f"{result['method_change_count']}건", border=True,
                      help="Test_Method 값이 바뀐 횟수 (예: A → B)")
            st.metric("확인 권고", f"{result['check_recommended']}건", border=True,
                      help="품질 변화·시험방법 변경 중, 변화 시점 전후로 등록된 4M 변경 이력이 확인되지 않은 건수")

    status = result["overall_status"]
    STATUS_BOX[status](f"**{STATUS_ICON[status]} {status}** — {result['overall_message']}")
    st.caption("상태 기준: 🟢 정상 · 🟡 주의 (변화 감지, 근접한 4M 변경 이력 있음) · "
               "🟠 확인 필요 (변화 감지, 등록된 4M 변경 이력 없음) · 🔴 규격 부적합")

    # 품질 특성별 그래프 + 변화 탐지 결과
    st.markdown("#### 품질 특성 추세와 변화 탐지")
    st.caption("그래프 보는 법: 파란 선 = 측정값 · 빨간 파선 = 규격 상/하한 · 주황 세로선 = 변화 시점 · "
               "가로 실선 = 구간 평균(회색: 변화 전, 주황: 변화 후) · 점선 = 시험방법 변경 / 4M 변경")
    for item in analyzer.QUALITY_ITEMS:
        info = result["items"][item]
        with st.container(border=True):
            st.markdown(f"**{item}** &nbsp; {badge(info['status'])}")
            chart_col, detail_col = st.columns([3, 2], gap="medium")
            with chart_col:
                st.plotly_chart(make_trend_chart(result, item), width="stretch",
                                config={"displayModeBar": False}, key=f"chart_{item}")
            with detail_col:
                show_item_detail(info)

    # 시험방법 변화
    st.markdown("#### 시험방법 변화")
    if not result["method_changes"]:
        st.success("🟢 분석 기간 동안 시험방법(Test_Method) 변경이 감지되지 않았습니다.")
    for ev in result["method_changes"]:
        with st.container(border=True):
            st.markdown(f"{badge(ev['status'])} &nbsp; **시험방법 변경 감지: {ev['from']} → {ev['to']}, "
                        f"Lot {ev['lot']}부터** ({fmt_date(ev['date'])})")
            st.caption("시험방법이 바뀌면 같은 원료라도 측정값이 달라질 수 있습니다. "
                       "공급사에 변경 사유와 이전 방법과의 비교(동등성) 자료를 확인해 주세요.")

    # 4M 변경 이력 비교
    st.markdown("#### 4M 변경 이력 비교")
    changes = result["changes_4m"]
    if changes is None:
        st.info("4M 변경 이력 파일이 없어 비교하지 않았습니다. 변화가 감지되었다면 공급사에 해당 기간의 변경 여부를 확인해 주세요.")
    if not result["events"]:
        st.success("🟢 감지된 변화가 없어 4M 변경 이력과 비교할 항목이 없습니다.")
    for e in result["events"]:
        four_m = e["four_m"]
        icon = STATUS_ICON[analyzer.WATCH if four_m["nearby"] else analyzer.CHECK]
        lines = [f"- {icon} **{e['title']}** (Lot {e['lot']}, {fmt_date(e['date'])}) — {e['four_m_message']}"]
        lines += [f"  - 근접한 4M 변경: {four_m_record_text(r)}" for r in four_m["nearby"]]
        if four_m["has_data"] and not four_m["nearby"] and four_m["nearest"]:
            lines.append(f"  - 가장 가까운 4M 기록: {four_m_record_text(four_m['nearest'])}")
        st.markdown("\n".join(lines))

    if changes is not None and (len(changes) or result["events"]):
        rows = [{"날짜": e["date"], "구분": e["kind"], "내용": f"{e['title']} (Lot {e['lot']}부터)",
                 "4M 비교": "근접한 4M 이력 있음" if e["four_m"]["nearby"] else "기간 내 등록된 4M 이력 없음"}
                for e in result["events"]]
        rows += [{"날짜": r.Date, "구분": f"4M 변경 · {r.Type}", "내용": r.Description, "4M 비교": ""}
                 for r in changes.itertuples()]
        timeline = pd.DataFrame(rows).sort_values("날짜", kind="stable")
        timeline["날짜"] = timeline["날짜"].dt.strftime("%Y-%m-%d")
        st.caption("변화와 4M 변경 이력을 날짜순으로 나열한 타임라인")
        st.dataframe(timeline, hide_index=True, width="stretch")

    # AI 품질 분석 (Gemini를 쓸 수 없으면 기본 요약 리포트로 자동 대체)
    st.markdown("#### AI 품질 분석")
    api_key = ai_report.get_api_key()
    if api_key:
        st.caption("위 통계 분석 결과를 Gemini AI가 QC 담당자용 리포트로 정리합니다.")
    else:
        st.info("AI 분석을 사용할 수 없습니다. AI 분석을 사용하려면 Gemini API Key를 설정하세요. "
                "(설정 방법: README.md의 'AI API 설정 방법')\n\n"
                "Key가 없어도 아래 버튼을 누르면 통계 분석 결과로 만든 기본 요약 리포트를 볼 수 있습니다.")
    if st.button("AI 분석 리포트 생성", icon=":material/auto_awesome:"):
        with st.spinner("AI가 분석 결과를 정리하고 있습니다... (보통 5~30초)"):
            st.session_state.ai_text, st.session_state.ai_error = ai_report.generate_report(result, api_key)
    if st.session_state.ai_error:
        st.warning(st.session_state.ai_error)
    if st.session_state.ai_text:
        with st.container(border=True):
            st.markdown(st.session_state.ai_text)
            st.caption(f"※ {ai_report.DISCLAIMER}")

    # Lot별 상세 데이터
    with st.expander("Lot별 상세 데이터 (규격 판정 결과)"):
        table = result["data"].copy()
        table["Date"] = table["Date"].dt.strftime("%Y-%m-%d")
        judge_cols = [f"{item}_판정" for item in analyzer.QUALITY_ITEMS] + ["종합판정"]
        styled = table.style.map(
            lambda v: "background-color: rgba(208, 59, 59, 0.15)" if v == "OUT OF SPEC" else "", subset=judge_cols
        ).format({item: "{:g}" for item in analyzer.QUALITY_ITEMS})  # 0.140000 → 0.14
        st.dataframe(styled, hide_index=True, width="stretch")
        st.download_button("분석 결과 CSV 다운로드", table.to_csv(index=False).encode("utf-8-sig"),
                           file_name="lotwatch_result.csv", mime="text/csv", on_click="ignore")


# ── 화면 시작 ─────────────────────────────────────────────
st.title("LotWatch AI")
st.markdown("**Supplier Quality Change Monitoring**")
st.markdown("### 규격 이내의 품질 변화까지 찾아내는 AI 품질 모니터링")
st.markdown("COA 및 수입검사 데이터를 시간순으로 분석하여 규격 이탈 전 나타나는 품질 변화를 찾아내고, "
            "확인이 필요한 Lot을 알려줍니다.")
st.caption("① 데이터 선택 → ② 규격 입력 → 분석 결과: 품질 변화 감지 · 추세 그래프 · 시험방법 변경 · 4M 비교 · AI 리포트")

with st.container(border=True):
    data_col, spec_col = st.columns([1.1, 1], gap="large")

    with data_col:
        st.markdown("##### ① 데이터 선택")
        st.markdown("**A. 샘플 데이터로 체험**")
        st.caption("A사 원료 40 Lot의 COA 예시 데이터와 4M 변경 이력 2건입니다. "
                   "모든 Lot이 규격을 통과하지만, 최근 데이터에 변화가 숨어 있습니다.")
        sample_clicked = st.button("샘플 데이터로 분석하기", type="primary", icon=":material/play_arrow:")

        st.markdown("**B. 내 파일 업로드 (CSV · Excel)**")
        coa_file = st.file_uploader("COA 파일 (CSV 또는 Excel, 필수)", type=["csv", "xlsx"],
                                    help="컬럼명이 달라도 자동으로 인식합니다. 예: Batch No., 검사일, 공급업체, 수분(%), 순도(%), 시험방법")
        changes_file = st.file_uploader("4M 변경 이력 CSV (선택)", type="csv",
                                        help="필수 컬럼: Date, Type, Description")
        coa_df, changes_df = read_uploaded_files(coa_file, changes_file)
        confirm_clicked = coa_file is not None and st.button(
            "이대로 분석하기", type="primary", icon=":material/check:", disabled=coa_df is None)

        with st.expander("CSV 형식 안내 · 예시 파일 받기"):
            st.markdown("**COA CSV** — 필수 컬럼: `Lot`, `Date`, `Supplier`, `Moisture`, `Purity`, `Test_Method`")
            st.code("Lot,Date,Supplier,Moisture,Purity,Test_Method\n"
                    "001,2026-01-05,A사,0.12,99.5,A\n002,2026-01-12,A사,0.13,99.5,A", language=None)
            st.markdown("**4M 변경 이력 CSV (선택)** — 필수 컬럼: `Date`, `Type`, `Description`")
            st.code("Date,Type,Description\n2026-05-01,Equipment,생산설비 변경\n"
                    "2026-06-15,Material,원료 공급처 변경", language=None)
            st.caption("날짜는 2026-01-05 형식을 권장합니다. 엑셀에서 저장한 한글 CSV도 읽을 수 있습니다. "
                       "Excel(.xlsx)은 첫 번째 시트를 읽고, 컬럼명이 달라도(예: Batch No., 검사일, 수분(%)) 자동으로 인식합니다.")
            d1, d2 = st.columns(2)
            d1.download_button("COA 예시 파일 받기", SAMPLE_COA.read_bytes(), file_name="coa_sample.csv",
                               mime="text/csv", on_click="ignore", width="stretch")
            d2.download_button("4M 예시 파일 받기", SAMPLE_CHANGES.read_bytes(), file_name="changes_sample.csv",
                               mime="text/csv", on_click="ignore", width="stretch")

    with spec_col:
        st.markdown("##### ② 규격 입력")
        specs = {}
        for item, (low, high, step) in DEFAULT_SPECS.items():
            c1, c2 = st.columns(2)
            spec_low = c1.number_input(f"{item} 최소값", value=low, step=step, format="%.2f")
            spec_high = c2.number_input(f"{item} 최대값", value=high, step=step, format="%.2f")
            specs[item] = (spec_low, spec_high)
        window_days = st.slider(
            "4M 비교 기간: 변화 시점 전후 (일)", min_value=7, max_value=90, value=30,
            help="변화 시점 앞뒤로 이 기간 안에 4M 변경 이력이 있으면 '근접한 4M 변경 이력이 있다'고 판단합니다.",
        )
        start_clicked = st.button("분석 시작", type="primary", icon=":material/analytics:", width="stretch")
        st.caption("COA 파일을 업로드하지 않고 누르면 샘플 데이터로 분석합니다.")

# 버튼을 눌렀을 때 분석 실행
if sample_clicked or (start_clicked and coa_file is None):
    try:
        sample_coa = analyzer.load_coa_csv(SAMPLE_COA)
        sample_changes = analyzer.load_changes_csv(SAMPLE_CHANGES)
    except (FileNotFoundError, analyzer.DataError):
        st.error("샘플 데이터 파일을 읽을 수 없습니다. sample_data 폴더에 CSV 파일이 있는지 확인해주세요.")
    else:
        analyze(sample_coa, sample_changes, "샘플 데이터 (coa_sample.csv, changes_sample.csv)", specs, window_days)
elif start_clicked or confirm_clicked:
    if coa_df is None:
        st.error("업로드한 COA 파일에 문제가 있어 분석할 수 없습니다. 위 안내에 따라 파일을 수정하거나 컬럼을 선택해주세요.")
    elif changes_file is not None and changes_df is None:
        st.error("업로드한 4M 변경 이력 CSV에 문제가 있습니다. 파일을 수정하거나 업로드를 취소해주세요.")
    else:
        source = f"업로드 파일 ({coa_file.name}" + (f", {changes_file.name})" if changes_file else ")")
        analyze(coa_df, changes_df, source, specs, window_days)

if st.session_state.result is None:
    st.info("👆 **샘플 데이터로 분석하기**를 누르거나, COA 파일을 업로드한 뒤 **이대로 분석하기**를 눌러주세요.")
else:
    show_results(st.session_state.result)

st.divider()
st.caption(f"LotWatch AI · {ai_report.DISCLAIMER}")
