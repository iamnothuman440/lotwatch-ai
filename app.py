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
SAMPLE_LONG_COA = BASE_DIR / "sample_data" / "coa_long_sample.csv"  # 세로형(Long Format) COA 예시
SAMPLE_BEFORE = BASE_DIR / "sample_data" / "coa_before_sample.csv"  # 변경 전·후 비교 예시: 변경 전 (Wide Format)
SAMPLE_AFTER = BASE_DIR / "sample_data" / "coa_after_sample.csv"  # 변경 후 (Long Format)

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
COA_FORMATS = ["Wide Format (가로형 · 1행 = 1 Lot)", "Long Format (세로형 · 1행 = 1 시험항목)"]
ANALYSIS_MODES = ["단일 COA 분석", "변경 전·후 COA 비교"]
COMPARE_SLOTS = (("before", "변경 전"), ("after", "변경 후"))  # 전후 구분은 사용자가 올린 칸을 그대로 따름

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


def fmt_period(period):
    return f"{fmt_date(period[0])} ~ {fmt_date(period[1])}"


def fmt_spec(spec):
    return f"{spec[0]:.2f} ~ {spec[1]:.2f}"


def badge(status):
    """상태 배지 (예: 🟠 확인 필요)"""
    return f":{STATUS_COLOR[status]}-badge[{STATUS_ICON[status]} {status}]"


def four_m_record_text(r, note=""):
    return f"{fmt_date(r['date'])} {r['type']} · {r['description']} ({analyzer.days_text(r['days'])}{note})"


# ── 데이터 불러오기와 분석 실행 ─────────────────────────────
def map_coa_columns(coa_file, raw, required, optional=(), key=""):
    """COA 컬럼 자동 인식 결과를 보여주고, 인식하지 못한 필수 컬럼은 사용자가 직접 고르게 합니다. {표준: 원본}을 돌려줌"""
    guessed = analyzer.guess_coa_mapping(raw.columns)
    mapping = {c: guessed[c] for c in [*required, *optional] if c in guessed}
    missing = [c for c in required if c not in mapping]
    if missing:
        notice = st.empty()  # 선택 결과에 따라 안내 문구를 바꾸기 위한 자리
        st.caption("아래에서 각 항목에 해당하는 원본 컬럼을 직접 선택해 주세요.")
        for col in missing:
            choice = st.selectbox(f"{col} 컬럼으로 사용할 원본 컬럼", list(raw.columns), index=None,
                                  placeholder="원본 컬럼 선택", key=f"map_{col}_{key}{coa_file.file_id}")
            if choice is not None:
                mapping[col] = choice
        still_missing = [c for c in missing if c not in mapping]
        if still_missing:
            notice.error(f"다음 필수 컬럼을 인식하지 못했습니다: {', '.join(still_missing)}")
        else:
            notice.info(f"자동으로 인식하지 못한 컬럼({', '.join(missing)})을 직접 선택한 원본 컬럼으로 연결했습니다.")
    shown = [*required, *(c for c in optional if c in mapping)]
    st.markdown("**COA 컬럼을 다음과 같이 인식했습니다.**")
    st.table(pd.DataFrame({"원본 컬럼": [str(mapping.get(c, "(인식하지 못함)")) for c in shown],
                           "LotWatch 컬럼": shown}), hide_index=True)
    return mapping


def show_long_preview(coa_df, ignored):
    """세로형 COA를 표준 형식으로 바꾼 결과를 분석 전에 보여줍니다."""
    st.markdown("**입력 형식:** Long Format COA (세로형) · **변환 상태:** 정상")
    st.caption("변환된 데이터 미리보기 (처음 5 Lot)")
    preview = coa_df.head().assign(Date=lambda d: d["Date"].dt.strftime("%Y-%m-%d"),
                                   Test_Method=lambda d: d["Test_Method"].replace("", "미기재"))
    st.dataframe(preview.style.format({item: "{:g}" for item in analyzer.QUALITY_ITEMS}), hide_index=True)
    if ignored:
        st.info(f"Moisture·Purity가 아닌 시험항목은 분석에서 제외했습니다: {', '.join(ignored)}")


def read_coa_file(coa_file, key=""):
    """업로드한 COA 파일 1개를 형식 인식 → 컬럼 매핑 → 검증·변환해서 (표준 표, 파일의 규격(LSL/USL), 형식 이름)을 돌려줍니다.
    key: 변경 전·후 비교처럼 파일을 여러 개 받을 때 위젯이 겹치지 않도록 붙이는 구분자. 문제가 있으면 표는 None"""
    coa_df, file_specs, is_long = None, {}, False
    try:
        raw = analyzer.read_table(coa_file)
        auto_long = analyzer.is_long_format(analyzer.guess_coa_mapping(raw.columns))
        is_long = st.radio(
            "COA 입력 형식", COA_FORMATS, index=int(auto_long), horizontal=True, key=f"format_{key}{coa_file.file_id}",
            help="컬럼을 보고 자동으로 선택했습니다. 잘못 인식했다면 직접 바꿔주세요.",
        ) == COA_FORMATS[1]
        if is_long:
            mapping = map_coa_columns(coa_file, raw, analyzer.LONG_COLUMNS, analyzer.LONG_OPTIONAL, key)
            if all(c in mapping for c in analyzer.LONG_COLUMNS):  # 필수 컬럼을 모두 연결했을 때만 변환·검증
                coa_df, file_specs, ignored = analyzer.prepare_long_coa(raw, mapping)
        else:
            mapping = map_coa_columns(coa_file, raw, analyzer.WIDE_COLUMNS, ["Test_Method"], key)
            if all(c in mapping for c in analyzer.WIDE_COLUMNS):
                coa_df = analyzer.prepare_coa(raw, mapping)
        if coa_df is not None:
            suppliers = sorted(s for s in coa_df["Supplier"].unique() if s)
            if len(suppliers) > 1:
                chosen = st.selectbox(
                    "분석할 공급사 선택",
                    suppliers,
                    help="여러 공급사의 데이터가 섞여 있으면 변화를 정확히 볼 수 없어 공급사별로 분석합니다.",
                    key=f"supplier_{key}{coa_file.file_id}",
                )
                coa_df = coa_df[coa_df["Supplier"] == chosen].reset_index(drop=True)
            if is_long:
                show_long_preview(coa_df, ignored)
            if (coa_df["Test_Method"] == "").all():  # 가로형·세로형 모두: 시험방법 정보 없음 안내
                st.info(analyzer.NO_METHOD_NOTICE)
            if key:  # 변경 전·후 비교: 파일마다 변환 상태·형식·Lot 수·기간을 한 줄로
                st.success(f"변환 상태: 정상 · 입력 형식: {'Long' if is_long else 'Wide'} Format · "
                           f"Lot {len(coa_df)}개 · 기간 {fmt_date(coa_df['Date'].min())} ~ {fmt_date(coa_df['Date'].max())}")
            else:
                st.success(f"COA 데이터 {len(coa_df)} Lot을 확인했습니다.")
    except analyzer.DataError as e:
        st.error(str(e))
        coa_df = None
    return coa_df, file_specs, "Long Format" if is_long else "Wide Format"


def read_changes_file(changes_file):
    """4M 변경 이력 CSV를 바로 검사합니다. 문제가 없으면 DataFrame, 파일이 없거나 문제가 있으면 None"""
    if changes_file is None:
        return None
    try:
        changes_df = analyzer.load_changes_csv(changes_file)
    except analyzer.DataError as e:
        st.error(str(e))
        return None
    st.success(f"4M 변경 이력 {len(changes_df)}건을 확인했습니다.")
    return changes_df


def show_comparison_warnings(c):
    """변경 전·후 데이터를 합칠 때 확인할 점을 경고합니다. (분석은 사용자가 지정한 전후 구분 그대로 진행)"""
    if c["duplicate_lots"]:
        shown = ", ".join(c["duplicate_lots"][:10]) + (" 등" if len(c["duplicate_lots"]) > 10 else "")
        st.warning("변경 전·후 데이터에 동일한 Lot 번호가 존재합니다. 동일 Lot이 중복 포함되었는지 확인해주세요. "
                   f"(중복 Lot: {shown} · 두 행 모두 분석에 포함)")
    if c["dates_reversed"]:
        st.warning("변경 후 데이터의 일부 날짜가 변경 전 데이터보다 빠릅니다. 입력한 전후 구분을 확인해주세요.")


def apply_file_specs(file_specs):
    """[COA 규격을 입력칸에 적용] 버튼: 파일의 LSL/USL을 규격 입력칸에 넣습니다. (값이 없는 쪽은 그대로 둠)"""
    for item, limits in file_specs.items():
        for side, value in zip(("low", "high"), limits):
            if value is not None:
                st.session_state[f"spec_{item}_{side}"] = value


def show_file_specs(file_specs, specs, label="업로드한 COA", key=""):
    """세로형 COA의 LSL/USL을 참고값으로 보여줍니다. 분석에는 항상 사용자가 확인한 입력칸의 규격을 씁니다."""
    limit = lambda v: "–" if v is None else f"{v:.2f}"
    text = " · ".join(f"{item} {limit(lo)} ~ {limit(hi)}" for item, (lo, hi) in file_specs.items()).replace("~", r"\~")
    if all(v is None or v == specs[item][i] for item, limits in file_specs.items() for i, v in enumerate(limits)):
        st.caption(f"✅ 입력한 규격이 {label}의 규격(LSL/USL)과 같습니다: {text}")
        return
    st.warning(f"{label}의 규격(LSL/USL): {text}\n\n입력한 규격과 다릅니다. 분석에는 위 입력칸의 규격이 사용됩니다.")
    st.button("COA 규격을 입력칸에 적용", on_click=apply_file_specs, args=(file_specs,), icon=":material/input:",
              key=f"apply_specs_{key}")


def analyze(coa_df, changes_df, source, specs, window_days, after_df=None, files=None):
    """분석을 실행하고 결과를 session_state에 저장합니다. (버튼을 다시 눌러도 결과가 유지되도록)
    after_df가 있으면 coa_df(변경 전)와 after_df(변경 후)를 합쳐 분석합니다. files: {"before": (파일 이름, 형식), "after": ...}"""
    for item, (low, high) in specs.items():
        if low >= high:
            st.error(f"{item} 규격의 최소값은 최대값보다 작아야 합니다.")
            return
    try:
        if after_df is None:
            result = analyzer.run_analysis(coa_df, specs, changes_df, window_days)
        else:
            result = analyzer.run_before_after(coa_df, after_df, specs, changes_df, window_days)
            result["comparison"]["files"] = files
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

    # 6) 변경 전·후 비교: 변경 후 COA 구간을 옅은 회색으로 칠해 전환 지점을 표시 (날짜가 겹치면 전환 지점이 없어 생략)
    c = result.get("comparison")
    if c and not c["dates_reversed"]:
        before_end, after_start = c["before"]["period"][1], c["after"]["period"][0]
        x = before_end + (after_start - before_end) / 2
        fig.add_vrect(x0=x, x1=dates.iloc[-1], fillcolor=GUIDE_COLOR, opacity=0.12, line_width=0, layer="below")
        for text, anchor, shift in (("← 변경 전", "right", -4), ("변경 후 →", "left", 4)):
            fig.add_annotation(x=x, y=0.86, yref="paper", text=text, showarrow=False, xanchor=anchor, xshift=shift,
                               font=dict(size=11, color=METHOD_COLOR))

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
def show_comparison(c):
    """변경 전·후 비교 모드: 분석에 쓴 두 파일과 전후 평균을 결과 맨 위에 보여줍니다."""
    st.markdown("#### 분석 데이터")
    rows = [{"구분": label, "파일": c["files"][key][0], "형식": c["files"][key][1],
             "Lot 수": f"{c[key]['lots']}개", "기간": fmt_period(c[key]["period"])} for key, label in COMPARE_SLOTS]
    rows.append({"구분": "통합 분석", "파일": "–", "형식": "–", "Lot 수": f"{c['combined']['lots']}개",
                 "기간": fmt_period(c["combined"]["period"])})
    st.table(pd.DataFrame(rows), hide_index=True)
    show_comparison_warnings(c)

    st.markdown("#### 변경 전·후 품질 비교")
    st.table(pd.DataFrame([{"항목": item, "변경 전 평균": fmt(v["before_mean"]), "변경 후 평균": fmt(v["after_mean"]),
                            "변화량": f"{v['diff']:+.3f}", "변화율": fmt_pct(v["diff_pct"]) or "계산 불가 (변경 전 평균 0)"}
                           for item, v in c["items"].items()]), hide_index=True)
    st.caption("변경 전·후 파일의 단순 평균 비교입니다. 통계적으로 의미 있는 변화인지는 아래 '품질 변화 감지'(변화점 분석) 결과를 "
               "기준으로 판단하며, 파일이 나뉜 시점을 변화의 원인으로 판단하지 않습니다.")


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
    comparison = result.get("comparison")  # 변경 전·후 비교 모드일 때만 있음
    if comparison:
        show_comparison(comparison)
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
            st.metric("시험방법 변경", f"{result['method_change_count']}건" if result["has_method"] else "미분석", border=True,
                      help="Test_Method 값이 바뀐 횟수 (예: A → B)" if result["has_method"] else result["method_notice"])
            st.metric("확인 권고", f"{result['check_recommended']}건", border=True,
                      help="품질 변화·시험방법 변경 중, 변화 시점 전후로 등록된 4M 변경 이력이 확인되지 않은 건수")

    status = result["overall_status"]
    STATUS_BOX[status](f"**{STATUS_ICON[status]} {status}** — {result['overall_message']}")
    st.caption("상태 기준: 🟢 정상 · 🟡 주의 (변화 감지, 근접한 4M 변경 이력 있음) · "
               "🟠 확인 필요 (변화 감지, 등록된 4M 변경 이력 없음) · 🔴 규격 부적합")

    # 품질 특성별 그래프 + 변화 탐지 결과
    st.markdown("#### 품질 특성 추세와 변화 탐지")
    band = ""
    if comparison:
        band = (" · 변경 전·후 날짜가 겹쳐 전환 지점은 표시하지 않았습니다" if comparison["dates_reversed"]
                else " · 회색 음영 = 변경 후 COA 구간")
    st.caption("그래프 보는 법: 파란 선 = 측정값 · 빨간 파선 = 규격 상/하한 · 주황 세로선 = 변화 시점 · "
               "가로 실선 = 구간 평균(회색: 변화 전, 주황: 변화 후) · 점선 = 시험방법 변경 / 4M 변경" + band)
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
    if not result["has_method"]:
        st.info(result["method_notice"])
    elif not result["method_changes"]:
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
            # 이 줄은 비교 범위 안에 기록이 없을 때만 나오므로, 항상 범위 밖 기록입니다.
            lines.append(f"  - 가장 가까운 4M 기록: {four_m_record_text(four_m['nearest'], ', 비교 범위 밖')}")
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
        if comparison:  # 변경 전·후 비교: 각 Lot이 어느 파일에서 왔는지 (Before / After)
            table.insert(1, "Change_Period", comparison["periods"])
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
        compare_mode = st.segmented_control(
            "분석 방식", ANALYSIS_MODES, default=ANALYSIS_MODES[0], required=True, key="analysis_mode",
            help="변경 전·후 COA 비교: 공정·원료 등의 변경 전후 COA 2개를 각각 올려, 합친 데이터를 기존 분석기로 분석합니다.",
        ) == ANALYSIS_MODES[1]
        sample_clicked = confirm_clicked = compare_clicked = False
        coa_file = coa_df = None
        file_specs, loaded = [], {}  # file_specs: (표시 이름, 위젯 구분자, 파일의 LSL/USL) 목록
        if compare_mode:
            st.caption("변경 전·후 COA를 각각 올리면 파일마다 형식(Wide/Long)을 인식해 표준 형식으로 바꾼 뒤, 두 데이터를 "
                       "날짜순으로 합쳐 분석합니다. 변경 전·후 구분은 날짜로 추측하지 않고 올린 칸을 그대로 따릅니다.")
            for key, label in COMPARE_SLOTS:
                with st.container(border=True):
                    upload = st.file_uploader(f"{label} COA 파일 (CSV 또는 Excel)", type=["csv", "xlsx"],
                                              key=f"{key}_coa_file")
                    if upload is not None:
                        loaded[key] = (upload, *read_coa_file(upload, key))  # (파일, 표준 표, LSL/USL, 형식 이름)
                        if loaded[key][2]:
                            file_specs.append((f"{label} COA", key, loaded[key][2]))
        else:
            st.markdown("**A. 샘플 데이터로 체험**")
            st.caption("A사 원료 40 Lot의 COA 예시 데이터와 4M 변경 이력 2건입니다. "
                       "모든 Lot이 규격을 통과하지만, 최근 데이터에 변화가 숨어 있습니다.")
            sample_clicked = st.button("샘플 데이터로 분석하기", type="primary", icon=":material/play_arrow:")

            st.markdown("**B. 내 파일 업로드 (CSV · Excel)**")
            coa_file = st.file_uploader("COA 파일 (CSV 또는 Excel, 필수)", type=["csv", "xlsx"],
                                        help="컬럼명이 달라도 자동으로 인식합니다. 예: Batch No., 검사일, 공급업체, 수분(%), 순도(%), 시험방법 · "
                                             "시험항목별로 한 행씩 적힌 세로형 COA(Parameter, MeasuredValue)도 인식합니다.")
        changes_file = st.file_uploader("4M 변경 이력 CSV (선택)", type="csv",
                                        help="필수 컬럼: Date, Type, Description")
        if coa_file is not None:
            coa_df, found_specs, _ = read_coa_file(coa_file)
            if found_specs:
                file_specs.append(("업로드한 COA", "", found_specs))
        changes_df = read_changes_file(changes_file)
        if compare_mode:
            ready = all(loaded.get(key, (None, None))[1] is not None for key, _ in COMPARE_SLOTS)
            if ready:  # 두 파일을 합치기 전에 통합 기간과 확인할 점을 보여줌
                check = analyzer.compare_before_after(loaded["before"][1], loaded["after"][1])
                st.markdown(f"**통합 분석:** Lot {check['combined']['lots']}개 · 기간 {fmt_period(check['combined']['period'])}")
                show_comparison_warnings(check)
                if check["partial_method"]:
                    st.info(analyzer.PARTIAL_METHOD_NOTICE)
            compare_clicked = st.button("변경 전·후 분석하기", type="primary", icon=":material/compare_arrows:",
                                        disabled=not ready)
        else:
            confirm_clicked = coa_file is not None and st.button(
                "이대로 분석하기", type="primary", icon=":material/check:", disabled=coa_df is None)

        with st.expander("CSV 형식 안내 · 예시 파일 받기"):
            st.markdown("**COA CSV** — 필수 컬럼: `Lot`, `Date`, `Supplier`, `Moisture`, `Purity` "
                        "(선택: `Test_Method` — 없으면 시험방법 변경 분석만 건너뜀)")
            st.code("Lot,Date,Supplier,Moisture,Purity,Test_Method\n"
                    "001,2026-01-05,A사,0.12,99.5,A\n002,2026-01-12,A사,0.13,99.5,A", language=None)
            st.markdown("**세로형(Long Format) COA** — 시험항목마다 한 행: `LotNo`, `InspectDate`, `VendorName`, "
                        "`Parameter`, `MeasuredValue` (선택: `LSL`, `USL`, 시험방법)")
            st.code("LotNo,InspectDate,VendorName,Parameter,LSL,USL,MeasuredValue\n"
                    "LOT001,2026-01-01,A사,Moisture,0.00,0.50,0.14\nLOT001,2026-01-01,A사,Purity,99.00,100.00,99.50",
                    language=None)
            st.markdown("**4M 변경 이력 CSV (선택)** — 필수 컬럼: `Date`, `Type`, `Description`")
            st.code("Date,Type,Description\n2026-05-01,Equipment,생산설비 변경\n"
                    "2026-06-15,Material,원료 공급처 변경", language=None)
            st.caption("날짜는 2026-01-05 형식을 권장합니다. 엑셀에서 저장한 한글 CSV도 읽을 수 있습니다. "
                       "Excel(.xlsx)은 첫 번째 시트를 읽고, 컬럼명이 달라도(예: Batch No., 검사일, 수분(%)) 자동으로 인식합니다.")
            d1, d2, d3 = st.columns(3)
            d1.download_button("COA 예시 파일 받기", SAMPLE_COA.read_bytes(), file_name="coa_sample.csv",
                               mime="text/csv", on_click="ignore", width="stretch")
            d2.download_button("세로형 COA 예시 받기", SAMPLE_LONG_COA.read_bytes(), file_name="coa_long_sample.csv",
                               mime="text/csv", on_click="ignore", width="stretch")
            d3.download_button("4M 예시 파일 받기", SAMPLE_CHANGES.read_bytes(), file_name="changes_sample.csv",
                               mime="text/csv", on_click="ignore", width="stretch")
            st.markdown("**변경 전·후 COA 비교** — 변경 전·후 COA를 각각 올립니다. 두 파일의 형식(Wide/Long)과 컬럼명이 "
                        "달라도 됩니다. (아래 예시: 변경 전 Wide, 변경 후 Long · 가상 데이터)")
            d4, d5 = st.columns(2)
            d4.download_button("변경 전 예시 받기 (Wide)", SAMPLE_BEFORE.read_bytes(), file_name=SAMPLE_BEFORE.name,
                               mime="text/csv", on_click="ignore", width="stretch")
            d5.download_button("변경 후 예시 받기 (Long)", SAMPLE_AFTER.read_bytes(), file_name=SAMPLE_AFTER.name,
                               mime="text/csv", on_click="ignore", width="stretch")

    with spec_col:
        st.markdown("##### ② 규격 입력")
        specs = {}
        for item, (low, high, step) in DEFAULT_SPECS.items():
            c1, c2 = st.columns(2)
            # 기본값은 session_state로 넣습니다. ([COA 규격을 입력칸에 적용] 버튼이 같은 칸의 값을 바꿀 수 있도록)
            st.session_state.setdefault(f"spec_{item}_low", low)
            st.session_state.setdefault(f"spec_{item}_high", high)
            spec_low = c1.number_input(f"{item} 최소값", step=step, format="%.2f", key=f"spec_{item}_low")
            spec_high = c2.number_input(f"{item} 최대값", step=step, format="%.2f", key=f"spec_{item}_high")
            specs[item] = (spec_low, spec_high)
        for label, key, found_specs in file_specs:
            show_file_specs(found_specs, specs, label, key)
        window_days = st.slider(
            "4M 비교 기간: 변화 시점 전후 (일)", min_value=7, max_value=90, value=30,
            help="변화 시점 앞뒤로 이 기간 안에 4M 변경 이력이 있으면 '근접한 4M 변경 이력이 있다'고 판단합니다.",
        )
        start_clicked = st.button("분석 시작", type="primary", icon=":material/analytics:", width="stretch")
        st.caption("변경 전·후 COA를 올리지 않고 누르면 예시 데이터(변경 전 Wide · 변경 후 Long)로 분석합니다." if compare_mode
                   else "COA 파일을 업로드하지 않고 누르면 샘플 데이터로 분석합니다.")

# 버튼을 눌렀을 때 분석 실행
if compare_mode:
    if start_clicked and not loaded:  # 파일 없이 [분석 시작] → 변경 전·후 예시 데이터 (단일 모드의 샘플과 같은 방식)
        try:
            sample_before, sample_after = analyzer.load_coa_csv(SAMPLE_BEFORE), analyzer.load_coa_csv(SAMPLE_AFTER)
            sample_changes = analyzer.load_changes_csv(SAMPLE_CHANGES)
        except (FileNotFoundError, analyzer.DataError):
            st.error("예시 데이터 파일을 읽을 수 없습니다. sample_data 폴더에 CSV 파일이 있는지 확인해주세요.")
        else:
            analyze(sample_before, sample_changes, "변경 전·후 예시 데이터 (coa_before_sample.csv → coa_after_sample.csv, "
                    "changes_sample.csv)", specs, window_days, sample_after,
                    {"before": (SAMPLE_BEFORE.name, "Wide Format"), "after": (SAMPLE_AFTER.name, "Long Format")})
    elif start_clicked or compare_clicked:
        if not ready:
            st.error("변경 전·후 COA 파일을 모두 올리고, 위 안내에 따라 문제를 해결해주세요.")
        elif changes_file is not None and changes_df is None:
            st.error("업로드한 4M 변경 이력 CSV에 문제가 있습니다. 파일을 수정하거나 업로드를 취소해주세요.")
        else:
            (before_file, before_df, _, before_fmt), (after_file, after_df, _, after_fmt) = loaded["before"], loaded["after"]
            source = f"변경 전·후 비교 ({before_file.name} → {after_file.name}" + (f", {changes_file.name})" if changes_file else ")")
            analyze(before_df, changes_df, source, specs, window_days, after_df,
                    {"before": (before_file.name, before_fmt), "after": (after_file.name, after_fmt)})
elif sample_clicked or (start_clicked and coa_file is None):
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
    st.info("👆 변경 전·후 COA를 올린 뒤 **변경 전·후 분석하기**를 누르거나, **분석 시작**을 눌러 예시 데이터로 확인해 보세요."
            if compare_mode else "👆 **샘플 데이터로 분석하기**를 누르거나, COA 파일을 업로드한 뒤 **이대로 분석하기**를 눌러주세요.")
else:
    show_results(st.session_state.result)

st.divider()
st.caption(f"LotWatch AI · {ai_report.DISCLAIMER}")
