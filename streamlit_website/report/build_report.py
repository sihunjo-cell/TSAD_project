"""ML 추정과 DP 경로를 화면에서 읽고 A4로 인쇄하는 운영 계획 보고서 HTML 한 장으로 만든다."""

from collections import defaultdict
from html import escape
from math import ceil, floor, log10
from pathlib import Path

from streamlit_website.db_connection.filter_candidates import FEATURE_LABELS, STATUS_LABELS
from streamlit_website.DP.recommend import TIGHT_BUDGET_RATIO, recommend
from streamlit_website.ml.pipeline import planned_final_rows

STYLE = Path(__file__).with_name("report.css").read_text(encoding="utf-8")
FONT = "https://cdn.jsdelivr.net/npm/pretendard@1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css"
ACTION_LABELS = {"adopt": "최초 도입", "keep": "유지", "retrain": "재학습", "switch": "교체"}
# 단위, 뒤에 붙는 '(으)로', 주격 조사
UNITS = {"KRW": ("원", "으로", "이"), "seconds": ("초", "로", "가")}
TRAINING_REQUIRED = "fit_full_prefix"
SECTIONS = (("conclusion", "결론"), ("schedule", "운영 일정"), ("budget", "예산별 대안"), ("inputs", "입력 조건"),
            ("evidence", "현재 데이터"), ("pool", "후보 풀"), ("remarks", "비고"))
EXCLUSION_REASONS = {
    "no_historical_performance_estimate_for_candidate": "비슷한 과거 prefix에 성능 기록이 없음",
    "no_training_row_count_for_stage": "구간 학습 행 수를 정하지 못함",
    "training_cost_insufficient_historical_execution_data": "학습비를 추정할 실행 기록이 부족함",
    "inference_cost_insufficient_historical_execution_data": "추론비를 추정할 실행 기록이 부족함",
    "structurally_infeasible": "실행 조건 미달",
}
MARKS = """<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs>
<symbol id="mark-adopt" viewBox="0 0 16 16"><rect x="3" y="3" width="10" height="10" stroke="none"/></symbol>
<symbol id="mark-keep" viewBox="0 0 16 16"><circle cx="8" cy="8" r="5.2" fill="none" stroke-width="1.6"/></symbol>
<symbol id="mark-retrain" viewBox="0 0 16 16"><circle cx="8" cy="8" r="5.8" stroke="none"/></symbol>
<symbol id="mark-switch" viewBox="0 0 16 16"><circle cx="8" cy="8" r="6" fill="none" stroke-width="1.5"/>
<circle cx="8" cy="8" r="2.6" stroke="none"/></symbol></defs></svg>"""
SCRIPT = """<script>
document.querySelectorAll('.plan [data-stage]').forEach(cell => {
  const column = document.querySelectorAll(`.plan [data-stage="${cell.dataset.stage}"]`);
  const paint = on => column.forEach(item => item.classList.toggle('is-hover', on));
  ['pointerenter', 'focus'].forEach(type => cell.addEventListener(type, () => paint(true)));
  ['pointerleave', 'blur'].forEach(type => cell.addEventListener(type, () => paint(false)));
});
document.querySelectorAll('.chart').forEach(chart => {
  const tip = chart.querySelector('.tip');
  chart.querySelectorAll('.point').forEach(point => {
    const show = () => {
      const frame = chart.getBoundingClientRect(), dot = point.querySelector('.dot').getBoundingClientRect();
      tip.textContent = point.dataset.tip;
      tip.style.left = Math.max(0, Math.min(dot.left - frame.left + 14, frame.width - 176)) + 'px';
      tip.style.top = Math.max(0, dot.top - frame.top - 66) + 'px';
      tip.classList.add('is-on');
    };
    const hide = () => tip.classList.remove('is-on');
    ['pointerenter', 'focus'].forEach(type => point.addEventListener(type, show));
    ['pointerleave', 'blur'].forEach(type => point.addEventListener(type, hide));
  });
});
</script>"""


def amount(value):
    """비용 표기: 1,000 이상은 정수, 그 아래는 유효숫자 세 자리."""
    if 0 < abs(value) < 0.001:
        return "0.001 미만"
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.3g}"


def days(value):
    return f"{value:,.1f}".rstrip("0").rstrip(".")


def unit_word(unit):
    return UNITS.get(unit, (unit, "로", "이"))


def money(value, unit):
    return "제한 없음" if value is None else amount(value) + unit_word(unit)[0]


def mark(action, *, muted=False):
    css = "mark is-muted" if muted else "mark"
    return (f'<svg class="{css}" aria-hidden="true"><use href="#mark-{action}"/></svg>'
            f'<span class="sr">{ACTION_LABELS[action]}</span>')


def now_class(stage, *extra):
    classes = [*extra, "is-now-col"] if stage == 0 else list(extra)
    return f' class="{" ".join(classes)}"' if classes else ""


def lane_label(model, config_id, head):
    head = f'<span class="head">head {escape(head)}</span>' if head else ""
    return f'<b>{escape(model)}</b>{head}<span class="id">{escape(config_id)}</span>'


def inline_label(model, config_id, head):
    head = f" · head {escape(head)}" if head else ""
    return f'{escape(model)} <code class="hash">{escape(config_id)}</code>{head}'


def describe_exclusion(reason):
    return EXCLUSION_REASONS.get(reason) or f"실행 조건 미달 ({escape(str(reason))})"


def count_trainings(path, target_uses):
    return sum(step["action"] != "keep" and target_uses.get(step["candidate_id"]) == TRAINING_REQUIRED
               for step in path)


def open_section(section_id, claim=None):
    """구획 머리띠와 본문 시작. 결론처럼 claim이 없으면 본문 칸을 열지 않는다."""
    title = dict(SECTIONS)[section_id]
    body = f'<div class="body"><p class="claim">{claim}</p>' if claim else ""
    return (f'<section class="part" id="{section_id}" aria-labelledby="{section_id}-title">'
            f'<h2 class="band" id="{section_id}-title">{title}</h2>{body}')


def render_masthead(context):
    payload, conditions, info = context["payload"], context["conditions"], context["input"]
    unit = payload["cost_unit"]
    if unit == "KRW":
        rates = (f'CPU {conditions.get("cpu_hourly_cost") or 0:,.0f}원'
                 f'<small>GPU {conditions.get("gpu_hourly_cost") or 0:,.0f}원</small>')
    else:
        rates = "단가 없음<small>비용은 계산 시간(초)</small>"
    source = escape(info.get("source_name") or "이름 없음")
    return f"""<header class="masthead">
<div class="titleblock"><h1>운영 계획 보고서</h1>
<p class="subtitle">TSAD 모델 도입·유지·재학습·교체 일정</p>
<p class="meta"><span>계획 번호 {context['plan_no']}</span> · <span>{context['generated']}</span> · 입력 파일&nbsp;{source}</p></div>
<table class="approval"><caption class="sr">계획 조건 요약</caption><tbody><tr>
<th class="vlabel" rowspan="2" scope="row">조건</th><th scope="col">데이터</th><th scope="col">운영 기간</th>
<th scope="col">예산</th><th scope="col">시간당 단가</th></tr>
<tr><td>{info['row_count']:,}행<small>센서 {info['channel_count']}개</small></td>
<td>{days(conditions.get('operating_days') or 0)}일</td><td>{money(payload.get('budget'), unit)}</td>
<td>{rates}</td></tr></tbody></table></header>"""


def render_conclusion(context):
    result, stages, unit = context["result"], context["stages"], context["payload"]["cost_unit"]
    html = [open_section("conclusion")]
    if result["status"] == "infeasible":
        html.append(f"""<div class="verdict"><div class="label now">지금 할 일</div><div class="value">
<p class="action">계획을 만들지 못함</p><p class="action-detail">구간 {result['blocked_stage'] + 1}에서 이어 갈 후보나
추론 행이 없다. 아래 후보 풀과 비고를 확인한다.</p></div></div></section>""")
        return "".join(html)
    step = result["path"][0]
    if step["action"] == "keep":
        detail = f'기존 checkpoint <code>{escape(str(step.get("checkpoint_id") or ""))}</code> · 다시 학습하지 않음'
        if step.get("last_trained_at"):
            detail += f' · 마지막 학습 {escape(step["last_trained_at"])}'
    elif context["target_uses"].get(step["candidate_id"]) == TRAINING_REQUIRED:
        detail = f'지금 확보한 {stages[0]["available_rows"]:,}행으로 학습'
    else:
        detail = "학습 없이 바로 적용"
    head = f' · head {escape(step["head"])}' if step["head"] else ""
    if len(stages) > 1:
        upcoming = stages[1]
        next_line = (f'데이터 {upcoming["ratio_percent"]}% 도달<small>{upcoming["available_rows"]:,}행 · '
                     f'약 {days(upcoming["elapsed_days"])}일 뒤</small>')
    else:
        next_line = "남은 기간에 다음 비율에 닿지 않음<small>운영 조건이나 예산이 바뀌면 다시 계산</small>"
    html.append(f"""<div class="verdict"><div class="label now">지금 할 일</div><div class="value">
<p class="action">{mark(step['action'])}<span>{escape(step['model'])} {ACTION_LABELS[step['action']]}</span></p>
<p class="action-detail">설정 <code>{escape(step['config_id'])}</code>{head} · {detail}</p></div></div>
<div class="verdict"><div class="label">다음 재계산</div><div class="value"><p class="next">{next_line}</p></div></div>""")
    reference, switched = context["reference"], context["switched"]
    performance = result["timeline_performance"]
    base = reference["timeline_performance"] if reference else None
    name = "참고 경로" if switched else "가장 싼 경로"
    budget = result["budget"]
    spent = money(result["total_cost"], unit)
    if budget:
        spent += f" / 예산 {money(budget, unit)} ({result['total_cost'] / budget:.0%})"
    facts = [("예측 VUS-PR", f"<b>{performance:.2f}</b>")]
    if base is not None:
        gap = "차이 근거 없음" if switched else f"대비 <b>{performance - base:+.2f}</b>"
        facts.append((name, f"<b>{base:.2f}</b> · {gap}"))
    facts += [("학습·추론비", f"<b>{spent}</b>"),
              ("학습", f'<b>{count_trainings(result["path"], context["target_uses"])}회</b>')]
    tally = "".join(f"<div><dt>{label}</dt><dd>{value}</dd></div>" for label, value in facts)
    html.append(f'<div class="verdict"><div class="label">계획 값</div><div class="value"><dl class="tally">{tally}</dl>'
                "</div></div>")
    if result["status"] == "over_budget":
        note = (f"예산 {money(budget, unit)} 안에서는 끝까지 갈 경로가 없다. 아래 일정은 가장 싼 경로이며 "
                f"{money(result['total_cost'], unit)}{unit_word(unit)[2]} 든다.")
    elif switched:
        note = (f"빠듯한 예산이라 가장 싼 경로를 추천한다(예산이 가장 싼 경로 비용의 {context['tight']:,.1f}배). "
                f"Dev18 백테스트에서 {TIGHT_BUDGET_RATIO}배 이하 예산은 어떤 방식도 가장 싼 계획을 통계적으로 넘지 못했다. "
                "예측이 더 높은 경로는 운영 일정에 참고로 둔다.")
    elif base is not None and performance - base > 1e-9:
        note = f"예측 VUS-PR은 새 현장에서 ±0.3 정도 틀리므로 가장 싼 경로와의 차이(+{performance - base:.2f})로 읽는다."
    else:
        note = "이 예산에서는 가장 싼 경로가 답이다. 예측 VUS-PR은 절대값이 ±0.3 정도 틀리므로 숫자를 그대로 약속하지 않는다."
    alert = " is-alert" if result["status"] == "over_budget" else ""
    html.append(f'<p class="conclusion-note{alert}">{note}</p></section>')
    return "".join(html)


def lane_rows(path, stage_count, *, muted=False):
    """경로에 쓰인 후보마다 한 행. 쓰는 구간에 보전 기호를 찍고 쓰는 동안 가로선을 잇는다."""
    lanes = {}
    for step in path:
        lanes.setdefault(step["candidate_id"], {"step": step, "stages": {}})["stages"][step["stage"]] = step["action"]
    html = []
    for lane in lanes.values():
        step, active = lane["step"], lane["stages"]
        cells = []
        for stage in range(stage_count):
            if stage not in active:
                cells.append(f'<td data-stage="{stage}"{now_class(stage)}></td>')
                continue
            ends = (["start"] if stage - 1 not in active else []) + (["end"] if stage + 1 not in active else [])
            cells.append(f'<td data-stage="{stage}"{now_class(stage, "on", *ends)} style="--i:{stage}">'
                         f"{mark(active[stage], muted=muted)}</td>")
        label = lane_label(step["model"], step["config_id"], step["head"])
        html.append(f'<tr class="lane-row"><th scope="row">{label}</th>{"".join(cells)}</tr>')
    return "".join(html)


def metric_row(label, values, *, css="metric"):
    cells = "".join(f'<td data-stage="{stage}"{now_class(stage)}>{value}</td>' for stage, value in enumerate(values))
    return f'<tr class="{css}"><th scope="row">{label}</th>{cells}</tr>'


def render_schedule(context):
    result, stages, unit = context["result"], context["stages"], context["payload"]["cost_unit"]
    if result["status"] == "infeasible":
        return ""
    path = result["path"]
    total_days = sum(stage["weight"] for stage in stages)
    changes = defaultdict(int)
    for step in path[1:]:
        changes[step["action"]] += 1
    if changes["retrain"] or changes["switch"]:
        parts = [f"{ACTION_LABELS[action]} {changes[action]}회" for action in ("retrain", "switch") if changes[action]]
        claim = f"{days(total_days)}일 동안 {', '.join(parts)}를 거친다"
    elif path[0]["action"] == "keep":
        claim = f"{days(total_days)}일 동안 기존 checkpoint를 다시 학습하지 않고 그대로 쓴다"
    else:
        claim = f"{days(total_days)}일 동안 {escape(path[0]['model'])} 하나를 다시 학습하지 않고 쓴다"
    symbol = unit_word(unit)[0]
    count = len(stages)
    widths = [max(stage["weight"] / total_days, 0.8 / count) if total_days else 1 / count for stage in stages]
    columns = "".join(f'<col style="width:{width / sum(widths) * 100:.2f}%">' for width in widths)
    ratios = [stage["ratio_percent"] for stage in stages] + [100]
    flag = '<span class="now-flag">지금</span>'
    head = "".join(f'<th scope="col" data-stage="{index}" tabindex="0" class="stage-head{" is-now" if index == 0 else ""}">'
                   f'구간 {index + 1}{flag if index == 0 else ""}</th>' for index in range(count))
    ratio_row = metric_row("DB 비율 구간", [f"{ratios[index]} → {ratios[index + 1]}%" if ratios[index] != ratios[index + 1]
                                           else f"{ratios[index]}%" for index in range(count)])
    facts = [
        f'<tbody><tr class="group"><th colspan="{count + 1}" scope="rowgroup">구간 정보</th></tr>',
        metric_row("운영 일차", [f"{days(stage['elapsed_days'])} – {days(stage['elapsed_days'] + stage['weight'])}"
                             for stage in stages]),
        metric_row("구간 시작 확보 행", [f"{stage['available_rows']:,}" for stage in stages]),
        metric_row("구간 추론량 (행)", [f"{stage['inference_rows']:,.0f}" for stage in stages], css="metric total"),
        "</tbody>",
    ]
    chosen = [
        f'<tbody><tr class="group"><th colspan="{count + 1}" scope="rowgroup">{"추천 : 가장 싼 경로" if context["switched"] else "고른 경로"}</th></tr>',
        lane_rows(path, count),
        metric_row("예측 VUS-PR", [f"{step['predicted_performance']:.2f}" for step in path]),
        metric_row(f"학습비 ({symbol})", [amount(step["training_cost"]) for step in path]),
        metric_row(f"추론비 ({symbol})", [amount(step["inference_cost"]) for step in path]),
        metric_row(f"누적 비용 ({symbol})", [amount(step["cumulative_cost"]) for step in path], css="metric total"),
        "</tbody>",
    ]
    other, reference = context["reference"], []
    if other and (context["switched"] or [step["candidate_id"] for step in other["path"]]
                  != [step["candidate_id"] for step in path]):
        title = "참고 : 예측으로만 고른 경로 · 빠듯한 예산이라 차이에 근거가 없다" if context["switched"] else "비교 : 가장 싼 경로"
        reference = [
            f'<tbody class="ref"><tr class="group"><th colspan="{count + 1}" scope="rowgroup">{title} '
            f"(예측 VUS-PR {other['timeline_performance']:.2f}, 전체 {money(other['total_cost'], unit)})</th></tr>",
            lane_rows(other["path"], count, muted=True),
            metric_row("예측 VUS-PR", [f"{step['predicted_performance']:.2f}" for step in other["path"]]),
            metric_row(f"누적 비용 ({symbol})", [amount(step["cumulative_cost"]) for step in other["path"]],
                       css="metric total"),
            "</tbody>",
        ]
    legend = "".join(f"<span>{mark(action)}{label}</span>" for action, label in ACTION_LABELS.items())
    return f"""{open_section("schedule", claim)}
<p class="lede">DB 비율은 운영 기간 끝까지 모을 {context['final_rows']:,}행을 100%로 본
값이며 데이터가 다음 비율에 닿는 날 구간이 바뀐다. 운영 기간이나 수집 속도를 바꾸면 이 비율과 재학습 시점도
바뀐다.</p>
<div class="scroll"><table class="plan{' is-single' if count == 1 else ''}"><colgroup><col class="c-label">{columns}</colgroup>
<thead><tr><th scope="row">구간</th>{head}</tr>{ratio_row}</thead>{"".join(chosen)}{"".join(reference)}{"".join(facts)}</table></div>
<p class="legend"><b>범례</b>{legend}</p></div></section>"""


def render_chart(frontier, budget, unit):
    """가로는 전체 비용(차이가 크면 로그 눈금), 세로는 운영 기간 예측 VUS-PR인 계단 그래프."""
    width, height, left, right, top, bottom = 560, 300, 48, 20, 26, 46
    costs = [point["total_cost"] for point in frontier]
    values = [point["timeline_performance"] for point in frontier]
    logarithmic = min(costs) > 0 and max(costs) / min(costs) > 30
    if logarithmic:
        low, high = floor(log10(min(costs))), ceil(log10(max(costs) * 1.0001))
        ticks = [10.0 ** power for power in range(low, high + 1)]

        def position(cost):
            return left + (log10(cost) - low) / max(high - low, 1) * (width - left - right)
    else:
        low, step = 0, nice_step(max(costs) * 1.05 or 1.0)
        span = ceil(max(costs) * 1.05 / step) * step or step
        ticks = [step * index for index in range(round(span / step) + 1)]

        def position(cost):
            return left + cost / span * (width - left - right)
    bottom_value, top_value = floor(min(values) * 10) / 10, ceil(max(values) * 10) / 10
    if top_value - bottom_value < 0.2:
        bottom_value, top_value = max(0.0, bottom_value - 0.1), min(1.0, top_value + 0.1)

    def level(value):
        return top + (top_value - value) / ((top_value - bottom_value) or 1) * (height - top - bottom)
    grades = [bottom_value + index * 0.1 for index in range(round((top_value - bottom_value) / 0.1) + 1)]
    grid = [f'<line x1="{left}" x2="{width - right}" y1="{level(grade):.1f}" y2="{level(grade):.1f}"/>' for grade in grades]
    grid += [f'<line x1="{position(tick):.1f}" x2="{position(tick):.1f}" y1="{top}" y2="{height - bottom}"/>'
             for tick in ticks]
    labels = [f'<text x="{left - 8}" y="{level(grade) + 4:.1f}" text-anchor="end">{grade:.1f}</text>' for grade in grades]
    labels += [f'<text x="{position(tick):.1f}" y="{height - bottom + 17}" text-anchor="middle">{amount(tick)}</text>'
               for tick in ticks]
    line = f"M{position(costs[0]):.1f} {level(values[0]):.1f}"
    for cost, value in zip(costs[1:], values[1:]):
        line += f" H{position(cost):.1f} V{level(value):.1f}"
    line += f" H{width - right}"
    symbol = unit_word(unit)[0]
    budget_x = min(position(budget), width - right) if budget and (not logarithmic or budget >= 10.0 ** low) else None
    points = []
    for index, point in enumerate(frontier):
        x, y = position(point["total_cost"]), level(point["timeline_performance"])
        css = "point is-chosen" if point["chosen"] else "point" if point["within_budget"] else "point is-over"
        tip = (f"전체 비용 {amount(point['total_cost'])}{symbol}\n예측 VUS-PR {point['timeline_performance']:.2f}\n"
               f"도입·재학습·교체 {point['trainings']}회" + ("" if point["within_budget"] else " · 예산 초과"))
        name = ("고른 경로" if point["chosen"] else "가장 싼 경로" if index == 0
                else "참고 경로" if point.get("reference") else "")
        text = ""
        if name:
            span = len(name) * 11 + 30
            crosses = budget_x is not None and x < budget_x < x + 10 + span
            anchor, shift = ("end", -10) if x > width - 160 or (crosses and x - span > left) else ("start", 10)
            text = (f'<text x="{x + shift:.1f}" y="{y - 10:.1f}" text-anchor="{anchor}">'
                    f"{name} {point['timeline_performance']:.2f}</text>")
        points.append(f'<g class="{css}" tabindex="0" data-tip="{escape(tip)}" aria-label="{escape(tip)}">'
                      f'<circle class="hit" cx="{x:.1f}" cy="{y:.1f}" r="13"/>'
                      f'<circle class="dot" cx="{x:.1f}" cy="{y:.1f}" r="5"/>{text}</g>')
    budget_mark = ""
    if budget_x is not None:
        x = budget_x
        label_at = (f'x="{x + 6:.1f}" y="{top + 12}" text-anchor="start"' if x - left < 120
                    else f'x="{x - 6:.1f}" y="{top - 12}" text-anchor="end"')
        budget_mark = (f'<g class="budget"><line x1="{x:.1f}" x2="{x:.1f}" y1="{top - 8}" y2="{height - bottom}"/>'
                       f"<text {label_at}>예산 {amount(budget)}{symbol}</text></g>")
    scale = "로그 눈금" if logarithmic else "선형 눈금"
    return f"""<figure class="chart"><svg viewBox="0 0 {width} {height}" role="img" aria-label="예산별 비용과 예측 VUS-PR">
<g class="grid">{"".join(grid)}</g><g class="axis">{"".join(labels)}
<text class="axis-title" x="{(left + width - right) / 2:.0f}" y="{height - 6}" text-anchor="middle">전체 학습·추론비 ({symbol}, {scale})</text>
<text class="axis-title" x="{left - 40}" y="{top - 12}">예측 VUS-PR</text></g>
<path class="step" d="{line}"/>{budget_mark}{"".join(points)}</svg><div class="tip" role="status"></div>
<figcaption>계단은 그 비용으로 얻을 수 있는 가장 높은 예측이다.<span class="screen-only"> 점에 마우스를 올리면 값이 보인다.</span></figcaption></figure>"""


def nice_step(maximum):
    """선형 눈금 간격: 네 칸 안팎이 되는 1·2·2.5·5 × 10^n."""
    raw = maximum / 4
    magnitude = 10.0 ** floor(log10(raw))
    return next(base * magnitude for base in (1, 2, 2.5, 5, 10) if base * magnitude >= raw)


def render_budget(context):
    result, unit = context["result"], context["payload"]["cost_unit"]
    frontier = result.get("frontier") or []
    if not frontier:
        return ""
    chosen = next(index for index, point in enumerate(frontier) if point["chosen"])
    better = [point for point in frontier[chosen + 1:]
              if point["timeline_performance"] > frontier[chosen]["timeline_performance"]]
    word, toward, _ = unit_word(unit)
    if context["switched"]:
        claim = "빠듯한 예산에서는 비용을 더 들여도 예측이 오른다는 근거가 없어 가장 싼 경로를 고른다"
    elif better:
        claim = (f"예산을 {amount(better[0]['total_cost'])}{word}{toward} 늘리면 예측 VUS-PR이 "
                 f"{better[0]['timeline_performance']:.2f}까지 오른다")
    else:
        claim = "예산을 더 늘려도 예측 VUS-PR은 오르지 않는다"
    base = frontier[0]["timeline_performance"]
    rows = []
    for index, point in enumerate(frontier):
        notes = (["가장 싼 경로"] * (index == 0) + ["고른 경로"] * point["chosen"]
                 + ["참고 경로"] * bool(point.get("reference")) + ["예산 초과"] * (not point["within_budget"]))
        unproven = context["switched"] and index and point["total_cost"] <= TIGHT_BUDGET_RATIO * frontier[0]["total_cost"]
        gap = "근거 없음" if unproven else f'{point["timeline_performance"] - base:+.2f}'
        css = ' class="chosen"' if point["chosen"] else ""
        rows.append(f'<tr{css}><td class="num">{amount(point["total_cost"])}</td>'
                    f'<td class="num">{point["timeline_performance"]:.2f}</td>'
                    f'<td class="num">{gap}</td>'
                    f'<td class="num">{point["trainings"]}회</td><td class="note-cell">{" · ".join(notes)}</td></tr>')
    table = f"""<table class="data"><thead><tr><th scope="col" class="num">전체 비용 ({word})</th>
<th scope="col" class="num">예측 VUS-PR</th><th scope="col" class="num">가장 싼 경로 대비</th>
<th scope="col" class="num">도입·재학습·교체</th><th scope="col">비고</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>"""
    return f"""{open_section("budget", claim)}
<p class="lede">비용이 늘 때 예측이 오르는 경로만 남겼다. 가장 싼 경로, 고른 경로, 가장 높은 경로는 항상 들어간다.</p>
<div class="frontier">{render_chart(frontier, result["budget"], unit)}<div class="scroll">{table}</div></div></div></section>"""


def render_inputs(context):
    info, conditions, payload, current = context["input"], context["conditions"], context["payload"], context["current"]
    unit = payload["cost_unit"]
    sensors = [escape(str(name)) for name in info.get("sensor_columns", [])]
    shown = ", ".join(sensors[:12]) + (f" 외 {len(sensors) - 12}개" if len(sensors) > 12 else "")
    if current.get("candidate_id"):
        row = context["candidates"].get(current["candidate_id"], {})
        details = [f'checkpoint {escape(current.get("checkpoint") or "없음")}',
                   f'학습 행 수 {current["trained_rows"]:,}' if current.get("trained_rows") else "학습 행 수 입력 없음",
                   f'마지막 학습 {escape(current["last_trained_at"])}' if current.get("last_trained_at") else ""]
        current_text = (inline_label(row.get("model", "등록되지 않은 후보"), row.get("config_id", current["candidate_id"]),
                                     row.get("head", "")) + f'<small>{" · ".join(filter(None, details))}</small>')
    else:
        current_text = "없음<small>처음 도입하는 경로만 비교</small>"
    rate = conditions.get("collection_rows_per_second") or 0
    batch = conditions.get("inference_batch_length")
    gpu = conditions.get("gpu_memory_gib") or 0
    priced = unit == "KRW"
    fields = [
        ("입력 파일", escape(info.get("source_name") or "이름 없음")),
        ("누적 정상 행 수", f"{info['row_count']:,}행<small>올린 구간 전체를 씀</small>"),
        ("센서 열", f"{len(sensors)}개<small>{shown}</small>"),
        ("현재 모델", current_text),
        ("남은 운영 기간", f"{days(conditions.get('operating_days') or 0)}일"),
        ("예상 수집 속도", f"{rate:g}행/초<small>하루 {rate * 86400:,.0f}행</small>"),
        ("기간 끝 예상 누적 행", f"{context['final_rows']:,}행<small>DB 비율 100% 기준</small>"),
        ("예상 추론량", f"하루 {conditions.get('inference_rows_per_day') or 0:,}행"),
        ("연속 추론 길이", f"{batch:,}행" if batch else "입력 없음<small>하루 추론량으로 대신 판정</small>"),
        ("남은 예산", money(payload.get("budget"), unit)),
        ("CPU 1시간 비용", f"{conditions.get('cpu_hourly_cost') or 0:,.0f}원" if priced else "입력 없음"),
        ("GPU 1시간 비용", f"{conditions.get('gpu_hourly_cost') or 0:,.0f}원" if priced else "입력 없음"),
        ("GPU 메모리", f"{gpu:g} GiB" if gpu else "없음<small>GPU에서만 잰 후보는 제외</small>"),
        ("성능 차이 허용 폭 ε", f"{payload.get('tolerance', 0):g}"),
        ("비용 단위", "원<small>측정 장치의 시간당 단가로 환산</small>" if priced
         else "초<small>L4 GPU·CPU 8개 기준 계산 시간</small>"),
    ]
    cells = []
    for index, (name, value) in enumerate(fields):
        wide = ' class="wide"' if index == len(fields) - 1 and len(fields) % 2 else ""
        cells.append(f"<div{wide}><dt>{name}</dt><dd>{value}</dd></div>")
    return f"""{open_section("inputs", "같은 CSV와 이 조건을 넣으면 같은 계획이 다시 나온다")}
<dl class="fields fields-end">{"".join(cells)}</dl></div></section>"""


def render_evidence(context):
    summary = (context["features"] or {}).get("summary") or {}
    matches = context["ml_output"].similarity_matches
    families = defaultdict(lambda: {"weight": 0.0, "files": set(), "count": 0})
    for match in matches:
        family = families[match.family]
        family["weight"] += match.weight
        family["files"].add(match.csv_file)
        family["count"] += 1
    total = sum(family["weight"] for family in families.values()) or 1
    ordered = sorted(families.items(), key=lambda item: -item[1]["weight"])
    if ordered:
        names = [name for name, family in ordered[:2] if family["weight"] / total >= 0.15] or [ordered[0][0]]
        claim = f'현재 데이터는 {"·".join(escape(name) for name in names)} 과거 prefix와 가장 가깝다'
    else:
        claim = "비슷한 과거 prefix를 찾지 못했다"

    def feature_value(key):
        value = summary.get(key)
        if value is None:
            return '<span class="muted">계산 안 됨</span>'
        return f"{value:,}" if key in ("observed_row", "input_column") else f"{value:.3g}"

    execution_only = {"observed_row", "input_column"}
    features = "".join(
        f'<tr><th scope="row">{label}{"<small>유사도 제외</small>" if key in execution_only else ""}</th>'
        f'<td class="num">{feature_value(key)}</td></tr>' for key, label in FEATURE_LABELS.items())
    similar = ""
    if matches:
        family_rows = "".join(
            f'<tr><th scope="row">{escape(name)}</th><td class="num">{len(family["files"])}</td>'
            f'<td class="num">{family["count"]}</td><td class="num"><span class="bar" '
            f'style="width:{family["weight"] / total * 64:.0f}px"></span>{family["weight"] / total:.0%}</td></tr>'
            for name, family in ordered)
        nearest = "".join(
            f'<tr><td>{escape(match.family)}</td><td><code class="file">{escape(match.csv_file)}</code></td>'
            f'<td class="num">{match.q_percent}%</td><td class="num">{match.distance:.2f}</td></tr>'
            for match in sorted(matches, key=lambda item: item.distance)[:5])
        wanted = (context["ml_output"].metadata.get("similarity_summary") or {}).get("requested_k", len(matches))
        similar = f"""<h3>비슷한 과거 prefix {len(matches)}개 (거리 가중 kNN, k={wanted})</h3>
<table class="data"><thead><tr><th scope="col">family</th><th scope="col" class="num">파일</th>
<th scope="col" class="num">prefix</th><th scope="col" class="num">가중치 비중</th></tr></thead><tbody>{family_rows}</tbody></table>
<h3>가장 가까운 prefix 5개</h3><table class="data"><thead><tr><th scope="col">family</th><th scope="col">파일</th>
<th scope="col" class="num">q</th><th scope="col" class="num">거리</th></tr></thead><tbody>{nearest}</tbody></table>"""
    return f"""{open_section("evidence", claim)}
<p class="lede">feature를 표준화한 뒤 Euclidean 거리로 가까운 과거 Dev18 prefix를 찾았다. 가까울수록 크게 가중해
후보별 성능을 추정했다. 행 수는 유사도에 넣지 않고 실행 조건과 비용 추정에만 쓴다.</p>
<div class="twin"><div><h3>현재 데이터의 feature</h3><table class="data"><thead><tr><th scope="col">feature</th>
<th scope="col" class="num">값</th></tr></thead><tbody>{features}</tbody></table></div><div>{similar}</div></div>
</div></section>"""


def render_pool(context):
    payload, result, ml_output = context["payload"], context["result"], context["ml_output"]
    eligible, excluded = context["eligible"], context["excluded"]
    total = len(eligible) + len(excluded)
    counts = dict.fromkeys(STATUS_LABELS, 0)
    for row in (*eligible, *excluded):
        counts[row.get("status", "eligible")] += 1
    parts = []
    if excluded:
        rows = "".join(f'<tr><td>{inline_label(row["model"], row["config_id"], row["head"])}</td>'
                       f'<td>{STATUS_LABELS[row["status"]]}</td><td>{escape(row["reason"])}</td></tr>' for row in excluded)
        parts.append(f"""<h3>계획에서 뺀 후보</h3><div class="scroll"><table class="data wide"><thead><tr><th scope="col">모델 · 설정</th>
<th scope="col">판정</th><th scope="col">사유</th></tr></thead><tbody>{rows}</tbody></table></div>""")
    dropped = defaultdict(list)
    for stage, estimates in ml_output.stage_candidates_excluded.items():
        for estimate in estimates:
            key = (estimate.model, estimate.configuration, estimate.head, estimate.exclusion_reason)
            dropped[key].append(int(stage) + 1)
    if dropped:
        rows = "".join(f'<tr><td>{inline_label(model, config_id, head)}</td>'
                       f'<td>{", ".join(map(str, sorted(stages)))}</td><td>{describe_exclusion(reason)}</td></tr>'
                       for (model, config_id, head, reason), stages in dropped.items())
        parts.append(f"""<h3>구간별로 뺀 후보</h3><div class="scroll"><table class="data wide"><thead><tr><th scope="col">모델 · 설정</th>
<th scope="col">구간</th><th scope="col">사유</th></tr></thead><tbody>{rows}</tbody></table></div>""")
    options = sorted((row for row in payload["training_options"] if row["stage"] == 0),
                     key=lambda row: -row["predicted_performance"])
    if options:
        names = {row["candidate_id"]: row for row in payload["candidates"]}
        inference = {row["candidate_id"]: row["inference_cost"] for row in payload["inference_options"]
                     if row["stage"] == 0 and row["trained_stage"] == 0}
        chosen = result["path"][0]["candidate_id"] if result.get("path") else None
        symbol = unit_word(payload["cost_unit"])[0]
        rows = []
        for option in options[:10]:
            name = names[option["candidate_id"]]
            css = ' class="chosen"' if option["candidate_id"] == chosen else ""
            rows.append(f'<tr{css}><td>{inline_label(name["model"], name["config_id"], name["head"])}</td>'
                        f'<td class="num">{option["predicted_performance"]:.2f}</td>'
                        f'<td class="num">{amount(option["training_cost"])}</td>'
                        f'<td class="num">{amount(inference.get(option["candidate_id"], 0))}</td></tr>')
        parts.append(f"""<h3>구간 1에서 비교한 후보 (예측 VUS-PR 상위 10개, 전체 {len(options)}개)</h3>
<div class="scroll"><table class="data wide"><thead><tr><th scope="col">모델 · 설정</th><th scope="col" class="num">예측 VUS-PR</th>
<th scope="col" class="num">학습비 ({symbol})</th><th scope="col" class="num">구간 추론비 ({symbol})</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>""")
    return f"""{open_section("pool", f"등록 후보 {total}개 중 {len(eligible)}개를 계획에 넣었다")}
<p class="lede">센서 수와 모델의 필수 입력 조건으로만 1차 축소했다. 고정 제약으로 {counts["infeasible"]}개를 뺐고
미검증 후보는 {counts["unverified"]}개다. 지금 데이터의 길이·통계·결측으로는 후보를 빼지 않으며 실행 가능 여부와 비용은
구간마다 다시 판정한다. 성능 순위로 후보를 자르지도 않는다.</p>{"".join(parts)}</div></section>"""


def render_remarks(context):
    payload, result, conditions, current = context["payload"], context["result"], context["conditions"], context["current"]
    stages = context["stages"]
    remarks = [
        "<b>예측 VUS-PR</b> : 구간마다 쓰는 모델의 예측을 구간 길이로 가중해 평균한 값이다. 새 현장에서는 절대값이 "
        "±0.3 정도 틀린다 → 가장 싼 경로와의 차이로 읽는다. 오차는 대부분 "
        "모든 후보에 같이 얹힌다. Dev18 검증에서는 예측 전체에 0.3을 더해도 고른 계획이 98% 같았다.",
        "<b>비용</b> : L4 GPU 1장·CPU 8개에서 잰 실행 시간의 중앙값이다(PaAno만 행 수 회귀) → 예산을 꽉 채운 계획은 "
        "실제로 넘을 수 있다. Dev18 백테스트에서 예산을 넘은 파일은 6~11%였고 대부분 1.2배 이내였다.",
        f"<b>미래 데이터</b> : 데이터 발생 특성이 지금과 같다고 본다 → 운영 기간 끝까지 모을 {context['final_rows']:,}행을 "
        "100%로 두고 DB가 성능을 잰 비율(5·10·20·40·60·80%) 경계에서 구간을 나눈다. 지금 데이터가 5%보다 적으면 "
        "5% 결과로 본다. 운영 기간이나 수집 속도를 바꾸면 지금 비율과 재학습 시점도 바뀐다.",
        "<b>성능 변화</b> : 목표 비율에 과거 결과가 없으면 더 적은 데이터에서 잰 마지막 값이 유지된다고 본다. "
        "데이터 비율에 따라 성능이 바뀌는 후보는 PaAno·GDN뿐이다. 나머지는 모든 비율에서 같은 실행 결과를 쓴다.",
    ]
    if context["tight"] is not None:
        remarks.append(f"<b>빠듯한 예산</b> : 이번 예산은 가장 싼 경로 비용의 {context['tight']:,.1f}배다 → "
                       f"Dev18 백테스트에서 {TIGHT_BUDGET_RATIO}배 이하 예산은 어떤 방식도 가장 싼 계획을 통계적으로 넘지 "
                       "못했다. 그래서 가장 싼 경로를 추천한다.")
    if len(stages) > 1:
        remarks.append(f"<b>재계산</b> : 지금 할 일만 적용한다 → 데이터가 {stages[1]['ratio_percent']}%에 닿으면 누적 "
                       "prefix 전체와 남은 기간, 이미 쓴 비용을 뺀 남은 예산으로 다시 계산한다.")
    else:
        remarks.append("<b>재계산</b> : 운영 조건이나 예산이 바뀌면 누적 prefix 전체로 다시 계산한다.")
    if not conditions.get("inference_batch_length"):
        remarks.append(f"<b>연속 추론 길이</b> : 입력하지 않아 하루 추론량 {conditions.get('inference_rows_per_day') or 0:,}행을 "
                       "한 번에 넣는 길이로 보고 실행 조건을 판정했다 → 실제 길이를 넣으면 이 근사를 쓰지 않는다.")
    if current.get("candidate_id") and payload.get("current_checkpoint") is None:
        remarks.append("<b>현재 모델</b> : 후보 풀에 없거나 checkpoint ID·성능 추정이 없어 유지 선택지를 만들지 못했다 → "
                       "새로 도입하는 경로만 비교했다.")
    elif current.get("candidate_id") and not current.get("trained_rows"):
        remarks.append(f"<b>기존 checkpoint</b> : 학습 행 수를 넣지 않아 지금 올린 {context['input']['row_count']:,}행으로 "
                       "학습했다고 봤다 → 오래전에 학습했다면 유지 성능을 높게 본 것이다.")
    found = len(context["ml_output"].similarity_matches)
    wanted = (context["ml_output"].metadata.get("similarity_summary") or {}).get("requested_k")
    if wanted and found < wanted:
        remarks.append(f"<b>비슷한 과거 prefix</b> : {wanted}개를 찾으려 했지만 {found}개만 찾았다 → 예측을 더 조심해서 읽는다.")
    if not conditions.get("gpu_memory_gib"):
        remarks.append("<b>GPU</b> : 메모리를 0으로 넣어 GPU에서만 잰 후보를 뺐다.")
    if payload.get("tolerance"):
        remarks.append(f"<b>허용 폭 ε</b> : 예측 차이가 {payload['tolerance']:g} 이내인 경로 중 가장 싼 경로를 골랐다. "
                       "Dev18 백테스트에서는 0보다 크게 두면 실제 성능이 낮아졌다.")
    if payload["cost_unit"] == "KRW":
        remarks.append("<b>금액 환산</b> : 후보를 잰 장치(CPU·GPU)의 시간당 단가로 초를 원으로 바꿨다. 장치를 모르면 "
                       "비싼 쪽 단가를 썼다.")
    items = "".join(f"<li>{remark}</li>" for remark in remarks)
    return f"""{open_section("remarks", "계획은 아래 가정 위에서만 성립한다")}
<div class="remarks"><div class="vlabel" aria-hidden="true">비고</div><ol>{items}</ol></div></div></section>"""


def build_report_html(*, payload, result, ml_input, ml_output, generated_at):
    """화면과 A4 인쇄에 같이 쓰는 보고서 HTML 문자열."""
    conditions, eligible = ml_input["operating_conditions"], ml_input["candidates"]
    plan, reference, tight = recommend(payload, result)
    context = {
        "payload": payload, "result": plan, "ml_output": ml_output, "conditions": conditions,
        "input": ml_input["input"], "current": ml_input.get("current_model") or {},
        "features": ml_input.get("current_features"), "eligible": eligible,
        "excluded": ml_input.get("excluded_candidates", []),
        "candidates": {f'{row["config_id"]}::{row["head"]}': row for row in eligible},
        "target_uses": {f'{row["config_id"]}::{row["head"]}': row.get("target_use") for row in eligible},
        "stages": payload["stages"], "reference": reference, "tight": tight, "switched": plan is not result,
        "final_rows": planned_final_rows(ml_input["input"]["row_count"], conditions),
        "plan_no": f"P-{generated_at:%Y%m%d-%H%M}", "generated": f"{generated_at:%Y-%m-%d %H:%M} KST",
    }
    links = "".join(f'<a href="#{section_id}-title">{title}</a>' for section_id, title in SECTIONS)
    page = ("@page { @bottom-left { content: \"TSAD 운영 계획 보고서 · " + context["plan_no"] + "\"; "
            "font: 7.5pt \"Pretendard Variable\", \"Malgun Gothic\", sans-serif; color: #66717d; } "
            "@bottom-right { content: counter(page) \" / \" counter(pages); "
            "font: 7.5pt \"Pretendard Variable\", \"Malgun Gothic\", sans-serif; color: #66717d; } }")
    sections = "".join(render(context) for render in (render_conclusion, render_schedule, render_budget, render_inputs,
                                                      render_evidence, render_pool, render_remarks))
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>TSAD 운영 계획 보고서 {context['plan_no']}</title>
<link rel="stylesheet" href="{FONT}"><style>{STYLE}{page}</style></head>
<body>{MARKS}
<div class="runhead"><nav aria-label="보고서 목차">{links}</nav>
<button class="print" type="button" onclick="window.print()">인쇄 · PDF 저장</button></div>
<main class="sheet"><article class="form">{render_masthead(context)}{sections}</article>
<footer class="colophon"><span>이 보고서는 추천만 담는다. 모델 학습·배포 상태와 DB는 바꾸지 않는다.</span>
<span>근거 DB : TSB-AD-M Dev18 튜닝 패널(18개 파일·10개 family) · {context['plan_no']}</span></footer></main>
{SCRIPT}</body></html>"""
