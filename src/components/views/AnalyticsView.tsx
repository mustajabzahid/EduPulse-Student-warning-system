import { useMemo, useState } from 'react';
import { motion } from 'motion/react';
import { CleanedStudentRecord, NumericFeatureKey } from '../../types/student';
import { FEATURE_LABELS } from '../../ml/pipeline';
import { NUMERIC_FEATURE_KEYS } from '../../ml/cleaning';
import { buildCgpaGrowthTrend, buildCgpaThreeBucketTrend } from '../../utils/analyticsTrends';
import { PlotlyChart } from '../PlotlyChart';
import { EASE_OUT_EXPO, GlassMotionCard } from '../MotionPrimitives';

interface AnalyticsViewProps {
  trainingRows: CleanedStudentRecord[];
}

type TrendMetricMode = 'all_pct' | 'midterm' | 'quiz';

function mean(arr: number[]): number {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function pearsonCorrelation(xs: number[], ys: number[]): number {
  const n = xs.length;
  if (n < 2) return 0;
  const mx = mean(xs);
  const my = mean(ys);
  let num = 0;
  let dx2 = 0;
  let dy2 = 0;
  for (let i = 0; i < n; i += 1) {
    const dx = xs[i] - mx;
    const dy = ys[i] - my;
    num += dx * dy;
    dx2 += dx * dx;
    dy2 += dy * dy;
  }
  const denom = Math.sqrt(dx2 * dy2);
  return denom > 0 ? num / denom : 0;
}

export function AnalyticsView({ trainingRows }: AnalyticsViewProps) {
  const [trendMetric, setTrendMetric] = useState<TrendMetricMode>('all_pct');

  const passRows = useMemo(
    () => trainingRows.filter((r) => r.final_result === 'Pass'),
    [trainingRows]
  );
  const failRows = useMemo(
    () => trainingRows.filter((r) => r.final_result === 'Fail'),
    [trainingRows]
  );

  const threeBucketTrend = useMemo(() => buildCgpaThreeBucketTrend(trainingRows), [trainingRows]);
  const cgpaTrend = useMemo(() => buildCgpaGrowthTrend(trainingRows), [trainingRows]);

  // Requested 3-Bucket Plotly Line Chart: Average Quiz (/10) & Mid-Term Marks (/30) by prev_cgpa (1.0-2.0, 2.0-3.0, 3.0-4.0)
  const threeBucketChartData = useMemo(() => {
    const pts = threeBucketTrend.points;
    const xLabels = pts.map((p) => p.bandLabel);

    return [
      {
        type: 'scatter',
        mode: 'lines+markers+text',
        name: 'Avg Quiz Marks (/10)',
        x: xLabels,
        y: pts.map((p) => Number(p.quizMean.toFixed(2))),
        text: pts.map((p) => `${p.quizMean.toFixed(1)}/10`),
        textposition: 'top center',
        textfont: { size: 11, family: 'JetBrains Mono, monospace', color: '#34d399' },
        customdata: pts.map((p) => [p.count, p.passRatePct.toFixed(1)]),
        hovertemplate:
          'CGPA Bucket %{x}<br>Avg Quiz: <b>%{y:.2f}/10</b><br>Students: %{customdata[0]} · Pass Rate: %{customdata[1]}%<extra></extra>',
        line: { color: '#34d399', width: 3.2, shape: 'spline', smoothing: 1.05 },
        fill: 'tozeroy',
        fillcolor: 'rgba(52, 211, 153, 0.07)',
        marker: { size: 10, color: '#34d399', line: { color: '#060913', width: 1.5 } },
        yaxis: 'y',
      },
      {
        type: 'scatter',
        mode: 'lines+markers+text',
        name: 'Avg Midterm Marks (/30)',
        x: xLabels,
        y: pts.map((p) => Number(p.midMarksMean.toFixed(2))),
        text: pts.map((p) => `${p.midMarksMean.toFixed(1)}/30`),
        textposition: 'bottom center',
        textfont: { size: 11, family: 'JetBrains Mono, monospace', color: '#60a5fa' },
        customdata: pts.map((p) => [p.count, p.passRatePct.toFixed(1)]),
        hovertemplate:
          'CGPA Bucket %{x}<br>Avg Midterm: <b>%{y:.2f}/30</b><br>Students: %{customdata[0]} · Pass Rate: %{customdata[1]}%<extra></extra>',
        line: { color: '#60a5fa', width: 3.2, shape: 'spline', smoothing: 1.05 },
        fill: 'tozeroy',
        fillcolor: 'rgba(96, 165, 250, 0.06)',
        marker: { size: 10, color: '#60a5fa', symbol: 'diamond', line: { color: '#060913', width: 1.5 } },
        yaxis: 'y2',
      },
    ];
  }, [threeBucketTrend]);

  // Chart 1: Assessment Component Averages (% of Max Scale)
  const assessmentBarData = useMemo(() => {
    const specs: Array<{ key: NumericFeatureKey; max: number; label: string }> = [
      { key: 'quiz_avg', max: 10, label: 'Quiz (/10)' },
      { key: 'assignment_avg', max: 10, label: 'Assignment (/10)' },
      { key: 'mid_marks', max: 30, label: 'Midterm (/30)' },
      { key: 'lab_marks', max: 20, label: 'Lab (/20)' },
    ];
    const labels = specs.map((s) => s.label);

    return [
      {
        type: 'bar',
        name: 'Pass Mean',
        x: labels,
        y: specs.map((s) =>
          Number(((mean(passRows.map((r) => r[s.key])) / s.max) * 100).toFixed(1))
        ),
        text: specs.map(
          (s) => `${mean(passRows.map((r) => r[s.key])).toFixed(1)}/${s.max}`
        ),
        textposition: 'outside',
        textangle: 0,
        textfont: { size: 10.5, family: 'JetBrains Mono, monospace', color: '#e2e8f0' },
        marker: { color: '#34d399' },
      },
      {
        type: 'bar',
        name: 'Fail Mean',
        x: labels,
        y: specs.map((s) =>
          Number(((mean(failRows.map((r) => r[s.key])) / s.max) * 100).toFixed(1))
        ),
        text: specs.map(
          (s) => `${mean(failRows.map((r) => r[s.key])).toFixed(1)}/${s.max}`
        ),
        textposition: 'outside',
        textangle: 0,
        textfont: { size: 10.5, family: 'JetBrains Mono, monospace', color: '#fda4af' },
        marker: { color: '#fb7185' },
      },
    ];
  }, [passRows, failRows]);

  // Chart 2: Attendance vs Midterm Marks Risk Quadrant
  const attVsMidData = useMemo(
    () => [
      {
        type: 'scatter',
        mode: 'markers',
        name: 'Pass',
        x: passRows.map((r) => r.attendance_pct),
        y: passRows.map((r) => r.mid_marks),
        text: passRows.map((r) => `${r.student_id} · CGPA ${r.prev_cgpa.toFixed(2)}`),
        marker: { color: '#34d399', size: 7, opacity: 0.72 },
      },
      {
        type: 'scatter',
        mode: 'markers',
        name: 'Fail',
        x: failRows.map((r) => r.attendance_pct),
        y: failRows.map((r) => r.mid_marks),
        text: failRows.map((r) => `${r.student_id} · CGPA ${r.prev_cgpa.toFixed(2)}`),
        marker: { color: '#fb7185', size: 8, opacity: 0.9, symbol: 'diamond' },
      },
    ],
    [passRows, failRows]
  );

  // Chart 3: Live Pearson Correlation Matrix Heatmap
  const correlationHeatmapData = useMemo(() => {
    const keys: Array<NumericFeatureKey | 'fail_flag'> = [
      'attendance_pct',
      'mid_marks',
      'quiz_avg',
      'assignment_avg',
      'lab_marks',
      'study_hours_week',
      'prev_cgpa',
      'fail_flag',
    ];
    const shortLabels: Record<NumericFeatureKey | 'fail_flag', string> = {
      attendance_pct: 'Attendance',
      mid_marks: 'Midterm',
      quiz_avg: 'Quiz',
      assignment_avg: 'Assign.',
      lab_marks: 'Lab',
      study_hours_week: 'Study Hrs',
      prev_cgpa: 'Prev CGPA',
      fail_flag: 'Fail Outcome',
    };

    const vectors = keys.map((k) =>
      trainingRows.map((r) => (k === 'fail_flag' ? (r.final_result === 'Fail' ? 1 : 0) : r[k]))
    );

    const z: number[][] = [];
    const text: string[][] = [];
    for (let i = 0; i < keys.length; i += 1) {
      const rowZ: number[] = [];
      const rowText: string[] = [];
      for (let j = 0; j < keys.length; j += 1) {
        const r = pearsonCorrelation(vectors[i], vectors[j]);
        rowZ.push(Number(r.toFixed(2)));
        rowText.push(r.toFixed(2));
      }
      z.push(rowZ);
      text.push(rowText);
    }

    const labels = keys.map((k) => shortLabels[k]);
    return [
      {
        type: 'heatmap',
        z,
        x: labels,
        y: labels,
        text,
        texttemplate: '%{text}',
        textfont: { size: 10.5, family: 'JetBrains Mono, monospace', color: '#f8fafc' },
        xgap: 5,
        ygap: 5,
        zmin: -1,
        zmax: 1,
        colorscale: [
          [0, '#0f766e'],
          [0.5, '#1e293b'],
          [1, '#e11d48'],
        ],
        colorbar: {
          title: { text: 'Pearson r', side: 'right', font: { size: 11, color: '#cbd5e1' } },
          tickfont: { color: '#cbd5e1' },
          thickness: 12,
          len: 0.9,
        },
        hovertemplate: '%{y} vs %{x}<br>Pearson r: %{z:.2f}<extra></extra>',
      },
    ];
  }, [trainingRows]);

  // Chart 4: Study Hours vs Previous CGPA Scatter
  const scatterData = useMemo(
    () => [
      {
        type: 'scatter',
        mode: 'markers',
        name: 'Pass',
        x: passRows.map((r) => r.study_hours_week),
        y: passRows.map((r) => r.prev_cgpa),
        text: passRows.map(
          (r) => `${r.student_id} · Attendance: ${r.attendance_pct.toFixed(0)}%`
        ),
        marker: { color: '#34d399', size: 7, opacity: 0.72 },
      },
      {
        type: 'scatter',
        mode: 'markers',
        name: 'Fail',
        x: failRows.map((r) => r.study_hours_week),
        y: failRows.map((r) => r.prev_cgpa),
        text: failRows.map(
          (r) => `${r.student_id} · Attendance: ${r.attendance_pct.toFixed(0)}%`
        ),
        marker: { color: '#fb7185', size: 8, opacity: 0.9, symbol: 'diamond' },
      },
    ],
    [passRows, failRows]
  );

  // Granular 7-Band Ordinal Cohort Growth Trend
  const trendPlotData = useMemo(() => {
    const pts = cgpaTrend.points;
    const xLabels = pts.map((p) => p.bandLabel);

    if (trendMetric === 'midterm') {
      return [
        {
          type: 'scatter',
          mode: 'lines+markers+text',
          name: 'Mean Midterm (/30)',
          x: xLabels,
          y: pts.map((p) => Number(p.midMarksMean.toFixed(1))),
          text: pts.map((p) => `${p.midMarksMean.toFixed(1)}`),
          textposition: 'top center',
          textfont: { size: 10.5, family: 'JetBrains Mono, monospace', color: '#93c5fd' },
          customdata: pts.map((p) => [p.count, p.passRatePct.toFixed(1)]),
          hovertemplate:
            'CGPA %{x}<br>Midterm Mean: <b>%{y:.1f}/30</b><br>Students: %{customdata[0]} · Pass Rate: %{customdata[1]}%<extra></extra>',
          line: { color: '#60a5fa', width: 3.2, shape: 'spline', smoothing: 1.1 },
          fill: 'tozeroy',
          fillcolor: 'rgba(96, 165, 250, 0.08)',
          marker: { size: 8, color: '#34d399', line: { color: '#060913', width: 1.5 } },
        },
      ];
    }

    if (trendMetric === 'quiz') {
      return [
        {
          type: 'scatter',
          mode: 'lines+markers+text',
          name: 'Mean Quiz Avg (/10)',
          x: xLabels,
          y: pts.map((p) => Number(p.quizMean.toFixed(2))),
          text: pts.map((p) => `${p.quizMean.toFixed(1)}`),
          textposition: 'top center',
          textfont: { size: 10.5, family: 'JetBrains Mono, monospace', color: '#6ee7b7' },
          customdata: pts.map((p) => [p.count, p.passRatePct.toFixed(1)]),
          hovertemplate:
            'CGPA %{x}<br>Quiz Avg Mean: <b>%{y:.2f}/10</b><br>Students: %{customdata[0]} · Pass Rate: %{customdata[1]}%<extra></extra>',
          line: { color: '#34d399', width: 3.2, shape: 'spline', smoothing: 1.1 },
          fill: 'tozeroy',
          fillcolor: 'rgba(52, 211, 153, 0.08)',
          marker: { size: 8, color: '#ffffff', line: { color: '#060913', width: 1.5 } },
        },
      ];
    }

    return [
      {
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Midterm (/30)',
        x: xLabels,
        y: pts.map((p) => Number(p.midMarksPct.toFixed(1))),
        customdata: pts.map((p) => [p.midMarksMean.toFixed(1), p.count]),
        hovertemplate:
          'CGPA %{x}<br>Midterm: <b>%{customdata[0]}/30</b> (%{y:.1f}%)<br>n = %{customdata[1]}<extra></extra>',
        line: { color: '#ffffff', width: 3, shape: 'spline', smoothing: 1.1 },
        marker: { size: 7, color: '#ffffff' },
      },
      {
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Quiz Avg (/10)',
        x: xLabels,
        y: pts.map((p) => Number(p.quizPct.toFixed(1))),
        customdata: pts.map((p) => [p.quizMean.toFixed(1), p.count]),
        hovertemplate:
          'CGPA %{x}<br>Quiz Avg: <b>%{customdata[0]}/10</b> (%{y:.1f}%)<br>n = %{customdata[1]}<extra></extra>',
        line: { color: '#34d399', width: 2.6, shape: 'spline', smoothing: 1.1 },
        marker: { size: 7, color: '#34d399' },
      },
      {
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Assignment Avg (/10)',
        x: xLabels,
        y: pts.map((p) => Number(p.assignmentPct.toFixed(1))),
        customdata: pts.map((p) => [p.assignmentMean.toFixed(1), p.count]),
        hovertemplate:
          'CGPA %{x}<br>Assignment Avg: <b>%{customdata[0]}/10</b> (%{y:.1f}%)<br>n = %{customdata[1]}<extra></extra>',
        line: { color: '#60a5fa', width: 2.2, dash: 'dash', shape: 'spline', smoothing: 1.1 },
        marker: { size: 6, color: '#60a5fa' },
      },
      {
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Lab Marks (/20)',
        x: xLabels,
        y: pts.map((p) => Number(p.labMarksPct.toFixed(1))),
        customdata: pts.map((p) => [p.labMarksMean.toFixed(1), p.count]),
        hovertemplate:
          'CGPA %{x}<br>Lab Marks: <b>%{customdata[0]}/20</b> (%{y:.1f}%)<br>n = %{customdata[1]}<extra></extra>',
        line: { color: '#fbbf24', width: 2.2, dash: 'dot', shape: 'spline', smoothing: 1.1 },
        marker: { size: 6, color: '#fbbf24' },
      },
    ];
  }, [cgpaTrend, trendMetric]);

  return (
    <div className="space-y-6">
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.24, ease: EASE_OUT_EXPO }}
      >
        <p className="text-sm text-slate-300">
          Multivariate comparison of assessment mastery, risk quadrant boundaries, feature
          correlations, and cohort-wide growth across prior CGPA buckets.
        </p>
      </motion.div>

      {/* Requested New Plotly Line Chart: Average Quiz & Mid-Term Marks by prev_cgpa Bucket (1.0-2.0, 2.0-3.0, 3.0-4.0) */}
      <GlassMotionCard as="section" variant="glass-strong" delay={0.04} className="p-6">
        <div className="flex flex-col gap-2 border-b border-white/10 pb-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="font-display text-base font-semibold text-white">
              Average Quiz &amp; Mid-Term Marks by Previous CGPA Bucket
            </h2>
            <p className="mt-0.5 text-xs text-slate-400">
              Cohort-wide academic growth aggregated across ordinal prev_cgpa buckets (1.0–2.0,
              2.0–3.0, 3.0–4.0).
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-4 font-mono text-xs tabular-nums">
            <span className="text-emerald-300">
              Quiz {threeBucketTrend.quizGrowthRaw >= 0 ? '+' : ''}
              {threeBucketTrend.quizGrowthRaw.toFixed(1)} /10
            </span>
            <span className="text-blue-300">
              Midterm {threeBucketTrend.midGrowthRaw >= 0 ? '+' : ''}
              {threeBucketTrend.midGrowthRaw.toFixed(1)} /30
            </span>
          </div>
        </div>

        <div className="mt-3">
          <PlotlyChart
            data={threeBucketChartData}
            layout={{
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              margin: { t: 44, r: 64, b: 52, l: 60 },
              xaxis: {
                title: { text: 'Previous CGPA Bucket (Ordinal Cohort Tiers)' },
                type: 'category',
                range: [-0.35, Math.max(0, threeBucketTrend.points.length - 1) + 0.35],
                gridcolor: 'rgba(255,255,255,0.045)',
                zeroline: false,
              },
              yaxis: {
                title: { text: 'Avg Quiz Marks (/10)', font: { color: '#34d399' } },
                range: [0, 10.5],
                gridcolor: 'rgba(255,255,255,0.045)',
                zeroline: false,
                tickfont: { color: '#6ee7b7' },
              },
              yaxis2: {
                title: { text: 'Avg Midterm Marks (/30)', font: { color: '#60a5fa' } },
                range: [0, 31.5],
                overlaying: 'y',
                side: 'right',
                showgrid: false,
                zeroline: false,
                tickfont: { color: '#93c5fd' },
              },
              legend: { orientation: 'h', y: 1.15, x: 0 },
            }}
            height={310}
            reportTitle="Average Quiz and Mid-Term Marks by Previous CGPA Bucket (1.0-2.0, 2.0-3.0, 3.0-4.0)"
            reportSubtitle={`Cohort growth: Quiz +${threeBucketTrend.quizGrowthRaw.toFixed(1)}/10 and Midterm +${threeBucketTrend.midGrowthRaw.toFixed(1)}/30`}
            ariaLabel="Dual-axis Plotly line chart showing average quiz and mid-term marks across prev_cgpa buckets 1.0-2.0, 2.0-3.0, and 3.0-4.0"
          />
        </div>
      </GlassMotionCard>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        <GlassMotionCard delay={0.08} className="p-6">
          <div className="flex items-baseline justify-between gap-2">
            <h2 className="font-display text-base font-semibold text-white">
              Assessment Score Averages (Pass vs Fail)
            </h2>
            <span className="font-mono text-[11px] text-amber-300 tabular-nums">
              Benchmark: 60%
            </span>
          </div>
          <p className="mt-0.5 text-xs text-slate-400">
            Mean scores normalized as % of component maximum with the 60% advising threshold
            (6/10, 18/30, 12/20).
          </p>
          <PlotlyChart
            data={assessmentBarData}
            layout={{
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              barmode: 'group',
              barcornerradius: 16,
              bargap: 0.28,
              bargroupgap: 0.16,
              margin: { t: 42, r: 18, b: 46, l: 54 },
              xaxis: { showgrid: false },
              yaxis: {
                title: { text: 'Mean (% of Max Marks)' },
                range: [0, 114],
                gridcolor: 'rgba(255,255,255,0.045)',
                zeroline: false,
              },
              legend: { orientation: 'h', y: 1.14, x: 0 },
              shapes: [
                {
                  type: 'line',
                  xref: 'paper',
                  x0: 0,
                  x1: 1,
                  y0: 60,
                  y1: 60,
                  line: { color: '#fbbf24', width: 1.5, dash: 'dash' },
                },
              ],
            }}
            height={320}
            reportTitle="Assessment Score Averages (Pass vs Fail)"
            reportSubtitle="Mean scores normalized as % of component maximum with 60% benchmark"
            ariaLabel="Grouped bar chart comparing Pass and Fail mean assessment scores"
          />
        </GlassMotionCard>

        <GlassMotionCard delay={0.12} className="p-6">
          <div className="flex items-baseline justify-between gap-2">
            <h2 className="font-display text-base font-semibold text-white">
              Attendance × Midterm Risk Quadrant
            </h2>
            <span className="font-mono text-[11px] text-rose-300 tabular-nums">
              Dual-risk: &lt;75% &amp; &lt;18/30
            </span>
          </div>
          <p className="mt-0.5 text-xs text-slate-400">
            Shaded bottom-left quadrant isolates students below both 75% attendance and 18/30
            midterm marks.
          </p>
          <PlotlyChart
            data={attVsMidData}
            layout={{
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              margin: { t: 42, r: 18, b: 46, l: 54 },
              xaxis: {
                title: { text: 'Attendance (%)' },
                gridcolor: 'rgba(255,255,255,0.045)',
                zeroline: false,
                range: [28, 102],
              },
              yaxis: {
                title: { text: 'Midterm Marks (/30)' },
                gridcolor: 'rgba(255,255,255,0.045)',
                zeroline: false,
                range: [0, 31],
              },
              legend: { orientation: 'h', y: 1.14, x: 0 },
              shapes: [
                {
                  type: 'rect',
                  x0: 28,
                  x1: 75,
                  y0: 0,
                  y1: 18,
                  fillcolor: 'rgba(251, 113, 133, 0.09)',
                  line: { width: 0 },
                  layer: 'below',
                },
                {
                  type: 'line',
                  x0: 75,
                  x1: 75,
                  y0: 0,
                  y1: 31,
                  line: { color: '#fbbf24', width: 1.2, dash: 'dot' },
                },
                {
                  type: 'line',
                  x0: 28,
                  x1: 102,
                  y0: 18,
                  y1: 18,
                  line: { color: '#fbbf24', width: 1.2, dash: 'dot' },
                },
              ],
            }}
            height={320}
            reportTitle="Attendance x Midterm Risk Quadrant"
            reportSubtitle="Dual-risk zone isolates students below 75% attendance and 18/30 midterm"
            ariaLabel="Scatter plot of attendance percentage versus midterm marks with shaded risk quadrant"
          />
        </GlassMotionCard>

        <GlassMotionCard delay={0.16} className="p-6">
          <div className="flex items-baseline justify-between gap-2">
            <h2 className="font-display text-base font-semibold text-white">
              Academic Feature Correlation Matrix
            </h2>
            <span className="font-mono text-[11px] text-slate-300 tabular-nums">
              Pearson r (−1 to +1)
            </span>
          </div>
          <p className="mt-0.5 text-xs text-slate-400">
            Pairwise linear correlations computed directly on the cleaned cohort (correlation only).
          </p>
          <PlotlyChart
            data={correlationHeatmapData}
            layout={{
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              margin: { t: 16, r: 24, b: 78, l: 88 },
              yaxis: { autorange: 'reversed', automargin: true, showgrid: false },
              xaxis: { tickangle: -28, automargin: true, showgrid: false },
            }}
            height={340}
            reportTitle="Academic Feature Correlation Matrix"
            reportSubtitle="Pairwise Pearson correlations across indicators and Fail outcome"
            ariaLabel="Correlation heatmap across academic indicators and final result"
          />
        </GlassMotionCard>

        <GlassMotionCard delay={0.2} className="p-6">
          <div className="flex items-baseline justify-between gap-2">
            <h2 className="font-display text-base font-semibold text-white">
              Study Hours vs Previous CGPA
            </h2>
            <span className="font-mono text-[11px] text-amber-300 tabular-nums">
              Cutoffs: 10h · 2.40
            </span>
          </div>
          <p className="mt-0.5 text-xs text-slate-400">
            Weekly independent study hours plotted against prior CGPA with early-warning cutoffs.
          </p>
          <PlotlyChart
            data={scatterData}
            layout={{
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              margin: { t: 42, r: 18, b: 46, l: 54 },
              xaxis: {
                title: { text: 'Study Hours / Week' },
                gridcolor: 'rgba(255,255,255,0.045)',
                zeroline: false,
              },
              yaxis: {
                title: { text: 'Previous CGPA' },
                gridcolor: 'rgba(255,255,255,0.045)',
                zeroline: false,
                range: [0.9, 4.1],
              },
              legend: { orientation: 'h', y: 1.14, x: 0 },
              shapes: [
                {
                  type: 'rect',
                  x0: 0,
                  x1: 10,
                  y0: 0.9,
                  y1: 2.4,
                  fillcolor: 'rgba(251, 113, 133, 0.09)',
                  line: { width: 0 },
                  layer: 'below',
                },
                {
                  type: 'line',
                  x0: 10,
                  x1: 10,
                  y0: 0.9,
                  y1: 4.1,
                  line: { color: '#fbbf24', width: 1.2, dash: 'dot' },
                },
                {
                  type: 'line',
                  yref: 'y',
                  xref: 'paper',
                  x0: 0,
                  x1: 1,
                  y0: 2.4,
                  y1: 2.4,
                  line: { color: '#fbbf24', width: 1.2, dash: 'dot' },
                },
              ],
            }}
            height={330}
            reportTitle="Study Hours vs Previous CGPA"
            reportSubtitle="Weekly study engagement vs prior CGPA (cutoffs: 10h/wk and 2.40)"
            ariaLabel="Scatter plot of weekly study hours versus previous CGPA"
          />
        </GlassMotionCard>
      </div>

      {/* Granular 7-Band Ordinal Cohort Performance Growth Trend */}
      <GlassMotionCard as="section" delay={0.24} className="p-6">
        <div className="flex flex-col gap-3 border-b border-white/10 pb-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <div className="flex flex-wrap items-baseline gap-2.5">
              <h2 className="font-display text-base font-semibold text-white">
                Granular Cohort Performance Trend (7 Prior CGPA Tiers)
              </h2>
              <span className="font-mono text-[11px] font-medium text-emerald-300 tabular-nums">
                Midterm {cgpaTrend.midGrowthPctPoints >= 0 ? '+' : ''}
                {cgpaTrend.midGrowthPctPoints.toFixed(1)} pp · Quiz{' '}
                {cgpaTrend.quizGrowthPctPoints >= 0 ? '+' : ''}
                {cgpaTrend.quizGrowthPctPoints.toFixed(1)} pp ({cgpaTrend.lowestBandLabel} →{' '}
                {cgpaTrend.highestBandLabel})
              </span>
            </div>
            <p className="mt-0.5 text-xs text-slate-400">
              Detailed progression aligned at the 2.40 CGPA advising cutoff against the 60% mastery
              benchmark.
            </p>
          </div>

          <div
            role="group"
            aria-label="Select trend metric"
            className="glass flex flex-wrap items-center gap-1 rounded-xl p-1 shrink-0"
          >
            <button
              type="button"
              onClick={() => setTrendMetric('all_pct')}
              className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
                trendMetric === 'all_pct'
                  ? 'bg-white text-slate-950 shadow-xs'
                  : 'text-slate-300 hover:text-white'
              }`}
            >
              All Metrics (% Scale)
            </button>
            <button
              type="button"
              onClick={() => setTrendMetric('midterm')}
              className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
                trendMetric === 'midterm'
                  ? 'bg-white text-slate-950 shadow-xs'
                  : 'text-slate-300 hover:text-white'
              }`}
            >
              Midterm (/30)
            </button>
            <button
              type="button"
              onClick={() => setTrendMetric('quiz')}
              className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
                trendMetric === 'quiz'
                  ? 'bg-white text-slate-950 shadow-xs'
                  : 'text-slate-300 hover:text-white'
              }`}
            >
              Quiz Avg (/10)
            </button>
          </div>
        </div>

        <div className="mt-3">
          <PlotlyChart
            data={trendPlotData}
            layout={{
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              margin: { t: 44, r: 24, b: 54, l: 58 },
              xaxis: {
                title: { text: 'Previous CGPA Band (Ordinal Standing Progression)' },
                type: 'category',
                range: [-0.35, Math.max(0, cgpaTrend.points.length - 1) + 0.35],
                gridcolor: 'rgba(255,255,255,0.045)',
                zeroline: false,
              },
              yaxis: {
                title: {
                  text:
                    trendMetric === 'midterm'
                      ? 'Mean Midterm Marks (/30)'
                      : trendMetric === 'quiz'
                      ? 'Mean Quiz Average (/10)'
                      : 'Cohort Mean (% of Max Score)',
                },
                range:
                  trendMetric === 'midterm'
                    ? [0, 31.5]
                    : trendMetric === 'quiz'
                    ? [0, 10.5]
                    : [20, 104],
                gridcolor: 'rgba(255,255,255,0.08)',
              },
              legend: { orientation: 'h', y: 1.15, x: 0 },
              shapes: [
                {
                  type: 'line',
                  xref: 'paper',
                  x0: 0,
                  x1: 1,
                  y0: trendMetric === 'midterm' ? 18 : trendMetric === 'quiz' ? 6 : 60,
                  y1: trendMetric === 'midterm' ? 18 : trendMetric === 'quiz' ? 6 : 60,
                  line: { color: '#fbbf24', width: 1.5, dash: 'dash' },
                },
              ],
              annotations: [
                {
                  xref: 'paper',
                  x: 0.99,
                  y: trendMetric === 'midterm' ? 18 : trendMetric === 'quiz' ? 6 : 60,
                  text:
                    trendMetric === 'midterm'
                      ? '18/30 benchmark'
                      : trendMetric === 'quiz'
                      ? '6.0/10 benchmark'
                      : '60% benchmark',
                  showarrow: false,
                  xanchor: 'right',
                  yanchor: 'bottom',
                  yshift: 3,
                  bgcolor: 'rgba(15, 23, 42, 0.85)',
                  borderpad: 2,
                  font: { size: 10.5, color: '#fcd34d' },
                },
              ],
            }}
            height={320}
            reportTitle="Granular Cohort Performance Trend (7 Prior CGPA Tiers)"
            reportSubtitle={
              trendMetric === 'midterm'
                ? 'Mean Midterm Marks (/30) across ordered prev_cgpa tiers'
                : trendMetric === 'quiz'
                ? 'Mean Quiz Average (/10) across ordered prev_cgpa tiers'
                : `Normalized assessment growth (+${cgpaTrend.midGrowthPctPoints.toFixed(1)} pp Midterm, +${cgpaTrend.quizGrowthPctPoints.toFixed(1)} pp Quiz)`
            }
            ariaLabel="Trend line chart showing cohort assessment performance growth across ordered previous CGPA tiers"
          />
        </div>
      </GlassMotionCard>

      {/* Cohort Statistical Summary Table */}
      <GlassMotionCard as="section" delay={0.28} className="p-6">
        <h2 className="font-display text-base font-semibold text-white">
          Pass vs Fail Cohort Feature Benchmarks
        </h2>
        <p className="mt-0.5 text-xs text-slate-400">
          Observed cohort means and gaps across numerical features (descriptive statistics only).
        </p>

        <div className="mt-4 overflow-x-auto">
          <table className="w-full border-collapse text-left text-xs">
            <thead>
              <tr className="border-b border-white/10 text-slate-400">
                <th className="py-2 pr-4 font-medium">Academic Feature</th>
                <th className="py-2 px-4 text-right font-medium">Pass Mean</th>
                <th className="py-2 px-4 text-right font-medium">Fail Mean</th>
                <th className="py-2 pl-4 text-right font-medium">Mean Gap (Pass − Fail)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.06]">
              {NUMERIC_FEATURE_KEYS.map((key) => {
                const pMean = mean(passRows.map((r) => r[key]));
                const fMean = mean(failRows.map((r) => r[key]));
                const decimals = key === 'prev_cgpa' ? 2 : 1;
                return (
                  <tr key={key} className="hover:bg-white/[0.025]">
                    <td className="py-2.5 pr-4 font-medium text-slate-100">
                      {FEATURE_LABELS[key]}
                    </td>
                    <td className="py-2.5 px-4 text-right font-mono text-emerald-300 tabular-nums">
                      {pMean.toFixed(decimals)}
                    </td>
                    <td className="py-2.5 px-4 text-right font-mono text-rose-300 tabular-nums">
                      {fMean.toFixed(decimals)}
                    </td>
                    <td className="py-2.5 pl-4 text-right font-mono font-semibold text-white tabular-nums">
                      +{(pMean - fMean).toFixed(decimals)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </GlassMotionCard>
    </div>
  );
}
