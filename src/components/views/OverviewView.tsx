import { useMemo } from 'react';
import { motion } from 'motion/react';
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Database,
  ShieldAlert,
  Sparkles,
} from 'lucide-react';
import {
  CleanedStudentRecord,
  DataCleaningSummary,
  ModelType,
  ScoredStudentRecord,
  TrainedModelBundle,
} from '../../types/student';
import { buildCgpaGrowthTrend } from '../../utils/analyticsTrends';
import { NavSection } from '../Sidebar';
import { PlotlyChart } from '../PlotlyChart';
import { EASE_OUT_EXPO, GlassMotionCard } from '../MotionPrimitives';

interface OverviewViewProps {
  summary: DataCleaningSummary;
  trainingRows: CleanedStudentRecord[];
  scoredCohort: ScoredStudentRecord[];
  bundle: TrainedModelBundle;
  activeModel: ModelType;
  warnings: string[];
  onNavigate: (section: NavSection) => void;
}

function buildSmoothPath(values: number[], width: number, height: number, padY = 12): string {
  if (values.length === 0) return `M 0 ${height / 2} L ${width} ${height / 2}`;
  if (values.length === 1) return `M 0 ${height / 2} L ${width} ${height / 2}`;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(1e-6, max - min);
  const usableH = height - padY * 2;
  const pts = values.map((v, idx) => ({
    x: (idx / (values.length - 1)) * width,
    y: padY + usableH - ((v - min) / span) * usableH,
  }));

  let d = `M ${pts[0].x.toFixed(1)} ${pts[0].y.toFixed(1)}`;
  for (let i = 0; i < pts.length - 1; i += 1) {
    const p0 = pts[i];
    const p1 = pts[i + 1];
    const cx = ((p0.x + p1.x) / 2).toFixed(1);
    d += ` C ${cx} ${p0.y.toFixed(1)}, ${cx} ${p1.y.toFixed(1)}, ${p1.x.toFixed(1)} ${p1.y.toFixed(1)}`;
  }
  return d;
}

export function OverviewView({
  summary,
  trainingRows,
  scoredCohort,
  bundle,
  activeModel,
  warnings,
  onNavigate,
}: OverviewViewProps) {
  const getRiskProb = (s: ScoredStudentRecord) =>
    activeModel === 'logistic_regression' ? s.lrFailProbability : s.rfFailProbability;

  const activeMetrics =
    activeModel === 'logistic_regression' ? bundle.lrMetrics : bundle.rfMetrics;

  const highRiskCount = useMemo(
    () => scoredCohort.filter((s) => getRiskProb(s) >= 0.65).length,
    [scoredCohort, activeModel]
  );
  const mediumRiskCount = useMemo(
    () =>
      scoredCohort.filter((s) => {
        const p = getRiskProb(s);
        return p >= 0.35 && p < 0.65;
      }).length,
    [scoredCohort, activeModel]
  );
  const lowRiskCount = useMemo(
    () => scoredCohort.filter((s) => getRiskProb(s) < 0.35).length,
    [scoredCohort, activeModel]
  );

  const totalScored = Math.max(1, scoredCohort.length);
  const passRate =
    summary.cleanedTrainingRows > 0
      ? (summary.passCount / summary.cleanedTrainingRows) * 100
      : 0;
  const failRate = 100 - passRate;
  const highRiskShare = (highRiskCount / totalScored) * 100;
  const retentionCleanRatio =
    summary.originalRows > 0 ? (summary.cleanedTrainingRows / summary.originalRows) * 100 : 100;

  const cgpaTrend = useMemo(() => buildCgpaGrowthTrend(trainingRows), [trainingRows]);

  const heroCurvePath = useMemo(() => {
    const vals =
      cgpaTrend.points.length >= 2
        ? cgpaTrend.points.map((p) => p.passRatePct)
        : [22, 38, 55, 74, 88, 96];
    return buildSmoothPath(vals, 420, 130, 14);
  }, [cgpaTrend]);

  const sparkPaths = useMemo(() => {
    const pts = cgpaTrend.points;
    const failVals =
      pts.length >= 2 ? pts.map((p) => 100 - p.passRatePct) : [78, 62, 44, 24, 10, 4];
    const recallVals =
      pts.length >= 2 ? pts.map((p) => p.midMarksPct) : [25, 42, 58, 69, 80, 91];
    const passVals =
      pts.length >= 2 ? pts.map((p) => p.passRatePct) : [22, 38, 56, 76, 90, 96];
    return {
      roseLine: buildSmoothPath(failVals, 240, 56, 6),
      blueLine: buildSmoothPath(recallVals, 240, 56, 6),
      emeraldLine: buildSmoothPath(passVals, 240, 56, 6),
    };
  }, [cgpaTrend]);

  // Chart 1: Attendance vs Final Result
  const attendanceChartData = useMemo(() => {
    const passAtt = trainingRows
      .filter((r) => r.final_result === 'Pass')
      .map((r) => r.attendance_pct);
    const failAtt = trainingRows
      .filter((r) => r.final_result === 'Fail')
      .map((r) => r.attendance_pct);

    return [
      {
        type: 'box',
        y: passAtt,
        name: 'Pass Cohort',
        marker: { color: '#34d399', size: 4, opacity: 0.6 },
        fillcolor: 'rgba(52, 211, 153, 0.16)',
        line: { color: '#34d399', width: 2 },
        boxmean: true,
        boxpoints: 'all',
        jitter: 0.35,
        pointpos: -1.65,
      },
      {
        type: 'box',
        y: failAtt,
        name: 'Fail Cohort',
        marker: { color: '#fb7185', size: 4, opacity: 0.65 },
        fillcolor: 'rgba(251, 113, 133, 0.16)',
        line: { color: '#fb7185', width: 2 },
        boxmean: true,
        boxpoints: 'all',
        jitter: 0.35,
        pointpos: -1.65,
      },
    ];
  }, [trainingRows]);

  // Chart 2: Previous CGPA Distribution
  const cgpaChartData = useMemo(() => {
    const passCgpa = trainingRows
      .filter((r) => r.final_result === 'Pass')
      .map((r) => r.prev_cgpa);
    const failCgpa = trainingRows
      .filter((r) => r.final_result === 'Fail')
      .map((r) => r.prev_cgpa);

    return [
      {
        type: 'histogram',
        x: passCgpa,
        name: 'Pass',
        opacity: 0.72,
        marker: { color: '#14b8a6', line: { color: 'rgba(255,255,255,0.2)', width: 0.6 } },
        xbins: { start: 1.0, end: 4.05, size: 0.2 },
      },
      {
        type: 'histogram',
        x: failCgpa,
        name: 'Fail',
        opacity: 0.8,
        marker: { color: '#fb7185', line: { color: 'rgba(255,255,255,0.2)', width: 0.6 } },
        xbins: { start: 1.0, end: 4.05, size: 0.2 },
      },
    ];
  }, [trainingRows]);

  // Chart 3: Part-Time Job vs Final Result
  const { partTimeChartData, jobFailRates } = useMemo(() => {
    const categories: Array<'No' | 'Yes'> = ['No', 'Yes'];
    const passCounts = categories.map(
      (cat) => trainingRows.filter((r) => r.part_time_job === cat && r.final_result === 'Pass').length
    );
    const failCounts = categories.map(
      (cat) => trainingRows.filter((r) => r.part_time_job === cat && r.final_result === 'Fail').length
    );
    const rates = categories.map((_, i) => {
      const tot = passCounts[i] + failCounts[i];
      return tot > 0 ? ((failCounts[i] / tot) * 100).toFixed(1) : '0.0';
    });

    return {
      jobFailRates: { no: rates[0], yes: rates[1] },
      partTimeChartData: [
        {
          type: 'bar',
          x: ['No Part-Time Job', 'Has Part-Time Job'],
          y: passCounts,
          name: 'Pass',
          marker: { color: '#14b8a6' },
          text: passCounts.map(String),
          textposition: 'auto',
          textfont: { color: '#f8fafc' },
        },
        {
          type: 'bar',
          x: ['No Part-Time Job', 'Has Part-Time Job'],
          y: failCounts,
          name: 'Fail',
          marker: { color: '#fb7185' },
          text: failCounts.map((c, idx) => `${c} (${rates[idx]}%)`),
          textposition: 'auto',
          textfont: { color: '#f8fafc' },
        },
      ],
    };
  }, [trainingRows]);

  // Chart 4: Risk Tier Distribution
  const riskDistData = useMemo(() => {
    const labels = ['High Risk (≥65%)', 'Medium Risk (35–65%)', 'Low Risk (<35%)'];
    const counts = [highRiskCount, mediumRiskCount, lowRiskCount];
    const shares = counts.map((c) => `${c} (${((c / totalScored) * 100).toFixed(1)}%)`);

    return [
      {
        type: 'bar',
        orientation: 'h',
        y: labels,
        x: counts,
        marker: {
          color: ['#fb7185', '#fbbf24', '#34d399'],
          cornerradius: 16,
        },
        text: shares,
        textposition: 'auto',
        insidetextfont: { color: '#0f172a', family: 'JetBrains Mono, monospace', size: 11 },
        outsidetextfont: { color: '#f1f5f9', family: 'JetBrains Mono, monospace', size: 11 },
      },
    ];
  }, [lowRiskCount, mediumRiskCount, highRiskCount, totalScored]);

  const topActivityRows = useMemo(() => {
    const highestRiskStudents = [...scoredCohort]
      .sort((a, b) => getRiskProb(b) - getRiskProb(a))
      .slice(0, 3);

    return [
      ...highestRiskStudents.map((s, idx) => ({
        id: s.student_id,
        title: `Priority Intervention: ${s.student_id}`,
        subtitle: `Attendance ${s.attendance_pct.toFixed(0)}% · Midterm ${s.mid_marks.toFixed(1)}/30 · CGPA ${s.prev_cgpa.toFixed(2)}`,
        value: `${(getRiskProb(s) * 100).toFixed(1)}% P(Fail)`,
        tone: (idx === 0 ? 'amber' : 'blue') as 'blue' | 'emerald' | 'amber',
        icon: idx === 0 ? ShieldAlert : Activity,
      })),
      {
        id: 'audit-cleaning',
        title: `Pipeline Cleaning & Imputation Complete`,
        subtitle: `${summary.duplicateRowsRemoved} duplicates removed · ${summary.totalMissingCells} cells imputed · ${summary.missingFinalResultExcluded} missing targets dropped`,
        value: `${summary.cleanedTrainingRows} Clean`,
        tone: 'emerald' as const,
        icon: Sparkles,
      },
    ];
  }, [scoredCohort, activeModel, summary]);

  return (
    <div className="space-y-6">
      {/* Hero Cohort Standing Section: glass-strong card with Outfit 48px display + 420x130 shimmerLine SVG */}
      <motion.section
        aria-label="Executive cohort brief"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.28, ease: EASE_OUT_EXPO }}
        className="glass-strong rounded-2xl p-6 sm:p-7"
      >
        <div className="flex flex-col gap-6 xl:flex-row xl:items-center xl:justify-between">
          <div className="min-w-0 flex-1">
            <p className="text-[11px] font-semibold uppercase tracking-[0.12em] text-teal-300">
              Cohort Pass Standing · {activeMetrics.modelName}
            </p>
            <div className="mt-2 flex flex-wrap items-baseline gap-4">
              <p className="font-outfit text-[48px] font-bold leading-none tracking-tight text-white tabular-nums">
                {passRate.toFixed(1)}%
              </p>
              <span className="text-xs font-medium text-slate-300">
                {summary.passCount} passing of {summary.cleanedTrainingRows} cleaned records
              </span>
            </div>

            <div className="mt-5 grid grid-cols-2 gap-4 border-t border-white/10 pt-4 sm:grid-cols-4">
              <div>
                <p className="text-[11px] uppercase tracking-wider text-slate-400">
                  Cleaned Cohort
                </p>
                <p className="mt-1 font-mono text-[18px] font-semibold text-white tabular-nums">
                  {summary.cleanedTrainingRows}
                </p>
              </div>
              <div>
                <p className="text-[11px] uppercase tracking-wider text-slate-400">
                  Recorded Fails
                </p>
                <p className="mt-1 font-mono text-[18px] font-semibold text-rose-300 tabular-nums">
                  {summary.failCount}
                </p>
              </div>
              <div>
                <p className="text-[11px] uppercase tracking-wider text-slate-400">
                  High-Risk Load
                </p>
                <p className="mt-1 font-mono text-[18px] font-semibold text-amber-300 tabular-nums">
                  {highRiskCount}
                </p>
              </div>
              <div>
                <p className="text-[11px] uppercase tracking-wider text-slate-400">
                  Fail Recall / AUC
                </p>
                <p className="mt-1 font-mono text-[18px] font-semibold text-emerald-300 tabular-nums">
                  {(activeMetrics.failRecall * 100).toFixed(1)}% · {activeMetrics.rocAuc.toFixed(2)}
                </p>
              </div>
            </div>
          </div>

          {/* Right: Large SVG area (420px x 130px) with smooth white line + shimmerLine animation */}
          <div className="flex flex-col items-start gap-3 xl:items-end">
            <div className="w-full max-w-[420px] rounded-[16px] border border-white/10 bg-white/[0.03] p-3">
              <div className="mb-1 flex items-center justify-between text-[11px] uppercase tracking-wider text-slate-400">
                <span>Cohort Pass Trajectory (CGPA Bands)</span>
                <span className="font-mono text-white">Live Trend</span>
              </div>
              <svg
                viewBox="0 0 420 130"
                className="h-[130px] w-full max-w-[420px] overflow-visible"
                role="img"
                aria-label="Smooth white cohort pass rate trajectory line chart"
              >
                <defs>
                  <linearGradient id="heroWhiteArea" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="rgba(255,255,255,0.22)" />
                    <stop offset="100%" stopColor="rgba(255,255,255,0)" />
                  </linearGradient>
                </defs>
                <path
                  d={`${heroCurvePath} L 420 130 L 0 130 Z`}
                  fill="url(#heroWhiteArea)"
                />
                <path
                  d={heroCurvePath}
                  fill="none"
                  stroke="rgba(255,255,255,0.32)"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                />
                <path
                  d={heroCurvePath}
                  fill="none"
                  stroke="#ffffff"
                  strokeWidth="3"
                  strokeLinecap="round"
                  className="animate-shimmer-line"
                />
              </svg>
            </div>

            <div className="flex flex-wrap items-center gap-2.5">
              <button
                type="button"
                onClick={() => onNavigate('prediction')}
                className="inline-flex items-center gap-2 rounded-xl bg-white px-4 py-2 text-xs font-semibold text-slate-950 shadow-[0_10px_28px_-8px_rgba(255,255,255,0.4)] transition-colors hover:bg-slate-100 whitespace-nowrap"
              >
                <span>Predict Student Risk</span>
                <ArrowRight className="h-3.5 w-3.5" aria-hidden="true" />
              </button>
              <button
                type="button"
                onClick={() => onNavigate('batch')}
                className="glass rounded-xl px-4 py-2 text-xs font-medium text-slate-100 transition-colors hover:bg-white/10 whitespace-nowrap"
              >
                Batch Roster Scoring
              </button>
            </div>
          </div>
        </div>
      </motion.section>

      {warnings.length > 0 && (
        <div
          role="status"
          className="glass rounded-xl border border-amber-400/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-100"
        >
          <p className="font-medium">CSV Normalization Notes</p>
          <ul className="mt-1 list-disc pl-5 text-xs text-amber-200/90">
            {warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Three-column Bento Metrics Grid: Rose / Blue / Emerald with h-14 SVG area charts */}
      <section aria-label="Cohort KPIs" className="grid grid-cols-1 gap-5 md:grid-cols-3">
        {/* Card 1: Rose accent */}
        <GlassMotionCard delay={0.05} className="p-5">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-rose-300">
              At-Risk &amp; Failure Load
            </span>
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-rose-500/15 text-rose-300">
              <AlertTriangle className="h-4 w-4" aria-hidden="true" />
            </span>
          </div>
          <div className="mt-2 flex items-baseline justify-between gap-2">
            <p className="font-mono text-2xl font-bold text-white tabular-nums">
              {highRiskCount} High-Risk
            </p>
            <span className="font-mono text-xs text-rose-300 tabular-nums">
              {failRate.toFixed(1)}% fail rate
            </span>
          </div>
          <p className="mt-0.5 text-xs text-slate-400 tabular-nums">
            {summary.failCount} recorded fails · {highRiskShare.toFixed(1)}% P(Fail) ≥ 65%
          </p>
          <svg viewBox="0 0 240 56" className="mt-3 h-14 w-full overflow-visible" aria-hidden="true">
            <defs>
              <linearGradient id="roseGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="rgba(251,113,133,0.35)" />
                <stop offset="100%" stopColor="rgba(251,113,133,0)" />
              </linearGradient>
            </defs>
            <path d={`${sparkPaths.roseLine} L 240 56 L 0 56 Z`} fill="url(#roseGrad)" />
            <path d={sparkPaths.roseLine} fill="none" stroke="#fb7185" strokeWidth="2.2" />
          </svg>
        </GlassMotionCard>

        {/* Card 2: Blue accent */}
        <GlassMotionCard delay={0.09} className="p-5">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-blue-300">
              Model Fail Sensitivity
            </span>
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-500/15 text-blue-300">
              <Activity className="h-4 w-4" aria-hidden="true" />
            </span>
          </div>
          <div className="mt-2 flex items-baseline justify-between gap-2">
            <p className="font-mono text-2xl font-bold text-white tabular-nums">
              {(activeMetrics.failRecall * 100).toFixed(1)}% Recall
            </p>
            <span className="font-mono text-xs text-blue-300 tabular-nums">
              AUC {activeMetrics.rocAuc.toFixed(3)}
            </span>
          </div>
          <p className="mt-0.5 text-xs text-slate-400 tabular-nums">
            Accuracy {(activeMetrics.accuracy * 100).toFixed(1)}% · Precision{' '}
            {(activeMetrics.failPrecision * 100).toFixed(1)}%
          </p>
          <svg viewBox="0 0 240 56" className="mt-3 h-14 w-full overflow-visible" aria-hidden="true">
            <defs>
              <linearGradient id="blueGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="rgba(96,165,250,0.35)" />
                <stop offset="100%" stopColor="rgba(96,165,250,0)" />
              </linearGradient>
            </defs>
            <path d={`${sparkPaths.blueLine} L 240 56 L 0 56 Z`} fill="url(#blueGrad)" />
            <path d={sparkPaths.blueLine} fill="none" stroke="#60a5fa" strokeWidth="2.2" />
          </svg>
        </GlassMotionCard>

        {/* Card 3: Emerald accent */}
        <GlassMotionCard delay={0.13} className="p-5">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-emerald-300">
              Passing Cohort &amp; Retention
            </span>
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-500/15 text-emerald-300">
              <CheckCircle2 className="h-4 w-4" aria-hidden="true" />
            </span>
          </div>
          <div className="mt-2 flex items-baseline justify-between gap-2">
            <p className="font-mono text-2xl font-bold text-white tabular-nums">
              {summary.passCount} Pass
            </p>
            <span className="font-mono text-xs text-emerald-300 tabular-nums">
              {retentionCleanRatio.toFixed(1)}% clean
            </span>
          </div>
          <p className="mt-0.5 text-xs text-slate-400 tabular-nums">
            {summary.originalRows} raw · {summary.duplicateRowsRemoved} duplicates removed
          </p>
          <svg viewBox="0 0 240 56" className="mt-3 h-14 w-full overflow-visible" aria-hidden="true">
            <defs>
              <linearGradient id="emeraldGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="rgba(52,211,153,0.35)" />
                <stop offset="100%" stopColor="rgba(52,211,153,0)" />
              </linearGradient>
            </defs>
            <path d={`${sparkPaths.emeraldLine} L 240 56 L 0 56 Z`} fill="url(#emeraldGrad)" />
            <path d={sparkPaths.emeraldLine} fill="none" stroke="#34d399" strokeWidth="2.2" />
          </svg>
        </GlassMotionCard>
      </section>

      {/* 4 Primary Cohort Plotly Charts */}
      <section aria-label="Primary cohort visualizations" className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        <GlassMotionCard delay={0.14} className="p-5">
          <div className="flex items-baseline justify-between gap-2">
            <h2 className="font-display text-base font-semibold text-white">
              Attendance vs Final Result
            </h2>
            <span className="font-mono text-[11px] text-amber-300 tabular-nums">
              Benchmark: 75%
            </span>
          </div>
          <p className="mt-0.5 text-xs text-slate-400">
            Individual student spread and distribution quartiles with the 75% advising benchmark.
          </p>
          <PlotlyChart
            data={attendanceChartData}
            layout={{
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              margin: { t: 26, r: 18, b: 44, l: 54 },
              yaxis: {
                title: { text: 'Attendance (%)' },
                gridcolor: 'rgba(255,255,255,0.045)',
                zeroline: false,
                range: [25, 105],
              },
              xaxis: { title: { text: '' }, showgrid: false },
              showlegend: false,
              shapes: [
                {
                  type: 'line',
                  xref: 'paper',
                  x0: 0,
                  x1: 1,
                  y0: 75,
                  y1: 75,
                  line: { color: '#fbbf24', width: 1.5, dash: 'dash' },
                },
              ],
              annotations: [
                {
                  xref: 'paper',
                  x: 0.5,
                  y: 78,
                  text: '75% risk threshold',
                  showarrow: false,
                  xanchor: 'center',
                  bgcolor: 'rgba(10, 16, 30, 0.85)',
                  borderpad: 3,
                  font: { size: 10.5, color: '#fcd34d' },
                },
              ],
            }}
            height={305}
            reportTitle="Attendance vs Final Result"
            reportSubtitle="Pass vs Fail attendance distribution with 75% advising benchmark"
            ariaLabel="Box and swarm plot comparing attendance percentages for Pass vs Fail students"
          />
        </GlassMotionCard>

        <GlassMotionCard delay={0.18} className="p-6">
          <div className="flex items-baseline justify-between gap-2">
            <h2 className="font-display text-base font-semibold text-white">
              Previous CGPA Distribution
            </h2>
            <span className="font-mono text-[11px] text-amber-300 tabular-nums">
              Risk cutoff: 2.40
            </span>
          </div>
          <p className="mt-0.5 text-xs text-slate-400">
            Cohort CGPA overlap comparing Pass and Fail distributions against the 2.40 threshold.
          </p>
          <PlotlyChart
            data={cgpaChartData}
            layout={{
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              barmode: 'overlay',
              barcornerradius: 16,
              bargap: 0.14,
              margin: { t: 42, r: 18, b: 46, l: 52 },
              xaxis: {
                title: { text: 'Previous CGPA (1.00–4.00)' },
                gridcolor: 'rgba(255,255,255,0.045)',
                zeroline: false,
              },
              yaxis: {
                title: { text: 'Student Count' },
                gridcolor: 'rgba(255,255,255,0.045)',
                zeroline: false,
              },
              legend: { orientation: 'h', y: 1.14, x: 0 },
              shapes: [
                {
                  type: 'line',
                  yref: 'paper',
                  y0: 0,
                  y1: 1,
                  x0: 2.4,
                  x1: 2.4,
                  line: { color: '#fbbf24', width: 1.5, dash: 'dash' },
                },
              ],
              annotations: [
                {
                  x: 2.4,
                  yref: 'paper',
                  y: 0.96,
                  text: '2.40 CGPA',
                  showarrow: false,
                  xanchor: 'left',
                  xshift: 4,
                  font: { size: 10.5, color: '#fcd34d' },
                },
              ],
            }}
            height={305}
            reportTitle="Previous CGPA Distribution"
            reportSubtitle="Pass vs Fail CGPA overlap against 2.40 academic risk cutoff"
            ariaLabel="Overlapping histogram of previous CGPA for Pass vs Fail"
          />
        </GlassMotionCard>

        <GlassMotionCard delay={0.22} className="p-6">
          <div className="flex items-baseline justify-between gap-2">
            <h2 className="font-display text-base font-semibold text-white">
              Part-Time Job vs Final Result
            </h2>
            <span className="font-mono text-[11px] text-slate-300 tabular-nums">
              Fail rate: {jobFailRates.no}% vs {jobFailRates.yes}%
            </span>
          </div>
          <p className="mt-0.5 text-xs text-slate-400">
            Student counts and observed failure share grouped by part-time employment status.
          </p>
          <PlotlyChart
            data={partTimeChartData}
            layout={{
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              barmode: 'group',
              barcornerradius: 16,
              bargap: 0.3,
              bargroupgap: 0.16,
              margin: { t: 42, r: 18, b: 46, l: 52 },
              xaxis: { title: { text: '' }, showgrid: false },
              yaxis: {
                title: { text: 'Student Count' },
                gridcolor: 'rgba(255,255,255,0.045)',
                zeroline: false,
              },
              legend: { orientation: 'h', y: 1.14, x: 0 },
            }}
            height={305}
            reportTitle="Part-Time Job vs Final Result"
            reportSubtitle={`Observed failure share: No Job ${jobFailRates.no}% vs Part-Time Job ${jobFailRates.yes}%`}
            ariaLabel="Grouped bar chart of Pass vs Fail by part-time job status"
          />
        </GlassMotionCard>

        <GlassMotionCard delay={0.26} className="p-6">
          <div className="flex items-baseline justify-between gap-2">
            <h2 className="font-display text-base font-semibold text-white">
              Predicted Risk Tier Distribution
            </h2>
            <span className="font-mono text-[11px] text-teal-300 tabular-nums">
              {activeMetrics.modelName}
            </span>
          </div>
          <p className="mt-0.5 text-xs text-slate-400">
            Cohort segmentation into Low (&lt;35%), Medium (35–65%), and High (≥65%) risk tiers.
          </p>
          <PlotlyChart
            data={riskDistData}
            layout={{
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              barcornerradius: 16,
              bargap: 0.38,
              margin: { t: 24, r: 28, b: 46, l: 145 },
              xaxis: {
                title: { text: 'Students' },
                gridcolor: 'rgba(255,255,255,0.045)',
                zeroline: false,
              },
              yaxis: { automargin: true, showgrid: false },
              showlegend: false,
            }}
            height={305}
            reportTitle="Predicted Risk Tier Distribution"
            reportSubtitle={`${activeMetrics.modelName} segmentation: Low (<35%), Medium (35-65%), High (>=65%)`}
            ariaLabel="Horizontal bar chart of Low, Medium, and High risk student counts"
          />
        </GlassMotionCard>
      </section>

      {/* Recent Activity Panel: Full-width glass-strong panel with 'View all' link & 40px rounded-xl glass icon boxes */}
      <GlassMotionCard
        as="section"
        variant="glass-strong"
        ariaLabel="Recent early-warning activity"
        delay={0.28}
        className="p-6"
      >
        <div className="flex items-center justify-between border-b border-white/10 pb-4">
          <div>
            <h2 className="font-display text-base font-semibold text-white">
              Recent Early-Warning Activity
            </h2>
            <p className="mt-0.5 text-xs text-slate-400">
              Highest-priority student flags and pipeline audit events for {summary.datasetName}.
            </p>
          </div>
          <button
            type="button"
            onClick={() => onNavigate('register')}
            className="text-xs font-semibold text-teal-300 transition-colors hover:text-white whitespace-nowrap"
          >
            View all
          </button>
        </div>

        <div className="mt-3 divide-y divide-white/[0.06]">
          {topActivityRows.map((row) => {
            const Icon = row.icon;
            const iconStyle =
              row.tone === 'emerald'
                ? 'bg-emerald-500/15 text-emerald-300 border-emerald-400/25'
                : row.tone === 'amber'
                ? 'bg-amber-500/15 text-amber-300 border-amber-400/25'
                : 'bg-blue-500/15 text-blue-300 border-blue-400/25';

            return (
              <div
                key={row.id}
                className="flex items-center justify-between gap-4 rounded-xl px-3 py-3 transition-colors hover:bg-white/[0.025]"
              >
                <div className="flex min-w-0 items-center gap-3.5">
                  <div
                    className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border backdrop-blur-md ${iconStyle}`}
                  >
                    <Icon className="h-4 w-4" aria-hidden="true" />
                  </div>
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium text-white">{row.title}</p>
                    <p className="truncate text-xs text-slate-400">{row.subtitle}</p>
                  </div>
                </div>
                <span className="shrink-0 font-mono text-sm font-semibold text-slate-100 tabular-nums">
                  {row.value}
                </span>
              </div>
            );
          })}
        </div>
      </GlassMotionCard>

      {/* Data Cleaning & Pipeline Audit Panel */}
      <GlassMotionCard
        as="section"
        ariaLabel="Data cleaning audit"
        delay={0.3}
        className="p-6"
      >
        <div className="flex flex-col gap-2 border-b border-white/10 pb-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-2.5">
            <Database className="h-5 w-5 text-teal-300" aria-hidden="true" />
            <div>
              <h2 className="font-display text-base font-semibold text-white">
                Data Cleaning &amp; Preprocessing Audit
              </h2>
              <p className="text-xs text-slate-400">
                Exact duplicates removed · Numerical median &amp; categorical mode imputation ·
                student_id excluded from modeling
              </p>
            </div>
          </div>
          <div className="font-mono text-xs text-slate-300 tabular-nums">
            Mode(part_time_job): {summary.categoricalMode} · Range Violations Flagged:{' '}
            <span className={summary.outOfRangeCount > 0 ? 'font-semibold text-amber-300' : ''}>
              {summary.outOfRangeCount}
            </span>
          </div>
        </div>

        <div className="mt-5 grid grid-cols-2 gap-4 border-b border-white/10 pb-5 sm:grid-cols-3 lg:grid-cols-6">
          <div>
            <p className="text-xs text-slate-400">Original Rows</p>
            <p className="mt-1 font-mono text-xl font-semibold text-white tabular-nums">
              {summary.originalRows}
            </p>
          </div>
          <div>
            <p className="text-xs text-slate-400">Duplicates Removed</p>
            <p className="mt-1 font-mono text-xl font-semibold text-white tabular-nums">
              {summary.duplicateRowsRemoved}
            </p>
          </div>
          <div>
            <p className="text-xs text-slate-400">Missing Values</p>
            <p className="mt-1 font-mono text-xl font-semibold text-white tabular-nums">
              {summary.totalMissingCells}
            </p>
          </div>
          <div>
            <p className="text-xs text-slate-400">Missing Target Excluded</p>
            <p className="mt-1 font-mono text-xl font-semibold text-amber-300 tabular-nums">
              {summary.missingFinalResultExcluded}
            </p>
          </div>
          <div>
            <p className="text-xs text-slate-400">Pass Records</p>
            <p className="mt-1 font-mono text-xl font-semibold text-emerald-300 tabular-nums">
              {summary.passCount}
            </p>
          </div>
          <div>
            <p className="text-xs text-slate-400">Fail Records</p>
            <p className="mt-1 font-mono text-xl font-semibold text-rose-300 tabular-nums">
              {summary.failCount}
            </p>
          </div>
        </div>

        <div className="mt-5 overflow-x-auto">
          <table className="w-full border-collapse text-left text-xs">
            <thead>
              <tr className="border-b border-white/10 text-slate-400">
                <th className="py-2 pr-4 font-medium">Column</th>
                <th className="py-2 px-4 text-right font-medium">Missing Cells</th>
                <th className="py-2 px-4 font-medium">Cleaning Rule</th>
                <th className="py-2 pl-4 text-right font-medium">Cohort Imputation / Role</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.06]">
              {summary.missingByColumn.map((col) => {
                let rule = 'Median (fit on train fold)';
                let valueDisplay = '-';
                if (col.column === 'student_id') {
                  rule = 'Identifier only';
                  valueDisplay = 'Excluded from model';
                } else if (col.column === 'part_time_job') {
                  rule = 'Mode (fit on train fold)';
                  valueDisplay = summary.categoricalMode;
                } else if (col.column === 'final_result') {
                  rule = 'Exclude rows with missing target';
                  valueDisplay = `${summary.missingFinalResultExcluded} excluded`;
                } else {
                  const med =
                    summary.numericMedians[
                      col.column as keyof typeof summary.numericMedians
                    ];
                  valueDisplay = med !== undefined ? String(med) : '-';
                }

                return (
                  <tr key={col.column} className="hover:bg-white/[0.025]">
                    <td className="py-2.5 pr-4 font-mono font-medium text-slate-100">
                      {col.column}
                    </td>
                    <td className="py-2.5 px-4 text-right font-mono text-slate-300 tabular-nums">
                      {col.missingCount}
                    </td>
                    <td className="py-2.5 px-4 text-slate-300">{rule}</td>
                    <td className="py-2.5 pl-4 text-right font-mono text-slate-200 tabular-nums">
                      {valueDisplay}
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
