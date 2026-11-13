import { useEffect, useMemo, useState } from 'react';
import { Info } from 'lucide-react';
import { ModelType, StudentFeatureInput, TrainedModelBundle } from '../../types/student';
import { predictSingleStudent, simulateAttendanceCurve } from '../../ml/pipeline';
import { ViewReportContext } from '../../utils/reportData';
import { PlotlyChart } from '../PlotlyChart';
import { GlassMotionCard } from '../MotionPrimitives';

interface WhatIfViewProps {
  bundle: TrainedModelBundle;
  activeModel: ModelType;
  baseStudent: StudentFeatureInput;
  studentLabel: string;
  onUpdateBaseStudent: (next: StudentFeatureInput) => void;
  onReportContextChange?: (patch: Partial<ViewReportContext>) => void;
}

export function WhatIfView({
  bundle,
  activeModel,
  baseStudent,
  studentLabel,
  onUpdateBaseStudent,
  onReportContextChange,
}: WhatIfViewProps) {
  const [simulatedAttendance, setSimulatedAttendance] = useState<number>(
    baseStudent.attendance_pct
  );

  const baselinePred = useMemo(
    () => predictSingleStudent(baseStudent, bundle, activeModel),
    [baseStudent, bundle, activeModel]
  );

  const simulatedPred = useMemo(
    () =>
      predictSingleStudent(
        { ...baseStudent, attendance_pct: simulatedAttendance },
        bundle,
        activeModel
      ),
    [baseStudent, simulatedAttendance, bundle, activeModel]
  );

  const curve = useMemo(
    () => simulateAttendanceCurve(baseStudent, bundle, activeModel),
    [baseStudent, bundle, activeModel]
  );

  useEffect(() => {
    onReportContextChange?.({
      whatif: {
        simulatedAttendance,
        studentLabel,
        baseStudent,
        baselinePred,
        simulatedPred,
      },
    });
  }, [
    onReportContextChange,
    simulatedAttendance,
    studentLabel,
    baseStudent,
    baselinePred,
    simulatedPred,
  ]);

  const deltaPct = (simulatedPred.failProbability - baselinePred.failProbability) * 100;

  const curvePlotData = useMemo(
    () => [
      {
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Hypothetical P(Fail)',
        x: curve.map((p) => p.attendance),
        y: curve.map((p) => Number((p.failProbability * 100).toFixed(1))),
        line: { color: '#f8fafc', width: 3.2, shape: 'spline', smoothing: 1.15 },
        fill: 'tozeroy',
        fillcolor: 'rgba(248, 250, 252, 0.055)',
        marker: { size: 6.5, color: '#34d399', line: { color: '#060913', width: 1.2 } },
      },
      {
        type: 'scatter',
        mode: 'markers',
        name: 'Baseline Attendance',
        x: [baseStudent.attendance_pct],
        y: [Number((baselinePred.failProbability * 100).toFixed(1))],
        marker: { size: 11, color: '#cbd5e1', symbol: 'circle-open', line: { width: 2.5 } },
      },
      {
        type: 'scatter',
        mode: 'markers',
        name: 'Simulated Attendance',
        x: [simulatedAttendance],
        y: [Number((simulatedPred.failProbability * 100).toFixed(1))],
        marker: {
          size: 13,
          color:
            simulatedPred.riskTier === 'High'
              ? '#fb7185'
              : simulatedPred.riskTier === 'Medium'
              ? '#fbbf24'
              : '#34d399',
          line: { color: '#ffffff', width: 1.5 },
        },
      },
    ],
    [curve, baseStudent.attendance_pct, baselinePred.failProbability, simulatedAttendance, simulatedPred]
  );

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-2xl font-bold tracking-tight text-white">
          What-if Attendance Simulator
        </h1>
        <p className="mt-1 text-sm text-slate-400">
          Explore how hypothetical changes in lecture &amp; lab attendance affect predicted failure
          probability while holding all other student features constant.
        </p>
      </div>

      {/* Explicit Hypothetical Disclaimer */}
      <div
        role="note"
        className="flex items-start gap-2.5 rounded-xl border border-blue-400/25 bg-blue-500/10 px-4 py-3 text-xs text-slate-200 backdrop-blur-md"
      >
        <Info className="mt-0.5 h-4 w-4 shrink-0 text-blue-300" aria-hidden="true" />
        <p>
          <strong className="text-white">Hypothetical Model Prediction:</strong> This simulation computes counterfactual
          model output using{' '}
          <span className="font-medium text-blue-200">
            {activeModel === 'random_forest' ? 'Random Forest' : 'Logistic Regression'}
          </span>{' '}
          by varying only attendance while keeping quiz, assignment, midterm, lab, study hours,
          CGPA, and employment unchanged. It reflects learned statistical associations, not a
          guaranteed causal outcome.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-12">
        {/* Controls & Fixed Profile */}
        <GlassMotionCard delay={0.04} className="space-y-5 p-6 lg:col-span-5">
          <div>
            <div className="flex items-center justify-between">
              <label
                htmlFor="whatif-attendance-slider"
                className="font-display text-sm font-semibold text-white"
              >
                Simulated Attendance (%)
              </label>
              <span
                data-testid="simulated-attendance-value"
                className="font-mono text-2xl font-bold text-emerald-400 tabular-nums"
              >
                {simulatedAttendance.toFixed(0)}%
              </span>
            </div>

            <input
              id="whatif-attendance-slider"
              type="range"
              min={30}
              max={100}
              step={1}
              value={simulatedAttendance}
              onChange={(e) => setSimulatedAttendance(Number(e.target.value))}
              className="mt-3 h-2 w-full cursor-pointer accent-emerald-400"
            />
            <div className="mt-1 flex justify-between font-mono text-[11px] text-slate-400 tabular-nums">
              <span>30%</span>
              <span>75% Benchmark</span>
              <span>100%</span>
            </div>
          </div>

          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => setSimulatedAttendance(baseStudent.attendance_pct)}
              className="rounded-lg border border-white/12 bg-white/[0.06] px-3 py-1.5 text-xs font-medium text-slate-200 backdrop-blur-sm transition-colors hover:bg-white/[0.12] whitespace-nowrap"
            >
              Reset ({baseStudent.attendance_pct.toFixed(0)}%)
            </button>
            <button
              type="button"
              onClick={() => setSimulatedAttendance(75)}
              className="rounded-lg border border-white/12 bg-white/[0.06] px-3 py-1.5 text-xs font-medium text-slate-200 backdrop-blur-sm transition-colors hover:bg-white/[0.12] whitespace-nowrap"
            >
              Test 75% Benchmark
            </button>
            <button
              type="button"
              onClick={() => setSimulatedAttendance(90)}
              className="rounded-lg border border-emerald-400/30 bg-emerald-500/15 px-3 py-1.5 text-xs font-medium text-emerald-200 backdrop-blur-sm transition-colors hover:bg-emerald-500/25 whitespace-nowrap"
            >
              Test 90% Attendance
            </button>
          </div>

          <div className="border-t border-white/10 pt-4">
            <div className="flex items-center justify-between gap-3">
              <h2 className="text-xs font-semibold text-slate-200">
                Held-Constant Profile ({studentLabel})
              </h2>
              <button
                type="button"
                onClick={() => {
                  const next = {
                    attendance_pct: 58,
                    quiz_avg: 5.2,
                    assignment_avg: 5.5,
                    mid_marks: 14.5,
                    lab_marks: 11.0,
                    study_hours_week: 7.5,
                    prev_cgpa: 2.25,
                    part_time_job: 'Yes' as const,
                  };
                  onUpdateBaseStudent(next);
                  setSimulatedAttendance(next.attendance_pct);
                }}
                className="text-xs font-medium text-blue-300 hover:text-blue-200 hover:underline whitespace-nowrap shrink-0"
              >
                Load At-Risk Profile
              </button>
            </div>
            <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
              <div className="flex justify-between border-b border-white/[0.08] py-1">
                <dt className="text-slate-400">Quiz Avg</dt>
                <dd className="font-mono text-white tabular-nums">
                  {baseStudent.quiz_avg.toFixed(1)}/10
                </dd>
              </div>
              <div className="flex justify-between border-b border-white/[0.08] py-1">
                <dt className="text-slate-400">Assignment</dt>
                <dd className="font-mono text-white tabular-nums">
                  {baseStudent.assignment_avg.toFixed(1)}/10
                </dd>
              </div>
              <div className="flex justify-between border-b border-white/[0.08] py-1">
                <dt className="text-slate-400">Midterm</dt>
                <dd className="font-mono text-white tabular-nums">
                  {baseStudent.mid_marks.toFixed(1)}/30
                </dd>
              </div>
              <div className="flex justify-between border-b border-white/[0.08] py-1">
                <dt className="text-slate-400">Lab Marks</dt>
                <dd className="font-mono text-white tabular-nums">
                  {baseStudent.lab_marks.toFixed(1)}/20
                </dd>
              </div>
              <div className="flex justify-between border-b border-white/[0.08] py-1">
                <dt className="text-slate-400">Study Hrs/Wk</dt>
                <dd className="font-mono text-white tabular-nums">
                  {baseStudent.study_hours_week.toFixed(1)}
                </dd>
              </div>
              <div className="flex justify-between border-b border-white/[0.08] py-1">
                <dt className="text-slate-400">Prev. CGPA</dt>
                <dd className="font-mono text-white tabular-nums">
                  {baseStudent.prev_cgpa.toFixed(2)}
                </dd>
              </div>
              <div className="flex justify-between py-1">
                <dt className="text-slate-400">Part-Time Job</dt>
                <dd className="font-medium text-white">{baseStudent.part_time_job}</dd>
              </div>
            </dl>
          </div>
        </GlassMotionCard>

        {/* Comparison & Sensitivity Curve */}
        <GlassMotionCard variant="glass-strong" delay={0.1} className="space-y-5 p-6 lg:col-span-7">
          <div className="grid grid-cols-1 gap-4 border-b border-white/10 pb-5 sm:grid-cols-3">
            <div>
              <p className="text-[11px] font-medium uppercase tracking-wider text-slate-400">
                Baseline Fail Risk ({baseStudent.attendance_pct.toFixed(0)}%)
              </p>
              <p className="mt-1 font-mono text-2xl font-bold text-white tabular-nums">
                {(baselinePred.failProbability * 100).toFixed(1)}%
              </p>
              <p className="text-xs text-slate-400">{baselinePred.riskTier} Risk</p>
            </div>

            <div>
              <p className="text-[11px] font-medium uppercase tracking-wider text-slate-400">
                Simulated Fail Risk ({simulatedAttendance.toFixed(0)}%)
              </p>
              <p
                data-testid="whatif-simulated-prob"
                className={`mt-1 font-mono text-2xl font-bold tabular-nums ${
                  simulatedPred.riskTier === 'High'
                    ? 'text-rose-400'
                    : simulatedPred.riskTier === 'Medium'
                    ? 'text-amber-300'
                    : 'text-emerald-400'
                }`}
              >
                {(simulatedPred.failProbability * 100).toFixed(1)}%
              </p>
              <p className="text-xs font-medium text-slate-300">
                {simulatedPred.predictedResult} · {simulatedPred.riskTier} Risk
              </p>
            </div>

            <div>
              <p className="text-[11px] font-medium uppercase tracking-wider text-slate-400">Risk Shift vs Baseline</p>
              <p
                className={`mt-1 font-mono text-2xl font-bold tabular-nums ${
                  deltaPct <= 0 ? 'text-emerald-400' : 'text-rose-400'
                }`}
              >
                {deltaPct > 0 ? '+' : ''}
                {deltaPct.toFixed(1)} pp
              </p>
              <p className="text-xs text-slate-400">Percentage points change</p>
            </div>
          </div>

          <div>
            <div className="flex items-baseline justify-between gap-2">
              <h2 className="font-display text-sm font-semibold text-white">
                Hypothetical Attendance Sensitivity Curve (30%–100%)
              </h2>
              <span className="font-mono text-[11px] text-slate-400 tabular-nums">
                Risk bands: 35% · 65%
              </span>
            </div>
            <PlotlyChart
              data={curvePlotData}
              layout={{
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                margin: { t: 48, r: 18, b: 46, l: 56 },
                xaxis: {
                  title: { text: 'Simulated Attendance (%)' },
                  gridcolor: 'rgba(255,255,255,0.045)',
                  zeroline: false,
                },
                yaxis: {
                  title: { text: 'Predicted Failure Probability (%)' },
                  range: [0, 104],
                  gridcolor: 'rgba(255,255,255,0.045)',
                  zeroline: false,
                },
                legend: { orientation: 'h', y: 1.18, x: 0 },
                shapes: [
                  {
                    type: 'rect',
                    xref: 'paper',
                    x0: 0,
                    x1: 1,
                    y0: 0,
                    y1: 35,
                    fillcolor: 'rgba(16, 185, 129, 0.08)',
                    line: { width: 0 },
                    layer: 'below',
                  },
                  {
                    type: 'rect',
                    xref: 'paper',
                    x0: 0,
                    x1: 1,
                    y0: 35,
                    y1: 65,
                    fillcolor: 'rgba(245, 158, 11, 0.08)',
                    line: { width: 0 },
                    layer: 'below',
                  },
                  {
                    type: 'rect',
                    xref: 'paper',
                    x0: 0,
                    x1: 1,
                    y0: 65,
                    y1: 104,
                    fillcolor: 'rgba(244, 63, 94, 0.08)',
                    line: { width: 0 },
                    layer: 'below',
                  },
                  {
                    type: 'line',
                    x0: 75,
                    x1: 75,
                    y0: 0,
                    y1: 104,
                    line: { color: '#fbbf24', width: 1.4, dash: 'dash' },
                  },
                ],
              }}
              height={325}
              reportTitle="Hypothetical Attendance Sensitivity Curve (30%-100%)"
              reportSubtitle={`Baseline ${baseStudent.attendance_pct.toFixed(0)}% vs Simulated ${simulatedAttendance.toFixed(0)}% for ${studentLabel}`}
              ariaLabel="Line chart showing hypothetical failure probability across attendance percentages with shaded risk bands"
            />
          </div>
        </GlassMotionCard>
      </div>
    </div>
  );
}
