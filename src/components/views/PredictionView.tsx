import React, { useEffect, useMemo, useState } from 'react';
import { motion } from 'motion/react';
import { AlertCircle, CheckCircle2, Info, ShieldAlert } from 'lucide-react';
import {
  ModelEvaluationMetrics,
  ModelType,
  NumericFeatureKey,
  StudentFeatureInput,
  TrainedModelBundle,
} from '../../types/student';
import { FEATURE_RANGES, validateStudentFeatureInput } from '../../ml/cleaning';
import { predictSingleStudent } from '../../ml/pipeline';
import { ViewReportContext } from '../../utils/reportData';
import { PlotlyChart } from '../PlotlyChart';
import { EASE_OUT_EXPO, GLASS_CARD_CLASS, GlassMotionCard } from '../MotionPrimitives';

interface PredictionViewProps {
  bundle: TrainedModelBundle;
  activeModel: ModelType;
  onChangeModel: (model: ModelType) => void;
  studentDraft: StudentFeatureInput;
  onUpdateDraft: (next: StudentFeatureInput) => void;
  onReportContextChange?: (patch: Partial<ViewReportContext>) => void;
}

function ConfusionMatrixCard({
  metrics,
  delay = 0,
}: {
  metrics: ModelEvaluationMetrics;
  delay?: number;
}) {
  const { tp, fp, fn, tn } = metrics.confusionMatrix;
  return (
    <GlassMotionCard delay={delay} className="p-5">
      <div className="flex items-start justify-between gap-2">
        <div>
          <h3 className="font-display text-base font-semibold text-white">
            {metrics.modelName}
          </h3>
          <p className="mt-0.5 text-xs text-slate-400 tabular-nums">
            Train: {metrics.trainSize} · Held-out Test: {metrics.testSize} · Positive Class: Fail
          </p>
        </div>
        <span className="font-mono text-xs font-semibold text-emerald-300 tabular-nums">
          ROC-AUC {metrics.rocAuc.toFixed(3)}
        </span>
      </div>

      <div className="mt-4 grid grid-cols-3 gap-4 border-y border-white/10 py-3.5">
        <div>
          <p className="text-xs text-slate-400">Accuracy</p>
          <p className="mt-0.5 font-mono text-lg font-semibold text-white tabular-nums">
            {(metrics.accuracy * 100).toFixed(1)}%
          </p>
          <div className="mt-1.5 h-1 w-full overflow-hidden rounded-full bg-white/10">
            <div
              className="h-full bg-white"
              style={{ width: `${(metrics.accuracy * 100).toFixed(1)}%` }}
            />
          </div>
        </div>
        <div>
          <p className="text-xs text-slate-400">Fail Precision</p>
          <p className="mt-0.5 font-mono text-lg font-semibold text-amber-300 tabular-nums">
            {(metrics.failPrecision * 100).toFixed(1)}%
          </p>
          <div className="mt-1.5 h-1 w-full overflow-hidden rounded-full bg-white/10">
            <div
              className="h-full bg-amber-400"
              style={{ width: `${(metrics.failPrecision * 100).toFixed(1)}%` }}
            />
          </div>
        </div>
        <div>
          <p className="text-xs text-slate-400">Fail Recall</p>
          <p className="mt-0.5 font-mono text-lg font-semibold text-emerald-300 tabular-nums">
            {(metrics.failRecall * 100).toFixed(1)}%
          </p>
          <div className="mt-1.5 h-1 w-full overflow-hidden rounded-full bg-white/10">
            <div
              className="h-full bg-emerald-400"
              style={{ width: `${(metrics.failRecall * 100).toFixed(1)}%` }}
            />
          </div>
        </div>
      </div>

      <div className="mt-4">
        <p className="mb-2 text-xs font-medium text-slate-300">
          Confusion Matrix (Held-Out Stratified Test Set)
        </p>
        <div className="grid grid-cols-3 gap-1.5 text-center text-xs">
          <div />
          <div className="py-1 font-medium text-slate-400">Pred: Fail</div>
          <div className="py-1 font-medium text-slate-400">Pred: Pass</div>

          <div className="flex items-center justify-end pr-2 font-medium text-slate-300">
            Actual: Fail
          </div>
          <div className="rounded-xl border border-emerald-400/25 bg-emerald-500/15 py-2.5 font-mono font-semibold text-emerald-200 tabular-nums">
            {tp}{' '}
            <span className="block text-[10px] font-normal text-emerald-300/90">
              TP (Caught)
            </span>
          </div>
          <div className="rounded-xl border border-rose-400/25 bg-rose-500/15 py-2.5 font-mono font-semibold text-rose-200 tabular-nums">
            {fn}{' '}
            <span className="block text-[10px] font-normal text-rose-300/90">FN (Missed)</span>
          </div>

          <div className="flex items-center justify-end pr-2 font-medium text-slate-300">
            Actual: Pass
          </div>
          <div className="rounded-xl border border-amber-400/25 bg-amber-500/15 py-2.5 font-mono font-semibold text-amber-200 tabular-nums">
            {fp}{' '}
            <span className="block text-[10px] font-normal text-amber-300/90">
              FP (False Flag)
            </span>
          </div>
          <div className="rounded-xl border border-white/10 bg-white/[0.06] py-2.5 font-mono font-semibold text-slate-100 tabular-nums">
            {tn}{' '}
            <span className="block text-[10px] font-normal text-slate-400">TN (Correct)</span>
          </div>
        </div>
      </div>
    </GlassMotionCard>
  );
}

export function PredictionView({
  bundle,
  activeModel,
  onChangeModel,
  studentDraft,
  onUpdateDraft,
  onReportContextChange,
}: PredictionViewProps) {
  const [explainView, setExplainView] = useState<'rf' | 'lr'>('rf');
  const [formValues, setFormValues] = useState<Record<NumericFeatureKey, string>>(() => ({
    attendance_pct: String(studentDraft.attendance_pct),
    quiz_avg: String(studentDraft.quiz_avg),
    assignment_avg: String(studentDraft.assignment_avg),
    mid_marks: String(studentDraft.mid_marks),
    lab_marks: String(studentDraft.lab_marks),
    study_hours_week: String(studentDraft.study_hours_week),
    prev_cgpa: String(studentDraft.prev_cgpa),
  }));
  const [partTimeJob, setPartTimeJob] = useState<'Yes' | 'No'>(studentDraft.part_time_job);
  const [submittedInput, setSubmittedInput] = useState<StudentFeatureInput | null>(null);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);

  const prediction = useMemo(() => {
    if (!submittedInput) return null;
    return predictSingleStudent(submittedInput, bundle, activeModel);
  }, [submittedInput, bundle, activeModel]);

  useEffect(() => {
    onReportContextChange?.({
      prediction: {
        submittedInput,
        result: prediction,
        explainView,
        draftInput: studentDraft,
      },
    });
  }, [onReportContextChange, submittedInput, prediction, explainView, studentDraft]);

  const riskGaugeData = useMemo(() => {
    if (!prediction) return [];
    const pct = Number((prediction.failProbability * 100).toFixed(1));
    const barColor =
      prediction.riskTier === 'High'
        ? '#fb7185'
        : prediction.riskTier === 'Medium'
        ? '#fbbf24'
        : '#34d399';

    return [
      {
        type: 'indicator',
        mode: 'gauge+number',
        value: pct,
        number: {
          suffix: '%',
          font: { family: 'JetBrains Mono, monospace', size: 28, color: '#ffffff' },
        },
        gauge: {
          axis: {
            range: [0, 100],
            tickwidth: 1,
            tickcolor: '#94a3b8',
            tickvals: [0, 35, 65, 100],
            ticktext: ['0%', '35%', '65%', '100%'],
            tickfont: { size: 10.5, color: '#cbd5e1' },
          },
          bar: { color: barColor, thickness: 0.28 },
          bgcolor: 'rgba(255,255,255,0.06)',
          borderwidth: 0,
          steps: [
            { range: [0, 35], color: 'rgba(52, 211, 153, 0.14)' },
            { range: [35, 65], color: 'rgba(251, 191, 36, 0.15)' },
            { range: [65, 100], color: 'rgba(251, 113, 133, 0.16)' },
          ],
          threshold: {
            line: { color: '#ffffff', width: 2.5 },
            thickness: 0.75,
            value: pct,
          },
        },
      },
    ];
  }, [prediction]);

  const benchmarkChecks = useMemo(() => {
    if (!submittedInput) return [];
    return [
      {
        label: 'Attendance',
        val: `${submittedInput.attendance_pct.toFixed(0)}%`,
        target: '≥ 75%',
        ok: submittedInput.attendance_pct >= 75,
      },
      {
        label: 'Quiz Avg',
        val: `${submittedInput.quiz_avg.toFixed(1)}/10`,
        target: '≥ 6.0/10',
        ok: submittedInput.quiz_avg >= 6,
      },
      {
        label: 'Assign. Avg',
        val: `${submittedInput.assignment_avg.toFixed(1)}/10`,
        target: '≥ 6.0/10',
        ok: submittedInput.assignment_avg >= 6,
      },
      {
        label: 'Midterm',
        val: `${submittedInput.mid_marks.toFixed(1)}/30`,
        target: '≥ 18/30',
        ok: submittedInput.mid_marks >= 18,
      },
      {
        label: 'Lab Marks',
        val: `${submittedInput.lab_marks.toFixed(1)}/20`,
        target: '≥ 12/20',
        ok: submittedInput.lab_marks >= 12,
      },
      {
        label: 'Study Hrs',
        val: `${submittedInput.study_hours_week.toFixed(1)}h`,
        target: '≥ 10h/wk',
        ok: submittedInput.study_hours_week >= 10,
      },
      {
        label: 'Prev. CGPA',
        val: submittedInput.prev_cgpa.toFixed(2),
        target: '≥ 2.40',
        ok: submittedInput.prev_cgpa >= 2.4,
      },
    ];
  }, [submittedInput]);

  const handleTextChange = (field: NumericFeatureKey, raw: string) => {
    setFormValues((prev) => ({ ...prev, [field]: raw }));
    setValidationErrors([]);
    setSubmittedInput(null);

    const num = Number(raw);
    if (raw.trim() !== '' && Number.isFinite(num)) {
      const spec = FEATURE_RANGES[field];
      if (num >= spec.min && num <= spec.max) {
        onUpdateDraft({ ...studentDraft, [field]: num, part_time_job: partTimeJob });
      }
    }
  };

  const applyPreset = (preset: 'high' | 'medium' | 'low') => {
    const next: StudentFeatureInput =
      preset === 'high'
        ? {
            attendance_pct: 54,
            quiz_avg: 4.5,
            assignment_avg: 4.8,
            mid_marks: 12.5,
            lab_marks: 10.0,
            study_hours_week: 5,
            prev_cgpa: 2.05,
            part_time_job: 'Yes',
          }
        : preset === 'medium'
        ? {
            attendance_pct: 73,
            quiz_avg: 6.2,
            assignment_avg: 6.6,
            mid_marks: 17.5,
            lab_marks: 13.0,
            study_hours_week: 10,
            prev_cgpa: 2.55,
            part_time_job: 'Yes',
          }
        : {
            attendance_pct: 92,
            quiz_avg: 8.6,
            assignment_avg: 8.9,
            mid_marks: 25.5,
            lab_marks: 18.0,
            study_hours_week: 18,
            prev_cgpa: 3.55,
            part_time_job: 'No',
          };

    setFormValues({
      attendance_pct: String(next.attendance_pct),
      quiz_avg: String(next.quiz_avg),
      assignment_avg: String(next.assignment_avg),
      mid_marks: String(next.mid_marks),
      lab_marks: String(next.lab_marks),
      study_hours_week: String(next.study_hours_week),
      prev_cgpa: String(next.prev_cgpa),
    });
    setPartTimeJob(next.part_time_job);
    setValidationErrors([]);
    setSubmittedInput(null);
    onUpdateDraft(next);
  };

  const handlePredictSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const errors: string[] = [];

    const parseField = (key: NumericFeatureKey): number => {
      const raw = formValues[key].trim();
      const spec = FEATURE_RANGES[key];
      if (raw === '') {
        errors.push(`${spec.label} is required (${spec.unitText}).`);
        return NaN;
      }
      const val = Number(raw);
      if (!Number.isFinite(val)) {
        errors.push(`${spec.label} must be a valid number (${spec.unitText}).`);
        return NaN;
      }
      if (val < spec.min || val > spec.max) {
        errors.push(
          `${spec.label} must be between ${spec.min} and ${spec.max} (entered ${val}).`
        );
      }
      return val;
    };

    const candidate: StudentFeatureInput = {
      attendance_pct: parseField('attendance_pct'),
      quiz_avg: parseField('quiz_avg'),
      assignment_avg: parseField('assignment_avg'),
      mid_marks: parseField('mid_marks'),
      lab_marks: parseField('lab_marks'),
      study_hours_week: parseField('study_hours_week'),
      prev_cgpa: parseField('prev_cgpa'),
      part_time_job: partTimeJob,
    };

    if (errors.length > 0) {
      setValidationErrors(errors);
      setSubmittedInput(null);
      return;
    }

    const extraErrors = validateStudentFeatureInput(candidate);
    if (extraErrors.length > 0) {
      setValidationErrors(extraErrors);
      setSubmittedInput(null);
      return;
    }

    setValidationErrors([]);
    onUpdateDraft(candidate);
    setSubmittedInput(candidate);
  };

  const explainChartData = useMemo(() => {
    if (explainView === 'rf') {
      const sorted = [...bundle.rfFeatureImportance].reverse();
      return [
        {
          type: 'bar',
          orientation: 'h',
          x: sorted.map((item) => Number((item.importance * 100).toFixed(1))),
          y: sorted.map((item) => item.label),
          marker: { color: '#34d399', cornerradius: 16 },
          text: sorted.map((item) => `${(item.importance * 100).toFixed(1)}%`),
          textposition: 'auto',
          insidetextfont: { color: '#0f172a', family: 'JetBrains Mono, monospace', size: 11 },
          outsidetextfont: { color: '#f1f5f9', family: 'JetBrains Mono, monospace', size: 11 },
        },
      ];
    }

    const sorted = [...bundle.lrFeatureImportance].reverse();
    return [
      {
        type: 'bar',
        orientation: 'h',
        x: sorted.map((item) => Number((item.coefficient ?? 0).toFixed(3))),
        y: sorted.map((item) => item.label),
        marker: {
          color: sorted.map((item) => ((item.coefficient ?? 0) >= 0 ? '#fb7185' : '#34d399')),
          cornerradius: 16,
        },
        text: sorted.map((item) => (item.coefficient ?? 0).toFixed(2)),
        textposition: 'auto',
        insidetextfont: { color: '#0f172a', family: 'JetBrains Mono, monospace', size: 11 },
        outsidetextfont: { color: '#f1f5f9', family: 'JetBrains Mono, monospace', size: 11 },
      },
    ];
  }, [bundle, explainView]);

  const activeExplainList =
    explainView === 'rf' ? bundle.rfFeatureImportance : bundle.lrFeatureImportance;

  const inputClass =
    'mt-1.5 w-full rounded-xl border border-white/15 bg-white/[0.05] px-3 py-2 font-mono text-sm text-white focus:border-teal-400 focus:outline-none focus:ring-1 focus:ring-teal-400';

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <p className="text-sm text-slate-300">
          Leak-free stratified ML pipelines treating{' '}
          <span className="font-semibold text-white">Fail</span> as the positive class with
          balanced class weighting.
        </p>

        <div className="glass flex items-center gap-1 rounded-xl p-1 shrink-0">
          <button
            type="button"
            onClick={() => onChangeModel('random_forest')}
            className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
              activeModel === 'random_forest'
                ? 'bg-white text-slate-950 shadow-xs'
                : 'text-slate-300 hover:text-white'
            }`}
          >
            Use Random Forest
          </button>
          <button
            type="button"
            onClick={() => onChangeModel('logistic_regression')}
            className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
              activeModel === 'logistic_regression'
                ? 'bg-white text-slate-950 shadow-xs'
                : 'text-slate-300 hover:text-white'
            }`}
          >
            Use Logistic Regression
          </button>
        </div>
      </div>

      {/* Individual Student Prediction + Personalized Advisor */}
      <section
        aria-label="Individual student risk predictor"
        className="grid grid-cols-1 gap-5 lg:grid-cols-12"
      >
        <GlassMotionCard delay={0.05} className="p-6 lg:col-span-6">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/10 pb-4">
            <div>
              <h2 className="font-display text-base font-semibold text-white">
                Individual Student Feature Input
              </h2>
              <p className="text-xs text-slate-400">
                Enter all 8 academic indicators within their valid scales and click Predict Student Risk.
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-1.5">
              <button
                type="button"
                onClick={() => applyPreset('high')}
                className="glass rounded-lg px-2.5 py-1 text-xs font-medium text-slate-200 hover:bg-white/10 whitespace-nowrap"
              >
                Sample High-Risk
              </button>
              <button
                type="button"
                onClick={() => applyPreset('medium')}
                className="glass rounded-lg px-2.5 py-1 text-xs font-medium text-slate-200 hover:bg-white/10 whitespace-nowrap"
              >
                Sample Borderline
              </button>
              <button
                type="button"
                onClick={() => applyPreset('low')}
                className="glass rounded-lg px-2.5 py-1 text-xs font-medium text-slate-200 hover:bg-white/10 whitespace-nowrap"
              >
                Sample On-Track
              </button>
            </div>
          </div>

          <form onSubmit={handlePredictSubmit} noValidate className="mt-5 space-y-5">
            {validationErrors.length > 0 && (
              <div
                role="alert"
                className="rounded-xl border border-rose-400/35 bg-rose-500/15 px-3.5 py-3 text-xs text-rose-100"
              >
                <p className="font-semibold">Invalid Academic Input Range</p>
                <ul className="mt-1 list-disc space-y-0.5 pl-4 text-rose-200">
                  {validationErrors.map((err, idx) => (
                    <li key={idx}>{err}</li>
                  ))}
                </ul>
              </div>
            )}

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div>
                <label
                  htmlFor="input-attendance"
                  className="block text-xs font-medium text-slate-300"
                >
                  Attendance (%) <span className="text-slate-400">· 30–100</span>
                </label>
                <input
                  id="input-attendance"
                  type="number"
                  min={30}
                  max={100}
                  step={0.5}
                  value={formValues.attendance_pct}
                  onChange={(e) => handleTextChange('attendance_pct', e.target.value)}
                  className={inputClass}
                />
              </div>

              <div>
                <label htmlFor="input-quiz" className="block text-xs font-medium text-slate-300">
                  Quiz Average <span className="text-slate-400">· 0–10</span>
                </label>
                <input
                  id="input-quiz"
                  type="number"
                  min={0}
                  max={10}
                  step={0.1}
                  value={formValues.quiz_avg}
                  onChange={(e) => handleTextChange('quiz_avg', e.target.value)}
                  className={inputClass}
                />
              </div>

              <div>
                <label
                  htmlFor="input-assignment"
                  className="block text-xs font-medium text-slate-300"
                >
                  Assignment Average <span className="text-slate-400">· 0–10</span>
                </label>
                <input
                  id="input-assignment"
                  type="number"
                  min={0}
                  max={10}
                  step={0.1}
                  value={formValues.assignment_avg}
                  onChange={(e) => handleTextChange('assignment_avg', e.target.value)}
                  className={inputClass}
                />
              </div>

              <div>
                <label htmlFor="input-mid" className="block text-xs font-medium text-slate-300">
                  Midterm Marks <span className="text-slate-400">· 0–30</span>
                </label>
                <input
                  id="input-mid"
                  type="number"
                  min={0}
                  max={30}
                  step={0.5}
                  value={formValues.mid_marks}
                  onChange={(e) => handleTextChange('mid_marks', e.target.value)}
                  className={inputClass}
                />
              </div>

              <div>
                <label htmlFor="input-lab" className="block text-xs font-medium text-slate-300">
                  Lab Marks <span className="text-slate-400">· 0–20</span>
                </label>
                <input
                  id="input-lab"
                  type="number"
                  min={0}
                  max={20}
                  step={0.5}
                  value={formValues.lab_marks}
                  onChange={(e) => handleTextChange('lab_marks', e.target.value)}
                  className={inputClass}
                />
              </div>

              <div>
                <label htmlFor="input-hours" className="block text-xs font-medium text-slate-300">
                  Study Hours / Week <span className="text-slate-400">· 0–40</span>
                </label>
                <input
                  id="input-hours"
                  type="number"
                  min={0}
                  max={40}
                  step={0.5}
                  value={formValues.study_hours_week}
                  onChange={(e) => handleTextChange('study_hours_week', e.target.value)}
                  className={inputClass}
                />
              </div>

              <div>
                <label htmlFor="input-cgpa" className="block text-xs font-medium text-slate-300">
                  Previous CGPA <span className="text-slate-400">· 1.00–4.00</span>
                </label>
                <input
                  id="input-cgpa"
                  type="number"
                  min={1}
                  max={4}
                  step={0.05}
                  value={formValues.prev_cgpa}
                  onChange={(e) => handleTextChange('prev_cgpa', e.target.value)}
                  className={inputClass}
                />
              </div>

              <div>
                <label htmlFor="input-job" className="block text-xs font-medium text-slate-300">
                  Part-Time Job
                </label>
                <select
                  id="input-job"
                  value={partTimeJob}
                  onChange={(e) => {
                    const nextJob = e.target.value === 'Yes' ? 'Yes' : 'No';
                    setPartTimeJob(nextJob);
                    setSubmittedInput(null);
                    onUpdateDraft({ ...studentDraft, part_time_job: nextJob });
                  }}
                  className="mt-1.5 w-full rounded-xl border border-white/15 bg-slate-900/90 px-3 py-2 text-sm text-white focus:border-teal-400 focus:outline-none focus:ring-1 focus:ring-teal-400"
                >
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                </select>
              </div>
            </div>

            <div className="flex items-center justify-end border-t border-white/10 pt-4">
              <button
                type="submit"
                className="rounded-xl bg-white px-5 py-2.5 text-sm font-semibold text-slate-950 shadow-[0_10px_28px_-8px_rgba(255,255,255,0.4)] transition-colors hover:bg-slate-100 whitespace-nowrap"
              >
                Predict Student Risk
              </button>
            </div>
          </form>
        </GlassMotionCard>

        {/* Gated Prediction Output */}
        <GlassMotionCard
          variant="glass-strong"
          delay={0.11}
          className="flex flex-col justify-between p-6 lg:col-span-6"
        >
          {prediction ? (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.22, ease: EASE_OUT_EXPO }}
            >
              <div className="flex items-center justify-between border-b border-white/10 pb-4">
                <div>
                  <h2 className="font-display text-base font-semibold text-white">
                    Prediction &amp; Advisor Output
                  </h2>
                  <p className="text-xs text-slate-400">
                    Scored via{' '}
                    {activeModel === 'random_forest' ? 'Random Forest' : 'Logistic Regression'}
                  </p>
                </div>
                {prediction.riskTier === 'High' ? (
                  <ShieldAlert className="h-5 w-5 text-rose-400" aria-hidden="true" />
                ) : prediction.riskTier === 'Medium' ? (
                  <AlertCircle className="h-5 w-5 text-amber-400" aria-hidden="true" />
                ) : (
                  <CheckCircle2 className="h-5 w-5 text-emerald-400" aria-hidden="true" />
                )}
              </div>

              <div className="mt-4 grid grid-cols-1 items-center gap-4 sm:grid-cols-12">
                <div className="sm:col-span-7">
                  <PlotlyChart
                    data={riskGaugeData}
                    layout={{
                      paper_bgcolor: 'transparent',
                      plot_bgcolor: 'transparent',
                      margin: { t: 26, r: 36, b: 14, l: 36 },
                    }}
                    config={{ displayModeBar: false }}
                    height={185}
                    reportTitle="Predicted Student Failure Risk Gauge"
                    reportSubtitle={`${prediction.predictedResult} (${(prediction.failProbability * 100).toFixed(1)}% Fail Probability - ${prediction.riskTier} Risk)`}
                    ariaLabel="Semi-circular gauge showing predicted failure probability"
                  />
                </div>
                <div className="space-y-3 sm:col-span-5">
                  <div>
                    <p className="text-xs text-slate-400">Predicted Outcome</p>
                    <p
                      data-testid="predicted-outcome"
                      className={`mt-0.5 font-display text-xl font-bold ${
                        prediction.predictedResult === 'Fail'
                          ? 'text-rose-300'
                          : 'text-emerald-300'
                      }`}
                    >
                      {prediction.predictedResult}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-slate-400">Failure Probability</p>
                    <p
                      data-testid="fail-probability"
                      className="mt-0.5 font-mono text-xl font-bold text-white tabular-nums"
                    >
                      {(prediction.failProbability * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-slate-400">Risk Tier</p>
                    <p
                      data-testid="risk-tier"
                      className={`mt-0.5 font-display text-lg font-bold ${
                        prediction.riskTier === 'High'
                          ? 'text-rose-300'
                          : prediction.riskTier === 'Medium'
                          ? 'text-amber-300'
                          : 'text-emerald-300'
                      }`}
                    >
                      {prediction.riskTier} Risk
                    </p>
                  </div>
                </div>
              </div>

              {/* 7-Indicator Benchmark Strip */}
              <div className="mt-4 border-t border-white/10 pt-4">
                <p className="text-xs font-semibold text-slate-200">Academic Benchmark Check</p>
                <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs sm:grid-cols-4">
                  {benchmarkChecks.map((b) => (
                    <div key={b.label} className="border-b border-white/[0.06] pb-1">
                      <p className="text-[11px] text-slate-400">{b.label}</p>
                      <p
                        className={`font-mono font-semibold tabular-nums ${
                          b.ok ? 'text-emerald-300' : 'text-rose-300'
                        }`}
                      >
                        {b.val}
                      </p>
                      <p className="font-mono text-[10px] text-slate-400">{b.target}</p>
                    </div>
                  ))}
                </div>
              </div>

              <div className="mt-4 border-t border-white/10 pt-4">
                <h3 className="text-xs font-semibold text-slate-200">Detected Key Risk Signals</h3>
                <ul className="mt-2 space-y-1.5 text-xs text-slate-300">
                  {prediction.riskFactors.map((factor, idx) => (
                    <li key={idx} className="flex items-start gap-2">
                      <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-slate-400" />
                      <span>{factor}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <div className="mt-4 border-t border-white/10 pt-4">
                <h3 className="text-xs font-semibold text-slate-200">
                  Personalized Advisor Recommendations
                </h3>
                <ul className="mt-2 space-y-2 text-xs leading-relaxed text-slate-200">
                  {prediction.recommendations.map((rec, idx) => (
                    <li key={idx} className="flex items-start gap-2">
                      <span className="font-mono font-semibold text-teal-300">{idx + 1}.</span>
                      <span>{rec}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </motion.div>
          ) : (
            <div className="flex h-full flex-col justify-center py-10 text-center">
              <div className="glass mx-auto flex h-12 w-12 items-center justify-center rounded-xl text-slate-200">
                <Info className="h-5 w-5" aria-hidden="true" />
              </div>
              <h2 className="mt-4 font-display text-base font-semibold text-white">
                Ready to Predict Student Risk
              </h2>
              <p className="mx-auto mt-1.5 max-w-sm text-xs leading-relaxed text-slate-400">
                Review or enter all eight academic inputs on the left and press{' '}
                <strong className="text-slate-200">Predict Student Risk</strong> to render the
                probability gauge, benchmark audit, and personalized advising plan.
              </p>
            </div>
          )}
        </GlassMotionCard>
      </section>

      {/* Model Comparison Metrics */}
      <section aria-label="Trained ML models evaluation" className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        <ConfusionMatrixCard metrics={bundle.rfMetrics} delay={0.14} />
        <ConfusionMatrixCard metrics={bundle.lrMetrics} delay={0.18} />
      </section>

      {/* Explainability Section */}
      <motion.section
        aria-label="Model explainability"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.28, ease: EASE_OUT_EXPO, delay: 0.22 }}
        className={`${GLASS_CARD_CLASS} p-6`}
      >
        <div className="flex flex-col gap-3 border-b border-white/10 pb-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="font-display text-base font-semibold text-white">
              Model Explainability &amp; Feature Drivers
            </h2>
            <p className="text-xs text-slate-400">
              Inspect global Random Forest Gini importance or Logistic Regression standardized
              log-odds coefficients.
            </p>
          </div>

          <div className="glass flex items-center gap-1 rounded-xl p-1 shrink-0">
            <button
              type="button"
              onClick={() => setExplainView('rf')}
              className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
                explainView === 'rf'
                  ? 'bg-white text-slate-950 shadow-xs'
                  : 'text-slate-300 hover:text-white'
              }`}
            >
              Random Forest Importance
            </button>
            <button
              type="button"
              onClick={() => setExplainView('lr')}
              className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
                explainView === 'lr'
                  ? 'bg-white text-slate-950 shadow-xs'
                  : 'text-slate-300 hover:text-white'
              }`}
            >
              Logistic Regression Coefficients
            </button>
          </div>
        </div>

        <div className="mt-4 grid grid-cols-1 gap-6 lg:grid-cols-12">
          <div className="lg:col-span-6">
            <PlotlyChart
              data={explainChartData}
              layout={{
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                barcornerradius: 16,
                bargap: 0.34,
                margin: { t: 18, r: 24, b: 46, l: 130 },
                xaxis: {
                  title: {
                    text:
                      explainView === 'rf'
                        ? 'Relative Importance (%)'
                        : 'Log-Odds Coefficient (+ Increases Fail Risk)',
                  },
                  gridcolor: 'rgba(255,255,255,0.045)',
                  zeroline: true,
                  zerolinecolor: 'rgba(255,255,255,0.28)',
                },
                yaxis: { automargin: true, showgrid: false },
              }}
              height={325}
              reportTitle={
                explainView === 'rf'
                  ? 'Random Forest Feature Importance'
                  : 'Logistic Regression Log-Odds Coefficients'
              }
              reportSubtitle="Relative predictor contributions (predictive association only)"
              ariaLabel="Horizontal bar chart of model feature importance and coefficients"
            />
          </div>

          <div className="space-y-3 lg:col-span-6">
            <h3 className="text-xs font-semibold text-slate-200">
              Plain-Language Factor Interpretation
            </h3>
            <div className="divide-y divide-white/[0.06]">
              {activeExplainList.slice(0, 5).map((item) => (
                <div key={item.feature} className="py-2.5">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-medium text-white">{item.label}</span>
                    <span className="font-mono text-teal-300 tabular-nums">
                      {explainView === 'rf'
                        ? `${(item.importance * 100).toFixed(1)}% weight`
                        : `${(item.coefficient ?? 0) >= 0 ? '+' : ''}${(item.coefficient ?? 0).toFixed(3)} log-odds`}
                    </span>
                  </div>
                  <p className="mt-1 text-xs leading-relaxed text-slate-300">
                    {item.plainExplanation}
                  </p>
                </div>
              ))}
            </div>

            <div className="glass flex items-start gap-2.5 rounded-xl px-3.5 py-3 text-xs text-slate-200">
              <Info className="mt-0.5 h-4 w-4 shrink-0 text-teal-300" aria-hidden="true" />
              <p>
                <strong>Methodological Note:</strong> Correlation and model feature importance
                measure predictive association within historical records—they do{' '}
                <strong>not</strong> prove causation. Use predictions alongside human advising.
              </p>
            </div>
          </div>
        </div>
      </motion.section>
    </div>
  );
}
