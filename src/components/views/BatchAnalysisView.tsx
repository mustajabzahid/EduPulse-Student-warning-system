import { ChangeEvent, useEffect, useMemo, useRef, useState } from 'react';
import { AlertTriangle, Download, FileSpreadsheet, Upload } from 'lucide-react';
import { BatchPredictionRow, ModelType, RiskTier, TrainedModelBundle } from '../../types/student';
import { parseAndValidateCsv } from '../../ml/cleaning';
import { batchPredictionsToCsv, predictBatchStudents } from '../../ml/pipeline';
import { ViewReportContext } from '../../utils/reportData';
import { GlassMotionCard } from '../MotionPrimitives';

interface BatchAnalysisViewProps {
  bundle: TrainedModelBundle;
  activeModel: ModelType;
  onReportContextChange?: (patch: Partial<ViewReportContext>) => void;
}

const SAMPLE_CLASS_CSV = `student_id,attendance_pct,quiz_avg,assignment_avg,mid_marks,lab_marks,study_hours_week,prev_cgpa,part_time_job
CLS-201,51.0,4.2,4.8,11.5,9.5,4.5,1.95,Yes
CLS-202,94.0,8.8,9.1,26.0,18.5,18.0,3.65,No
CLS-203,68.5,5.6,6.1,16.0,12.0,8.5,2.45,Yes
CLS-204,82.0,7.6,7.9,22.0,15.5,14.0,3.15,No
CLS-205,44.0,3.6,4.2,10.0,8.0,3.0,1.72,Yes
CLS-206,77.0,7.0,7.4,20.5,14.5,11.5,2.82,No
CLS-207,61.0,5.2,5.8,14.5,11.0,7.0,2.28,Yes
CLS-208,89.5,8.4,8.6,24.5,17.0,16.5,3.48,No
CLS-209,72.0,,6.4,17.0,12.5,9.5,2.55,
CLS-210,56.5,4.9,5.3,13.5,10.5,6.0,2.10,Yes
CLS-211,96.0,9.2,9.4,27.5,19.0,21.0,3.88,No
CLS-212,74.5,6.5,6.9,18.5,13.5,10.5,2.71,No`;

export function BatchAnalysisView({
  bundle,
  activeModel,
  onReportContextChange,
}: BatchAnalysisViewProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [batchSourceName, setBatchSourceName] = useState<string>('Sample Class Roster (12 Students)');
  const [rawCsv, setRawCsv] = useState<string>(SAMPLE_CLASS_CSV);
  const [tierFilter, setTierFilter] = useState<'All' | RiskTier>('All');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);

  const predictions: BatchPredictionRow[] = useMemo(() => {
    const parsed = parseAndValidateCsv(rawCsv, false);
    if (!parsed.ok) {
      return [];
    }
    return predictBatchStudents(parsed.rows, bundle, activeModel);
  }, [rawCsv, bundle, activeModel]);

  const filteredPredictions = useMemo(() => {
    if (tierFilter === 'All') return predictions;
    return predictions.filter((r) => r.risk_tier === tierFilter);
  }, [predictions, tierFilter]);

  useEffect(() => {
    onReportContextChange?.({
      batch: {
        batchSourceName,
        tierFilter,
        predictions,
        filteredPredictions,
      },
    });
  }, [onReportContextChange, batchSourceName, tierFilter, predictions, filteredPredictions]);

  const handleCsvInput = (csvText: string, fileName: string) => {
    const parsed = parseAndValidateCsv(csvText, false);
    if (!parsed.ok) {
      setErrorMessage(parsed.error ?? 'Invalid CSV file.');
      setWarnings([]);
      return;
    }
    setErrorMessage(null);
    setWarnings(parsed.warnings);
    setBatchSourceName(fileName);
    setRawCsv(csvText);
    setTierFilter('All');
  };

  const handleFileUpload = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        handleCsvInput(reader.result, file.name);
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  const handleDownloadCsv = () => {
    if (predictions.length === 0) return;
    const csvOutput = batchPredictionsToCsv(predictions);
    const blob = new Blob([csvOutput], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', 'edupulse_batch_predictions.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const highCount = predictions.filter((r) => r.risk_tier === 'High').length;
  const medCount = predictions.filter((r) => r.risk_tier === 'Medium').length;
  const lowCount = predictions.filter((r) => r.risk_tier === 'Low').length;
  const total = Math.max(1, predictions.length);

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 xl:flex-row xl:items-end xl:justify-between">
        <div className="max-w-xl">
          <h1 className="font-display text-2xl font-bold tracking-tight text-white">
            Batch Class Prediction
          </h1>
          <p className="mt-1 text-sm text-slate-400">
            Upload any class roster CSV to predict failure probabilities, rank students from highest
            to lowest risk, and export advisor rosters.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2 shrink-0">
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,text/csv"
            onChange={handleFileUpload}
            aria-label="Upload class CSV for batch scoring"
            className="sr-only"
          />
          <button
            type="button"
            onClick={() => handleCsvInput(SAMPLE_CLASS_CSV, 'Sample Class Roster (12 Students)')}
            className="rounded-xl border border-white/15 bg-white/[0.06] px-3.5 py-2 text-xs font-medium text-slate-200 backdrop-blur-md transition-colors hover:bg-white/[0.12] whitespace-nowrap shrink-0"
          >
            Load Sample Class CSV
          </button>
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="inline-flex items-center gap-1.5 rounded-xl bg-white px-3.5 py-2 text-xs font-semibold text-slate-950 shadow-[0_10px_28px_-8px_rgba(255,255,255,0.4)] transition-colors hover:bg-slate-100 whitespace-nowrap shrink-0"
          >
            <Upload className="h-3.5 w-3.5" aria-hidden="true" />
            <span>Upload Class CSV</span>
          </button>
          <button
            type="button"
            onClick={handleDownloadCsv}
            disabled={predictions.length === 0}
            className="inline-flex items-center gap-1.5 rounded-xl border border-emerald-400/35 bg-emerald-500/20 px-3.5 py-2 text-xs font-medium text-emerald-200 transition-colors hover:bg-emerald-500/30 disabled:opacity-50 whitespace-nowrap shrink-0"
          >
            <Download className="h-3.5 w-3.5" aria-hidden="true" />
            <span>Download Predictions CSV</span>
          </button>
        </div>
      </div>

      {errorMessage && (
        <div
          role="alert"
          className="flex items-start gap-2.5 rounded-xl border border-rose-500/30 bg-rose-500/15 px-4 py-3 text-sm text-rose-200 backdrop-blur-md"
        >
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-rose-400" aria-hidden="true" />
          <div>
            <p className="font-semibold text-rose-100">Invalid Batch CSV File</p>
            <p className="mt-0.5 text-xs text-rose-200/90">{errorMessage}</p>
          </div>
        </div>
      )}

      {warnings.length > 0 && (
        <div
          role="status"
          className="rounded-xl border border-amber-500/30 bg-amber-500/15 px-4 py-3 text-xs text-amber-200 backdrop-blur-md"
        >
          {warnings.map((w, i) => (
            <p key={i}>{w}</p>
          ))}
        </div>
      )}

      {/* Summary Strip + Proportional Risk Composition Bar */}
      <GlassMotionCard delay={0.04} className="space-y-4 p-5">
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <div>
            <p className="text-[11px] font-medium uppercase tracking-wider text-slate-400">Roster Size</p>
            <p className="mt-1 font-mono text-2xl font-semibold text-white tabular-nums">
              {predictions.length}
            </p>
            <p className="mt-0.5 truncate text-xs text-slate-400">{batchSourceName}</p>
          </div>
          <div>
            <p className="text-[11px] font-medium uppercase tracking-wider text-slate-400">High Risk (≥65%)</p>
            <p className="mt-1 font-mono text-2xl font-semibold text-rose-400 tabular-nums">
              {highCount}
            </p>
            <p className="mt-0.5 text-xs text-slate-400">Immediate advising</p>
          </div>
          <div>
            <p className="text-[11px] font-medium uppercase tracking-wider text-slate-400">Medium Risk (35–65%)</p>
            <p className="mt-1 font-mono text-2xl font-semibold text-amber-300 tabular-nums">
              {medCount}
            </p>
            <p className="mt-0.5 text-xs text-slate-400">Preventive check-in</p>
          </div>
          <div>
            <p className="text-[11px] font-medium uppercase tracking-wider text-slate-400">Low Risk (&lt;35%)</p>
            <p className="mt-1 font-mono text-2xl font-semibold text-emerald-400 tabular-nums">
              {lowCount}
            </p>
            <p className="mt-0.5 text-xs text-slate-400">On-track progress</p>
          </div>
        </div>

        <div className="pt-1">
          <div className="flex h-2 w-full overflow-hidden rounded-full bg-white/10">
            <div
              className="bg-rose-500 transition-all duration-300"
              style={{ width: `${((highCount / total) * 100).toFixed(1)}%` }}
              title={`High Risk: ${highCount}`}
            />
            <div
              className="bg-amber-400 transition-all duration-300"
              style={{ width: `${((medCount / total) * 100).toFixed(1)}%` }}
              title={`Medium Risk: ${medCount}`}
            />
            <div
              className="bg-emerald-500 transition-all duration-300"
              style={{ width: `${((lowCount / total) * 100).toFixed(1)}%` }}
              title={`Low Risk: ${lowCount}`}
            />
          </div>
        </div>
      </GlassMotionCard>

      {/* Sorted Batch Results Table */}
      <GlassMotionCard as="section" variant="glass-strong" delay={0.1} className="overflow-hidden p-0">
        <div className="flex flex-col gap-3 border-b border-white/10 px-5 py-3.5 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-2">
            <FileSpreadsheet className="h-4 w-4 text-emerald-400" aria-hidden="true" />
            <h2 className="font-display text-sm font-semibold text-white">
              Sorted Batch Predictions (Highest Risk First)
            </h2>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <div
              role="group"
              aria-label="Filter batch predictions by tier"
              className="flex items-center gap-1 rounded-lg border border-white/10 bg-white/[0.04] p-0.5 backdrop-blur-sm"
            >
              {(['All', 'High', 'Medium', 'Low'] as const).map((tier) => (
                <button
                  key={tier}
                  type="button"
                  onClick={() => setTierFilter(tier)}
                  className={`rounded-md px-2.5 py-1 text-[11px] font-medium transition-colors whitespace-nowrap ${
                    tierFilter === tier
                      ? 'bg-white text-slate-950 shadow-xs'
                      : 'text-slate-300 hover:text-white'
                  }`}
                >
                  {tier}
                </button>
              ))}
            </div>
            <span className="font-mono text-xs text-slate-400">
              {activeModel === 'random_forest' ? 'Random Forest' : 'Logistic Regression'}
            </span>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-left text-xs">
            <thead>
              <tr className="border-b border-white/10 bg-white/[0.04] text-slate-400 backdrop-blur-sm">
                <th className="py-2.5 pl-5 pr-3 font-medium">Rank</th>
                <th className="py-2.5 px-3 font-medium">Student ID</th>
                <th className="py-2.5 px-3 text-right font-medium">Fail Prob.</th>
                <th className="py-2.5 px-3 font-medium">Risk Tier</th>
                <th className="py-2.5 px-3 font-medium">Predicted</th>
                <th className="py-2.5 px-3 text-right font-medium">Attendance</th>
                <th className="py-2.5 px-3 text-right font-medium">CGPA</th>
                <th className="py-2.5 pl-3 pr-5 font-medium">Primary Advisor Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.07]">
              {filteredPredictions.map((row, idx) => {
                const barColor =
                  row.risk_tier === 'High'
                    ? 'bg-rose-500'
                    : row.risk_tier === 'Medium'
                    ? 'bg-amber-400'
                    : 'bg-emerald-500';

                return (
                  <tr
                    key={`${row.student_id}-${idx}`}
                    className="transition-colors hover:bg-white/[0.025]"
                  >
                    <td className="py-2.5 pl-5 pr-3 font-mono text-slate-400 tabular-nums">
                      #{idx + 1}
                    </td>
                    <td className="py-2.5 px-3 font-mono font-medium text-white">
                      {row.student_id}
                      {row.imputed_fields.length > 0 && (
                        <span className="ml-1.5 text-[10px] font-normal text-slate-400">
                          ({row.imputed_fields.length} imputed)
                        </span>
                      )}
                    </td>
                    <td className="py-2.5 px-3 text-right">
                      <div className="inline-flex items-center justify-end gap-2">
                        <div className="hidden h-1.5 w-14 overflow-hidden rounded-full bg-white/10 sm:block">
                          <div
                            className={`h-full ${barColor}`}
                            style={{
                              width: `${Math.min(100, Math.max(3, row.fail_probability * 100)).toFixed(0)}%`,
                            }}
                          />
                        </div>
                        <span className="font-mono font-semibold text-white tabular-nums">
                          {(row.fail_probability * 100).toFixed(1)}%
                        </span>
                      </div>
                    </td>
                    <td className="py-2.5 px-3 font-semibold">
                      <span
                        className={
                          row.risk_tier === 'High'
                            ? 'text-rose-400'
                            : row.risk_tier === 'Medium'
                            ? 'text-amber-300'
                            : 'text-emerald-400'
                        }
                      >
                        {row.risk_tier}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 font-medium">
                      <span
                        className={
                          row.predicted_result === 'Fail' ? 'text-rose-400' : 'text-emerald-400'
                        }
                      >
                        {row.predicted_result}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono text-slate-300 tabular-nums">
                      {row.attendance_pct.toFixed(1)}%
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono text-slate-300 tabular-nums">
                      {row.prev_cgpa.toFixed(2)}
                    </td>
                    <td className="max-w-sm py-2.5 pl-3 pr-5 text-slate-300">
                      {row.primary_recommendation}
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
