import {
  BatchPredictionRow,
  CleanedStudentRecord,
  DataCleaningSummary,
  ModelType,
  NumericFeatureKey,
  RiskTier,
  ScoredStudentRecord,
  SinglePredictionResult,
  StudentFeatureInput,
  TrainedModelBundle,
} from '../types/student';
import { FEATURE_LABELS } from '../ml/pipeline';
import { NUMERIC_FEATURE_KEYS } from '../ml/cleaning';
import { buildCgpaGrowthTrend } from './analyticsTrends';
import { NavSection } from '../components/Sidebar';

export interface ReportMetric {
  label: string;
  value: string;
  note?: string;
  accent?: 'navy' | 'teal' | 'red' | 'amber';
}

export interface ReportSectionBlock {
  title: string;
  items: string[];
}

export interface ReportTable {
  title: string;
  headers: string[];
  rows: string[][];
}

export interface ReportVectorBar {
  title: string;
  subtitle?: string;
  segments: Array<{
    label: string;
    count: number;
    colorHex: string;
  }>;
}

export interface ViewReportContext {
  prediction?: {
    submittedInput: StudentFeatureInput | null;
    result: SinglePredictionResult | null;
    explainView: 'rf' | 'lr';
    draftInput: StudentFeatureInput;
  };
  register?: {
    tierFilter: 'All' | RiskTier;
    searchQuery: string;
    filteredRows: Array<
      ScoredStudentRecord & { activeFailProb: number; activeTier: RiskTier }
    >;
    selectedStudent:
      | (ScoredStudentRecord & { activeFailProb: number; activeTier: RiskTier })
      | null;
  };
  batch?: {
    batchSourceName: string;
    tierFilter: 'All' | RiskTier;
    predictions: BatchPredictionRow[];
    filteredPredictions: BatchPredictionRow[];
  };
  whatif?: {
    simulatedAttendance: number;
    studentLabel: string;
    baseStudent: StudentFeatureInput;
    baselinePred: SinglePredictionResult;
    simulatedPred: SinglePredictionResult;
  };
}

export interface ViewReportSpec {
  section: NavSection;
  title: string;
  subtitle: string;
  filename: string;
  generatedAt: string;
  datasetName: string;
  modelLabel: string;
  metrics: ReportMetric[];
  highlights: ReportSectionBlock[];
  vectorChart?: ReportVectorBar;
  table?: ReportTable;
  notes: string[];
}

export interface BuildReportParams {
  activeSection: NavSection;
  activeModel: ModelType;
  datasetName: string;
  summary: DataCleaningSummary;
  trainingRows: CleanedStudentRecord[];
  bundle: TrainedModelBundle;
  context?: ViewReportContext;
  now?: Date;
}

/**
 * Normalizes Unicode glyphs unsupported by standard WinAnsi PDF fonts into crisp ASCII equivalents.
 */
export function toPdfSafeText(input: string): string {
  return input
    .replaceAll('≥', '>=')
    .replaceAll('≤', '<=')
    .replaceAll('×', 'x')
    .replaceAll('−', '-')
    .replaceAll('–', '-')
    .replaceAll('—', ' - ')
    .replaceAll('·', '|')
    .replaceAll('→', '->')
    .replaceAll('‘', "'")
    .replaceAll('’', "'")
    .replaceAll('“', '"')
    .replaceAll('”', '"')
    .replaceAll('…', '...')
    .replace(/[^\x20-\x7E\n]/g, ' ')
    .replace(/[ \t]{2,}/g, ' ')
    .trim();
}

export function formatReportFilename(section: NavSection, now: Date = new Date()): string {
  const yyyy = now.getFullYear();
  const mm = String(now.getMonth() + 1).padStart(2, '0');
  const dd = String(now.getDate()).padStart(2, '0');
  const hh = String(now.getHours()).padStart(2, '0');
  const min = String(now.getMinutes()).padStart(2, '0');
  return `edupulse-${section}-report-${yyyy}${mm}${dd}-${hh}${min}.pdf`;
}

function mean(arr: number[]): number {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function formatTimestamp(now: Date): string {
  return now.toISOString().slice(0, 16).replace('T', ' ') + ' UTC';
}

export function buildViewReportSpec({
  activeSection,
  activeModel,
  datasetName,
  summary,
  trainingRows,
  bundle,
  context = {},
  now = new Date(),
}: BuildReportParams): ViewReportSpec {
  const activeMetrics =
    activeModel === 'logistic_regression' ? bundle.lrMetrics : bundle.rfMetrics;
  const modelLabel = activeMetrics.modelName;
  const generatedAt = formatTimestamp(now);
  const filename = formatReportFilename(activeSection, now);
  const safeDataset = toPdfSafeText(datasetName);

  const getProb = (s: ScoredStudentRecord) =>
    activeModel === 'logistic_regression' ? s.lrFailProbability : s.rfFailProbability;

  if (activeSection === 'overview') {
    const totalScored = Math.max(1, bundle.scoredCohort.length);
    const highRiskCount = bundle.scoredCohort.filter((s) => getProb(s) >= 0.65).length;
    const medRiskCount = bundle.scoredCohort.filter((s) => {
      const p = getProb(s);
      return p >= 0.35 && p < 0.65;
    }).length;
    const lowRiskCount = bundle.scoredCohort.filter((s) => getProb(s) < 0.35).length;
    const passRate =
      summary.cleanedTrainingRows > 0
        ? (summary.passCount / summary.cleanedTrainingRows) * 100
        : 0;
    const failRate = 100 - passRate;

    return {
      section: 'overview',
      title: 'Academic Cohort Overview Report',
      subtitle: 'Executive early-warning cohort KPIs, distributions, and preprocessing audit',
      filename,
      generatedAt,
      datasetName: safeDataset,
      modelLabel,
      metrics: [
        {
          label: 'Total Cleaned Students',
          value: String(summary.cleanedTrainingRows),
          note: `${summary.originalRows} raw | ${summary.duplicateRowsRemoved} duplicates removed`,
          accent: 'navy',
        },
        {
          label: 'Cohort Pass Rate',
          value: `${passRate.toFixed(1)}%`,
          note: `${summary.passCount} passing records`,
          accent: 'teal',
        },
        {
          label: 'Recorded Fail Count',
          value: String(summary.failCount),
          note: `${failRate.toFixed(1)}% historical fail rate`,
          accent: 'red',
        },
        {
          label: 'High-Risk Students',
          value: String(highRiskCount),
          note: `${((highRiskCount / totalScored) * 100).toFixed(1)}% of cohort (P(Fail) >= 65%)`,
          accent: 'amber',
        },
      ],
      highlights: [
        {
          title: 'Active Pipeline & Risk Segmentation',
          items: [
            `Active Model: ${modelLabel} | Accuracy ${(activeMetrics.accuracy * 100).toFixed(1)}% | Fail Recall ${(activeMetrics.failRecall * 100).toFixed(1)}% | Fail Precision ${(activeMetrics.failPrecision * 100).toFixed(1)}% | ROC-AUC ${activeMetrics.rocAuc.toFixed(3)}.`,
            `Risk Distribution: ${highRiskCount} High Risk (>=65%), ${medRiskCount} Medium Risk (35-65%), ${lowRiskCount} Low Risk (<35%).`,
            `Cleaning Audit: ${summary.totalMissingCells} missing cells imputed (median/mode fit inside train split), ${summary.missingFinalResultExcluded} missing-target rows excluded, ${summary.outOfRangeCount} out-of-range values flagged.`,
          ],
        },
      ],
      table: {
        title: 'Data Cleaning & Imputation Summary',
        headers: ['Column', 'Missing Cells', 'Cleaning Rule', 'Cohort Value / Role'],
        rows: summary.missingByColumn.map((col) => {
          let rule = 'Median (train fold)';
          let val = '-';
          if (col.column === 'student_id') {
            rule = 'Identifier only';
            val = 'Excluded from ML';
          } else if (col.column === 'part_time_job') {
            rule = 'Mode (train fold)';
            val = summary.categoricalMode;
          } else if (col.column === 'final_result') {
            rule = 'Drop missing target';
            val = `${summary.missingFinalResultExcluded} excluded`;
          } else {
            const med = summary.numericMedians[col.column as NumericFeatureKey];
            val = med !== undefined ? String(med) : '-';
          }
          return [col.column, String(col.missingCount), rule, val];
        }),
      },
      notes: [
        'student_id and final_result are strictly excluded from model input features to prevent leakage.',
      ],
    };
  }

  if (activeSection === 'analytics') {
    const passRows = trainingRows.filter((r) => r.final_result === 'Pass');
    const failRows = trainingRows.filter((r) => r.final_result === 'Fail');
    const passAtt = mean(passRows.map((r) => r.attendance_pct));
    const failAtt = mean(failRows.map((r) => r.attendance_pct));
    const passMid = mean(passRows.map((r) => r.mid_marks));
    const failMid = mean(failRows.map((r) => r.mid_marks));
    const passCgpa = mean(passRows.map((r) => r.prev_cgpa));
    const failCgpa = mean(failRows.map((r) => r.prev_cgpa));
    const dualRiskCount = trainingRows.filter(
      (r) => r.attendance_pct < 75 && r.mid_marks < 18
    ).length;
    const cgpaTrend = buildCgpaGrowthTrend(trainingRows);

    return {
      section: 'analytics',
      title: 'Exploratory Academic Analytics Report',
      subtitle: 'Multivariate Pass vs Fail score benchmarks, risk quadrants, and correlations',
      filename,
      generatedAt,
      datasetName: safeDataset,
      modelLabel,
      metrics: [
        {
          label: 'Attendance Mean Gap',
          value: `+${(passAtt - failAtt).toFixed(1)} pp`,
          note: `Pass ${passAtt.toFixed(1)}% vs Fail ${failAtt.toFixed(1)}%`,
          accent: 'teal',
        },
        {
          label: 'Midterm Mean Gap',
          value: `+${(passMid - failMid).toFixed(1)} /30`,
          note: `Pass ${passMid.toFixed(1)} vs Fail ${failMid.toFixed(1)}`,
          accent: 'navy',
        },
        {
          label: 'Prev. CGPA Mean Gap',
          value: `+${(passCgpa - failCgpa).toFixed(2)}`,
          note: `Pass ${passCgpa.toFixed(2)} vs Fail ${failCgpa.toFixed(2)}`,
          accent: 'teal',
        },
        {
          label: 'Dual-Risk Quadrant',
          value: String(dualRiskCount),
          note: 'Students <75% attendance & <18/30 midterm',
          accent: 'red',
        },
      ],
      highlights: [
        {
          title: 'Exploratory Cohort Benchmarks',
          items: [
            `Cohort Composition: ${passRows.length} Pass records vs ${failRows.length} Fail records (${trainingRows.length} cleaned total).`,
            `Dual-risk boundary (<75% attendance and <18/30 midterm) captures ${dualRiskCount} students (${trainingRows.length > 0 ? ((dualRiskCount / trainingRows.length) * 100).toFixed(1) : '0.0'}% of cohort).`,
            `Ordinal Prior CGPA Growth (${toPdfSafeText(cgpaTrend.lowestBandLabel)} -> ${toPdfSafeText(cgpaTrend.highestBandLabel)}): Midterm ${cgpaTrend.midGrowthPctPoints >= 0 ? '+' : ''}${cgpaTrend.midGrowthPctPoints.toFixed(1)} pp and Quiz ${cgpaTrend.quizGrowthPctPoints >= 0 ? '+' : ''}${cgpaTrend.quizGrowthPctPoints.toFixed(1)} pp across CGPA tiers.`,
          ],
        },
      ],
      table: {
        title: 'Pass vs Fail Cohort Feature Benchmarks',
        headers: ['Academic Feature', 'Pass Mean', 'Fail Mean', 'Mean Gap (Pass - Fail)'],
        rows: NUMERIC_FEATURE_KEYS.map((key) => {
          const pMean = mean(passRows.map((r) => r[key]));
          const fMean = mean(failRows.map((r) => r[key]));
          const decimals = key === 'prev_cgpa' ? 2 : 1;
          return [
            toPdfSafeText(FEATURE_LABELS[key] ?? key),
            pMean.toFixed(decimals),
            fMean.toFixed(decimals),
            `+${(pMean - fMean).toFixed(decimals)}`,
          ];
        }),
      },
      notes: [
        'Pearson correlations and cohort mean gaps describe historical associations only and do not imply causation.',
      ],
    };
  }

  if (activeSection === 'prediction') {
    const predCtx = context.prediction;
    const submitted = predCtx?.submittedInput ?? null;
    const result = predCtx?.result ?? null;
    const explainView = predCtx?.explainView ?? 'rf';
    const explainList =
      explainView === 'rf' ? bundle.rfFeatureImportance : bundle.lrFeatureImportance;

    const metrics: ReportMetric[] = result
      ? [
          {
            label: 'Predicted Outcome',
            value: result.predictedResult,
            note: `Active model: ${modelLabel}`,
            accent: result.predictedResult === 'Fail' ? 'red' : 'teal',
          },
          {
            label: 'Failure Probability',
            value: `${(result.failProbability * 100).toFixed(1)}%`,
            note: `Pass probability ${(result.passProbability * 100).toFixed(1)}%`,
            accent: result.riskTier === 'High' ? 'red' : result.riskTier === 'Medium' ? 'amber' : 'teal',
          },
          {
            label: 'Assigned Risk Tier',
            value: `${result.riskTier} Risk`,
            note: 'Thresholds: <35% Low | 35-65% Med | >=65% High',
            accent: result.riskTier === 'High' ? 'red' : result.riskTier === 'Medium' ? 'amber' : 'teal',
          },
          {
            label: 'Model Fail Recall / AUC',
            value: `${(activeMetrics.failRecall * 100).toFixed(1)}%`,
            note: `Fail Precision ${(activeMetrics.failPrecision * 100).toFixed(1)}% | AUC ${activeMetrics.rocAuc.toFixed(3)}`,
            accent: 'navy',
          },
        ]
      : [
          {
            label: 'Prediction Status',
            value: 'Pending Submit',
            note: 'Click Predict Student Risk to score input',
            accent: 'amber',
          },
          {
            label: 'RF Accuracy / AUC',
            value: `${(bundle.rfMetrics.accuracy * 100).toFixed(1)}%`,
            note: `Fail Recall ${(bundle.rfMetrics.failRecall * 100).toFixed(1)}% | AUC ${bundle.rfMetrics.rocAuc.toFixed(3)}`,
            accent: 'teal',
          },
          {
            label: 'LR Accuracy / AUC',
            value: `${(bundle.lrMetrics.accuracy * 100).toFixed(1)}%`,
            note: `Fail Recall ${(bundle.lrMetrics.failRecall * 100).toFixed(1)}% | AUC ${bundle.lrMetrics.rocAuc.toFixed(3)}`,
            accent: 'navy',
          },
          {
            label: 'Held-Out Test Size',
            value: String(activeMetrics.testSize),
            note: `Stratified split (Train ${activeMetrics.trainSize})`,
            accent: 'navy',
          },
        ];

    const highlights: ReportSectionBlock[] = [];
    if (submitted && result) {
      highlights.push({
        title: 'Submitted Student Profile (8 Features)',
        items: [
          `Attendance: ${submitted.attendance_pct.toFixed(1)}% | Quiz: ${submitted.quiz_avg.toFixed(1)}/10 | Assignment: ${submitted.assignment_avg.toFixed(1)}/10 | Midterm: ${submitted.mid_marks.toFixed(1)}/30`,
          `Lab: ${submitted.lab_marks.toFixed(1)}/20 | Study Hours: ${submitted.study_hours_week.toFixed(1)} hrs/wk | Prev CGPA: ${submitted.prev_cgpa.toFixed(2)} | Part-Time Job: ${submitted.part_time_job}`,
        ],
      });
      highlights.push({
        title: 'Personalized Advisor Recommendations',
        items: result.recommendations.map((r) => toPdfSafeText(r)),
      });
    }

    highlights.push({
      title: `Top Predictive Drivers (${explainView === 'rf' ? 'Random Forest Importance' : 'Logistic Regression Log-Odds'})`,
      items: explainList.slice(0, 5).map((item) => {
        const stat =
          explainView === 'rf'
            ? `${(item.importance * 100).toFixed(1)}% weight`
            : `${(item.coefficient ?? 0) >= 0 ? '+' : ''}${(item.coefficient ?? 0).toFixed(3)} log-odds`;
        return `${toPdfSafeText(item.label)} (${stat}): ${toPdfSafeText(item.plainExplanation)}`;
      }),
    });

    return {
      section: 'prediction',
      title: 'AI Prediction & Model Explainability Report',
      subtitle: 'Stratified model metrics (Fail positive class), individual prediction, and feature drivers',
      filename,
      generatedAt,
      datasetName: safeDataset,
      modelLabel,
      metrics,
      highlights,
      table: {
        title: 'Held-Out Stratified Model Evaluation Summary (Positive Class: Fail)',
        headers: ['Model', 'Accuracy', 'Fail Precision', 'Fail Recall', 'ROC-AUC', 'TP / FN / FP / TN'],
        rows: [bundle.rfMetrics, bundle.lrMetrics].map((m) => [
          m.modelName,
          `${(m.accuracy * 100).toFixed(1)}%`,
          `${(m.failPrecision * 100).toFixed(1)}%`,
          `${(m.failRecall * 100).toFixed(1)}%`,
          m.rocAuc.toFixed(3),
          `${m.confusionMatrix.tp} / ${m.confusionMatrix.fn} / ${m.confusionMatrix.fp} / ${m.confusionMatrix.tn}`,
        ]),
      },
      notes: [
        'Methodological Note: Model feature importance and coefficients quantify predictive association in historical records and do not prove causation.',
      ],
    };
  }

  if (activeSection === 'register') {
    const regCtx = context.register;
    const allEnriched = bundle.scoredCohort.map((s) => {
      const p = getProb(s);
      return {
        ...s,
        activeFailProb: p,
        activeTier: (p >= 0.65 ? 'High' : p >= 0.35 ? 'Medium' : 'Low') as RiskTier,
      };
    });
    const rows = regCtx?.filteredRows ?? allEnriched.sort((a, b) => b.activeFailProb - a.activeFailProb);
    const tierFilter = regCtx?.tierFilter ?? 'All';
    const selected = regCtx?.selectedStudent ?? rows[0] ?? null;
    const highCount = allEnriched.filter((s) => s.activeTier === 'High').length;
    const medCount = allEnriched.filter((s) => s.activeTier === 'Medium').length;
    const lowCount = allEnriched.filter((s) => s.activeTier === 'Low').length;

    const highlights: ReportSectionBlock[] = [];
    if (selected) {
      highlights.push({
        title: `Selected Student Diagnostic: ${selected.student_id} (${(selected.activeFailProb * 100).toFixed(1)}% - ${selected.activeTier} Risk)`,
        items: [
          `Attendance: ${selected.attendance_pct.toFixed(1)}% | Quiz: ${selected.quiz_avg.toFixed(1)}/10 | Assignment: ${selected.assignment_avg.toFixed(1)}/10 | Midterm: ${selected.mid_marks.toFixed(1)}/30 | Lab: ${selected.lab_marks.toFixed(1)}/20 | CGPA: ${selected.prev_cgpa.toFixed(2)} | Job: ${selected.part_time_job}`,
          ...selected.recommendations.slice(0, 3).map((r) => toPdfSafeText(r)),
        ],
      });
    }

    return {
      section: 'register',
      title: 'Student Risk Register Report',
      subtitle: 'Cohort risk stratification, selected student diagnostic, and priority advising queue',
      filename,
      generatedAt,
      datasetName: safeDataset,
      modelLabel,
      metrics: [
        {
          label: 'Filtered Cohort Shown',
          value: String(rows.length),
          note: `Filter: ${tierFilter}${regCtx?.searchQuery ? ` | Search: "${toPdfSafeText(regCtx.searchQuery)}"` : ''}`,
          accent: 'navy',
        },
        {
          label: 'High Risk (>=65%)',
          value: String(highCount),
          note: 'Immediate intervention queue',
          accent: 'red',
        },
        {
          label: 'Medium Risk (35-65%)',
          value: String(medCount),
          note: 'Preventive check-in queue',
          accent: 'amber',
        },
        {
          label: 'Low Risk (<35%)',
          value: String(lowCount),
          note: 'On-track students',
          accent: 'teal',
        },
      ],
      highlights,
      vectorChart: {
        title: 'Cohort Risk Tier Composition',
        subtitle: `Total scored cohort: ${allEnriched.length} students`,
        segments: [
          { label: 'High Risk', count: highCount, colorHex: '#dc2626' },
          { label: 'Medium Risk', count: medCount, colorHex: '#d97706' },
          { label: 'Low Risk', count: lowCount, colorHex: '#0d9488' },
        ],
      },
      table: {
        title: `Priority Risk Register Snapshot (Top ${Math.min(15, rows.length)} of ${rows.length} Filtered)`,
        headers: ['Student ID', 'Fail Prob', 'Tier', 'Attend.', 'Midterm', 'CGPA', 'Outcome'],
        rows: rows.slice(0, 15).map((s) => [
          toPdfSafeText(s.student_id),
          `${(s.activeFailProb * 100).toFixed(1)}%`,
          s.activeTier,
          `${s.attendance_pct.toFixed(1)}%`,
          `${s.mid_marks.toFixed(1)}/30`,
          s.prev_cgpa.toFixed(2),
          s.final_result ?? '-',
        ]),
      },
      notes: [
        'Risk probabilities are produced by the trained model pipeline with student_id excluded from predictors.',
      ],
    };
  }

  if (activeSection === 'batch') {
    const batchCtx = context.batch;
    const predictions = batchCtx?.predictions ?? [];
    const filtered = batchCtx?.filteredPredictions ?? predictions;
    const sourceName = batchCtx?.batchSourceName ?? 'Batch Class Roster';
    const highCount = predictions.filter((r) => r.risk_tier === 'High').length;
    const medCount = predictions.filter((r) => r.risk_tier === 'Medium').length;
    const lowCount = predictions.filter((r) => r.risk_tier === 'Low').length;
    const avgFailProb =
      predictions.length > 0
        ? (predictions.reduce((acc, r) => acc + r.fail_probability, 0) / predictions.length) * 100
        : 0;

    return {
      section: 'batch',
      title: 'Batch Class Risk Analysis Report',
      subtitle: `Roster-wide risk ranking for ${toPdfSafeText(sourceName)}`,
      filename,
      generatedAt,
      datasetName: safeDataset,
      modelLabel,
      metrics: [
        {
          label: 'Batch Roster Size',
          value: String(predictions.length),
          note: `Mean Fail Prob: ${avgFailProb.toFixed(1)}%`,
          accent: 'navy',
        },
        {
          label: 'High Risk (>=65%)',
          value: String(highCount),
          note: 'Immediate advising needed',
          accent: 'red',
        },
        {
          label: 'Medium Risk (35-65%)',
          value: String(medCount),
          note: 'Preventive check-in',
          accent: 'amber',
        },
        {
          label: 'Low Risk (<35%)',
          value: String(lowCount),
          note: 'On-track progress',
          accent: 'teal',
        },
      ],
      highlights: [
        {
          title: 'Batch Scoring Summary',
          items: [
            `Source Roster: ${toPdfSafeText(sourceName)} | Active Filter: ${batchCtx?.tierFilter ?? 'All'} (${filtered.length} rows shown).`,
            predictions.length > 0
              ? `Highest-Risk Student: ${toPdfSafeText(predictions[0].student_id)} at ${(predictions[0].fail_probability * 100).toFixed(1)}% Fail Probability (${predictions[0].risk_tier} Risk).`
              : 'No batch rows currently loaded.',
          ],
        },
      ],
      vectorChart: {
        title: 'Batch Risk Composition Breakdown',
        subtitle: `Proportional risk distribution across ${predictions.length} scored roster records`,
        segments: [
          { label: 'High Risk', count: highCount, colorHex: '#dc2626' },
          { label: 'Medium Risk', count: medCount, colorHex: '#d97706' },
          { label: 'Low Risk', count: lowCount, colorHex: '#0d9488' },
        ],
      },
      table: {
        title: `Ranked Batch Predictions (Top ${Math.min(18, filtered.length)} Highest Risk)`,
        headers: ['Rank', 'Student ID', 'Fail Prob', 'Tier', 'Attend.', 'CGPA', 'Primary Action'],
        rows: filtered.slice(0, 18).map((r, idx) => [
          `#${idx + 1}`,
          toPdfSafeText(r.student_id),
          `${(r.fail_probability * 100).toFixed(1)}%`,
          r.risk_tier,
          `${r.attendance_pct.toFixed(1)}%`,
          r.prev_cgpa.toFixed(2),
          toPdfSafeText(r.primary_recommendation).slice(0, 58),
        ]),
      },
      notes: [
        'Missing numeric/categorical fields in uploaded class CSVs are imputed using training-fold medians and mode.',
      ],
    };
  }

  // What-if Simulator
  const wCtx = context.whatif;
  const base = wCtx?.baseStudent;
  const baselineProb = wCtx ? wCtx.baselinePred.failProbability * 100 : 0;
  const simProb = wCtx ? wCtx.simulatedPred.failProbability * 100 : 0;
  const deltaPp = simProb - baselineProb;

  return {
    section: 'whatif',
    title: 'What-if Attendance Simulation Report',
    subtitle: 'Hypothetical attendance sensitivity analysis with all other features held constant',
    filename,
    generatedAt,
    datasetName: safeDataset,
    modelLabel,
    metrics: [
      {
        label: 'Baseline Fail Risk',
        value: `${baselineProb.toFixed(1)}%`,
        note: base ? `At ${base.attendance_pct.toFixed(0)}% attendance` : 'Baseline profile',
        accent: 'navy',
      },
      {
        label: 'Simulated Attendance',
        value: wCtx ? `${wCtx.simulatedAttendance.toFixed(0)}%` : '-',
        note: '75% advising benchmark',
        accent: 'teal',
      },
      {
        label: 'Simulated Fail Risk',
        value: `${simProb.toFixed(1)}%`,
        note: wCtx ? `${wCtx.simulatedPred.predictedResult} | ${wCtx.simulatedPred.riskTier} Risk` : '-',
        accent:
          wCtx?.simulatedPred.riskTier === 'High'
            ? 'red'
            : wCtx?.simulatedPred.riskTier === 'Medium'
            ? 'amber'
            : 'teal',
      },
      {
        label: 'Risk Shift vs Baseline',
        value: `${deltaPp > 0 ? '+' : ''}${deltaPp.toFixed(1)} pp`,
        note: 'Hypothetical percentage-point shift',
        accent: deltaPp <= 0 ? 'teal' : 'red',
      },
    ],
    highlights: [
      {
        title: `Held-Constant Student Profile (${toPdfSafeText(wCtx?.studentLabel ?? 'Student Draft')})`,
        items: base
          ? [
              `Quiz Avg: ${base.quiz_avg.toFixed(1)}/10 | Assignment Avg: ${base.assignment_avg.toFixed(1)}/10 | Midterm: ${base.mid_marks.toFixed(1)}/30 | Lab: ${base.lab_marks.toFixed(1)}/20`,
              `Study Hours/Week: ${base.study_hours_week.toFixed(1)} | Previous CGPA: ${base.prev_cgpa.toFixed(2)} | Part-Time Job: ${base.part_time_job}`,
            ]
          : ['Default draft profile active.'],
      },
    ],
    notes: [
      'Hypothetical Model Prediction: This counterfactual curve varies only attendance while holding all other academic indicators constant. It reflects learned statistical associations, not a guaranteed causal outcome.',
    ],
  };
}
