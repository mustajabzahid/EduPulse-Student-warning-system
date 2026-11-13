import test from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_DATASET_CSV } from '../data/defaultDataset';
import { cleanStudentDataset, parseAndValidateCsv } from '../ml/cleaning';
import { predictBatchStudents, predictSingleStudent, trainEduPulsePipeline } from '../ml/pipeline';
import { buildViewReportSpec, formatReportFilename, toPdfSafeText } from './reportData';
import { StudentFeatureInput } from '../types/student';

const parsed = parseAndValidateCsv(DEFAULT_DATASET_CSV, true);
assert.equal(parsed.ok, true);
const cleaned = cleanStudentDataset(
  parsed.rows,
  'Track 4 - student_performance.csv',
  parsed.outOfRangeCount,
  parsed.rangeViolations
);
const rawWithTarget = parsed.rows.filter((r) => r.final_result !== null);
const bundle = trainEduPulsePipeline(rawWithTarget, cleaned.trainingRows);

const SAMPLE_DRAFT: StudentFeatureInput = {
  attendance_pct: 62,
  quiz_avg: 5.1,
  assignment_avg: 5.5,
  mid_marks: 14.0,
  lab_marks: 11.0,
  study_hours_week: 7.5,
  prev_cgpa: 2.2,
  part_time_job: 'Yes',
};

test('toPdfSafeText normalizes non-WinAnsi Unicode symbols into ASCII', () => {
  const cleanedText = toPdfSafeText('P(Fail) ≥ 65% · Attendance × Midterm — Pass − Fail → OK');
  assert.equal(cleanedText, 'P(Fail) >= 65% | Attendance x Midterm - Pass - Fail -> OK');
});

test('formatReportFilename generates deterministic dated PDF filenames', () => {
  const fixedDate = new Date(2026, 10, 12, 9, 5); // Nov 12, 2026 09:05
  assert.equal(
    formatReportFilename('overview', fixedDate),
    'edupulse-overview-report-20261112-0905.pdf'
  );
  assert.equal(
    formatReportFilename('whatif', fixedDate),
    'edupulse-whatif-report-20261112-0905.pdf'
  );
});

test('buildViewReportSpec creates complete metrics and audit table for Overview', () => {
  const spec = buildViewReportSpec({
    activeSection: 'overview',
    activeModel: 'random_forest',
    datasetName: 'Track 4 - student_performance.csv',
    summary: cleaned.summary,
    trainingRows: cleaned.trainingRows,
    bundle,
  });

  assert.equal(spec.section, 'overview');
  assert.equal(spec.metrics.length, 4);
  assert.equal(spec.metrics[0].value, String(cleaned.summary.cleanedTrainingRows));
  assert.ok(spec.table && spec.table.rows.length > 0);
});

test('buildViewReportSpec builds Analytics report with feature benchmark table', () => {
  const spec = buildViewReportSpec({
    activeSection: 'analytics',
    activeModel: 'logistic_regression',
    datasetName: 'Track 4 - student_performance.csv',
    summary: cleaned.summary,
    trainingRows: cleaned.trainingRows,
    bundle,
  });

  assert.equal(spec.section, 'analytics');
  assert.equal(spec.metrics.length, 4);
  assert.equal(spec.table?.rows.length, 7);
});

test('buildViewReportSpec handles Prediction view before and after submission', () => {
  const pendingSpec = buildViewReportSpec({
    activeSection: 'prediction',
    activeModel: 'random_forest',
    datasetName: 'Track 4 - student_performance.csv',
    summary: cleaned.summary,
    trainingRows: cleaned.trainingRows,
    bundle,
    context: {
      prediction: {
        submittedInput: null,
        result: null,
        explainView: 'rf',
        draftInput: SAMPLE_DRAFT,
      },
    },
  });
  assert.equal(pendingSpec.metrics[0].value, 'Pending Submit');

  const pred = predictSingleStudent(SAMPLE_DRAFT, bundle, 'random_forest');
  const submittedSpec = buildViewReportSpec({
    activeSection: 'prediction',
    activeModel: 'random_forest',
    datasetName: 'Track 4 - student_performance.csv',
    summary: cleaned.summary,
    trainingRows: cleaned.trainingRows,
    bundle,
    context: {
      prediction: {
        submittedInput: SAMPLE_DRAFT,
        result: pred,
        explainView: 'lr',
        draftInput: SAMPLE_DRAFT,
      },
    },
  });
  assert.equal(submittedSpec.metrics[0].value, pred.predictedResult);
  assert.ok(submittedSpec.highlights.some((h) => h.title.includes('Advisor Recommendations')));
});

test('buildViewReportSpec builds Risk Register, Batch Analysis, and What-If specs with vector/tables', () => {
  const regSpec = buildViewReportSpec({
    activeSection: 'register',
    activeModel: 'random_forest',
    datasetName: 'Track 4 - student_performance.csv',
    summary: cleaned.summary,
    trainingRows: cleaned.trainingRows,
    bundle,
  });
  assert.ok(regSpec.vectorChart);
  assert.ok(regSpec.table && regSpec.table.rows.length > 0);

  const batchPreds = predictBatchStudents(parsed.rows.slice(0, 10), bundle, 'random_forest');
  const batchSpec = buildViewReportSpec({
    activeSection: 'batch',
    activeModel: 'random_forest',
    datasetName: 'Track 4 - student_performance.csv',
    summary: cleaned.summary,
    trainingRows: cleaned.trainingRows,
    bundle,
    context: {
      batch: {
        batchSourceName: 'Class A.csv',
        tierFilter: 'All',
        predictions: batchPreds,
        filteredPredictions: batchPreds,
      },
    },
  });
  assert.equal(batchSpec.metrics[0].value, '10');
  assert.ok(batchSpec.vectorChart);
  assert.equal(batchSpec.table?.rows.length, 10);

  const basePred = predictSingleStudent(SAMPLE_DRAFT, bundle, 'random_forest');
  const simPred = predictSingleStudent(
    { ...SAMPLE_DRAFT, attendance_pct: 92 },
    bundle,
    'random_forest'
  );
  const whatIfSpec = buildViewReportSpec({
    activeSection: 'whatif',
    activeModel: 'random_forest',
    datasetName: 'Track 4 - student_performance.csv',
    summary: cleaned.summary,
    trainingRows: cleaned.trainingRows,
    bundle,
    context: {
      whatif: {
        simulatedAttendance: 92,
        studentLabel: 'STU-999',
        baseStudent: SAMPLE_DRAFT,
        baselinePred: basePred,
        simulatedPred: simPred,
      },
    },
  });
  assert.equal(whatIfSpec.metrics[1].value, '92%');
  assert.ok(whatIfSpec.notes[0].includes('Hypothetical Model Prediction'));
});
