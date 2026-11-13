import test from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_DATASET_CSV } from '../data/defaultDataset';
import {
  cleanStudentDataset,
  parseAndValidateCsv,
  validateStudentFeatureInput,
} from './cleaning';
import {
  batchPredictionsToCsv,
  fitPreprocessor,
  generatePersonalizedAdvice,
  predictBatchStudents,
  predictSingleStudent,
  stratifiedTrainTestSplit,
  trainEduPulsePipeline,
} from './pipeline';

test('parses default CSV and executes data cleaning accurately', () => {
  const parsed = parseAndValidateCsv(DEFAULT_DATASET_CSV, true);
  assert.equal(parsed.ok, true);
  assert.equal(parsed.rows.length, 420);
  assert.equal(parsed.outOfRangeCount, 0);

  const cleaned = cleanStudentDataset(
    parsed.rows,
    'Track 4 - student_performance.csv',
    parsed.outOfRangeCount,
    parsed.rangeViolations
  );
  assert.equal(cleaned.summary.originalRows, 420);
  assert.equal(cleaned.summary.duplicateRowsRemoved, 10);
  assert.equal(cleaned.summary.deduplicatedRows, 410);
  assert.equal(cleaned.summary.missingFinalResultExcluded, 8);
  assert.equal(cleaned.summary.cleanedTrainingRows, 402);
  assert.equal(
    cleaned.summary.passCount + cleaned.summary.failCount,
    cleaned.summary.cleanedTrainingRows
  );
  assert.ok(cleaned.summary.totalMissingCells > 0);
});

test('flags out-of-range CSV values and validates individual inputs against competition scales', () => {
  const csvWithInvalidRange = [
    'student_id,attendance_pct,quiz_avg,assignment_avg,mid_marks,lab_marks,study_hours_week,prev_cgpa,part_time_job,final_result',
    'S1,120,15,12,35,25,45,4.5,Yes,Fail',
    'S2,78,7.0,7.5,20,14,12,3.0,No,Pass',
  ].join('\n');

  const parsed = parseAndValidateCsv(csvWithInvalidRange, true);
  assert.equal(parsed.ok, true);
  assert.equal(parsed.outOfRangeCount, 7);
  assert.ok(parsed.warnings.some((w) => w.includes('Flagged 7 out-of-range')));
  assert.equal(parsed.rows[0].quiz_avg, null);
  assert.equal(parsed.rows[0].mid_marks, null);
  assert.equal(parsed.rows[0].lab_marks, null);

  const badInputErrors = validateStudentFeatureInput({
    attendance_pct: 25,
    quiz_avg: 11,
    assignment_avg: 5,
    mid_marks: 32,
    lab_marks: 21,
    study_hours_week: 42,
    prev_cgpa: 0.8,
    part_time_job: 'Yes',
  });
  assert.equal(badInputErrors.length, 6);
});

test('gracefully rejects invalid or missing-column CSVs', () => {
  const emptyRes = parseAndValidateCsv('', true);
  assert.equal(emptyRes.ok, false);
  assert.ok(emptyRes.error?.includes('empty'));

  const missingColRes = parseAndValidateCsv('student_id,attendance_pct\nSTU-1,80', true);
  assert.equal(missingColRes.ok, false);
  assert.ok(missingColRes.error?.includes('Missing required column'));
});

test('stratified split preserves Fail proportion and prevents train/test overlap', () => {
  const parsed = parseAndValidateCsv(DEFAULT_DATASET_CSV, true);
  const deduplicatedWithTarget = parsed.rows.filter((r) => r.final_result !== null);

  const { train, test: testSet } = stratifiedTrainTestSplit(deduplicatedWithTarget, 0.2, 42);
  assert.equal(train.length + testSet.length, deduplicatedWithTarget.length);

  const trainFailRate =
    train.filter((r) => r.final_result === 'Fail').length / train.length;
  const testFailRate =
    testSet.filter((r) => r.final_result === 'Fail').length / testSet.length;

  assert.ok(Math.abs(trainFailRate - testFailRate) < 0.03);
});

test('preprocessor fits strictly on training rows without test leakage', () => {
  const trainRows = [
    {
      student_id: 'T1',
      attendance_pct: 60,
      quiz_avg: 5.0,
      assignment_avg: 5.0,
      mid_marks: 14.0,
      lab_marks: 10.0,
      study_hours_week: 10,
      prev_cgpa: 2.5,
      part_time_job: 'No' as const,
      final_result: 'Fail' as const,
    },
    {
      student_id: 'T2',
      attendance_pct: 80,
      quiz_avg: 8.0,
      assignment_avg: 8.0,
      mid_marks: 24.0,
      lab_marks: 16.0,
      study_hours_week: 14,
      prev_cgpa: 3.5,
      part_time_job: 'No' as const,
      final_result: 'Pass' as const,
    },
  ];

  const preprocessor = fitPreprocessor(trainRows);
  assert.equal(preprocessor.numericMedians.attendance_pct, 70);
  assert.equal(preprocessor.numericMedians.mid_marks, 19);
  assert.equal(preprocessor.partTimeMode, 'No');
  assert.equal(preprocessor.featureNames.includes('student_id'), false);
  assert.equal(preprocessor.featureNames.includes('final_result'), false);
  assert.equal(preprocessor.featureNames.length, 8);
});

test('trains models with real Fail metrics and authentic advice denominators', () => {
  const parsed = parseAndValidateCsv(DEFAULT_DATASET_CSV, true);
  const cleaned = cleanStudentDataset(parsed.rows, 'Track 4');

  const seen = new Set<string>();
  const rawWithTarget = parsed.rows.filter((r) => {
    const key = JSON.stringify(r);
    if (seen.has(key)) return false;
    seen.add(key);
    return r.final_result !== null;
  });

  const bundle = trainEduPulsePipeline(rawWithTarget, cleaned.trainingRows);

  for (const metrics of [bundle.lrMetrics, bundle.rfMetrics]) {
    const cm = metrics.confusionMatrix;
    assert.equal(cm.tp + cm.fp + cm.fn + cm.tn, metrics.testSize);
    assert.ok(metrics.accuracy > 0.7, `${metrics.modelName} accuracy should exceed 0.70`);
    assert.ok(metrics.failRecall > 0.65, `${metrics.modelName} failRecall should exceed 0.65`);
    assert.ok(metrics.failPrecision > 0.5, `${metrics.modelName} failPrecision should exceed 0.50`);
    assert.ok(metrics.rocAuc > 0.75, `${metrics.modelName} ROC-AUC should exceed 0.75`);
  }

  const highRiskStudent = {
    attendance_pct: 48,
    quiz_avg: 4.5,
    assignment_avg: 4.8,
    mid_marks: 12.0,
    lab_marks: 9.5,
    study_hours_week: 4,
    prev_cgpa: 1.9,
    part_time_job: 'Yes' as const,
  };
  const lowRiskStudent = {
    attendance_pct: 95,
    quiz_avg: 8.8,
    assignment_avg: 9.2,
    mid_marks: 26.5,
    lab_marks: 18.5,
    study_hours_week: 20,
    prev_cgpa: 3.7,
    part_time_job: 'No' as const,
  };

  const predHigh = predictSingleStudent(highRiskStudent, bundle, 'logistic_regression');
  const predLow = predictSingleStudent(lowRiskStudent, bundle, 'logistic_regression');
  assert.equal(predHigh.riskTier, 'High');
  assert.equal(predHigh.predictedResult, 'Fail');
  assert.equal(predLow.riskTier, 'Low');
  assert.equal(predLow.predictedResult, 'Pass');
  assert.ok(predHigh.failProbability > predLow.failProbability);

  const advice = generatePersonalizedAdvice(highRiskStudent, bundle.cohortMedians);
  assert.ok(advice.recommendations.some((r) => r.includes('4.5/10')));
  assert.ok(advice.recommendations.some((r) => r.includes('4.8/10')));
  assert.ok(advice.recommendations.some((r) => r.includes('12.0/30')));
  assert.ok(advice.recommendations.some((r) => r.includes('9.5/20')));
  assert.ok(advice.riskFactors.some((r) => r.includes('below 2.40')));

  // What-If attendance sensitivity (all other inputs unchanged)
  const lowAtt = predictSingleStudent(
    { ...highRiskStudent, attendance_pct: 40, mid_marks: 19.0, prev_cgpa: 2.8 },
    bundle,
    'logistic_regression'
  );
  const highAtt = predictSingleStudent(
    { ...highRiskStudent, attendance_pct: 96, mid_marks: 19.0, prev_cgpa: 2.8 },
    bundle,
    'logistic_regression'
  );
  assert.ok(lowAtt.failProbability > highAtt.failProbability);

  // Batch prediction sorting & CSV export
  const batchSample = parsed.rows.slice(0, 25);
  const batchRes = predictBatchStudents(batchSample, bundle, 'random_forest');
  assert.equal(batchRes.length, 25);
  for (let i = 1; i < batchRes.length; i++) {
    assert.ok(batchRes[i - 1].fail_probability >= batchRes[i].fail_probability);
  }
  const csvOut = batchPredictionsToCsv(batchRes);
  assert.ok(csvOut.startsWith('student_id,fail_probability_pct,risk_tier'));
});
