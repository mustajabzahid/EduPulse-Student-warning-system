import test from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_DATASET_CSV } from '../data/defaultDataset';
import { cleanStudentDataset, parseAndValidateCsv } from '../ml/cleaning';
import { buildCgpaGrowthTrend, buildCgpaThreeBucketTrend } from './analyticsTrends';

test('buildCgpaThreeBucketTrend aggregates quiz and midterm averages by 1.0-2.0, 2.0-3.0, and 3.0-4.0 buckets', () => {
  const parsed = parseAndValidateCsv(DEFAULT_DATASET_CSV, true);
  assert.equal(parsed.ok, true);
  const cleaned = cleanStudentDataset(
    parsed.rows,
    'Track 4 - student_performance.csv',
    parsed.outOfRangeCount,
    parsed.rangeViolations
  );

  const buckets = buildCgpaThreeBucketTrend(cleaned.trainingRows);
  assert.deepEqual(
    buckets.points.map((p) => p.bandLabel),
    ['1.0–2.0', '2.0–3.0', '3.0–4.0']
  );

  const totalCounted = buckets.points.reduce((acc, p) => acc + p.count, 0);
  assert.equal(totalCounted, cleaned.trainingRows.length);

  assert.ok(buckets.points[2].quizMean > buckets.points[0].quizMean);
  assert.ok(buckets.points[2].midMarksMean > buckets.points[0].midMarksMean);
  assert.ok(buckets.quizGrowthRaw > 0);
  assert.ok(buckets.midGrowthRaw > 0);
});

test('buildCgpaGrowthTrend aggregates default cohort across ordered prev_cgpa bands', () => {
  const parsed = parseAndValidateCsv(DEFAULT_DATASET_CSV, true);
  assert.equal(parsed.ok, true);
  const cleaned = cleanStudentDataset(
    parsed.rows,
    'Track 4 - student_performance.csv',
    parsed.outOfRangeCount,
    parsed.rangeViolations
  );

  const trend = buildCgpaGrowthTrend(cleaned.trainingRows);
  assert.ok(trend.points.length >= 5, 'Expected at least 5 populated CGPA progression bands');

  const totalCounted = trend.points.reduce((acc, p) => acc + p.count, 0);
  assert.equal(totalCounted, cleaned.trainingRows.length);

  const first = trend.points[0];
  const last = trend.points[trend.points.length - 1];
  assert.ok(last.midMarksMean > first.midMarksMean, 'Midterm mean should grow across higher CGPA bands');
  assert.ok(last.quizMean > first.quizMean, 'Quiz mean should grow across higher CGPA bands');
  assert.ok(trend.midGrowthPctPoints > 0);
  assert.ok(trend.quizGrowthPctPoints > 0);
});

test('buildCgpaGrowthTrend handles empty input and boundary CGPA values safely', () => {
  const emptyTrend = buildCgpaGrowthTrend([]);
  assert.equal(emptyTrend.points.length, 0);
  assert.equal(emptyTrend.midGrowthPctPoints, 0);

  const boundaryTrend = buildCgpaGrowthTrend([
    {
      student_id: 'S-1',
      attendance_pct: 50,
      quiz_avg: 4,
      assignment_avg: 5,
      mid_marks: 12,
      lab_marks: 10,
      study_hours_week: 6,
      prev_cgpa: 1.0,
      part_time_job: 'No',
      final_result: 'Fail',
      imputed_fields: [],
    },
    {
      student_id: 'S-2',
      attendance_pct: 95,
      quiz_avg: 9,
      assignment_avg: 9,
      mid_marks: 27,
      lab_marks: 18,
      study_hours_week: 20,
      prev_cgpa: 4.0,
      part_time_job: 'No',
      final_result: 'Pass',
      imputed_fields: [],
    },
  ]);

  assert.equal(boundaryTrend.points.length, 2);
  assert.equal(boundaryTrend.points[0].bandLabel, '1.00–1.49');
  assert.equal(boundaryTrend.points[1].bandLabel, '3.60–4.00');
  assert.equal(boundaryTrend.points[0].midMarksMean, 12);
  assert.equal(boundaryTrend.points[1].midMarksMean, 27);
});
