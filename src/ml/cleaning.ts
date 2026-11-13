import Papa from 'papaparse';
import {
  CleanedStudentRecord,
  CsvParseResult,
  DataCleaningSummary,
  FinalResult,
  MissingColumnCount,
  NumericFeatureKey,
  PartTimeJob,
  RangeViolationItem,
  RawStudentRow,
  StudentFeatureInput,
} from '../types/student';

export const REQUIRED_FEATURE_COLUMNS = [
  'student_id',
  'attendance_pct',
  'quiz_avg',
  'assignment_avg',
  'mid_marks',
  'lab_marks',
  'study_hours_week',
  'prev_cgpa',
  'part_time_job',
] as const;

export const ALL_DATASET_COLUMNS = [...REQUIRED_FEATURE_COLUMNS, 'final_result'] as const;

export const NUMERIC_FEATURE_KEYS: NumericFeatureKey[] = [
  'attendance_pct',
  'quiz_avg',
  'assignment_avg',
  'mid_marks',
  'lab_marks',
  'study_hours_week',
  'prev_cgpa',
];

export const FEATURE_RANGES: Record<
  NumericFeatureKey,
  { min: number; max: number; step: number; label: string; unitText: string }
> = {
  attendance_pct: { min: 30, max: 100, step: 0.5, label: 'Attendance (%)', unitText: '30–100%' },
  quiz_avg: { min: 0, max: 10, step: 0.1, label: 'Quiz Average', unitText: '0–10' },
  assignment_avg: {
    min: 0,
    max: 10,
    step: 0.1,
    label: 'Assignment Average',
    unitText: '0–10',
  },
  mid_marks: { min: 0, max: 30, step: 0.5, label: 'Midterm Marks', unitText: '0–30' },
  lab_marks: { min: 0, max: 20, step: 0.5, label: 'Lab Marks', unitText: '0–20' },
  study_hours_week: {
    min: 0,
    max: 40,
    step: 0.5,
    label: 'Study Hours / Week',
    unitText: '0–40 hrs',
  },
  prev_cgpa: { min: 1, max: 4, step: 0.05, label: 'Previous CGPA', unitText: '1.00–4.00' },
};

export function validateStudentFeatureInput(input: StudentFeatureInput): string[] {
  const errors: string[] = [];
  for (const key of NUMERIC_FEATURE_KEYS) {
    const val = input[key];
    const spec = FEATURE_RANGES[key];
    if (typeof val !== 'number' || !Number.isFinite(val)) {
      errors.push(`${spec.label} must be a valid number (${spec.unitText}).`);
    } else if (val < spec.min || val > spec.max) {
      errors.push(
        `${spec.label} must be between ${spec.min} and ${spec.max} (entered ${val}).`
      );
    }
  }
  if (input.part_time_job !== 'Yes' && input.part_time_job !== 'No') {
    errors.push('Part-Time Job must be Yes or No.');
  }
  return errors;
}

function parseNumericCell(
  value: unknown,
  key: NumericFeatureKey
): { value: number | null; invalidFormat: boolean; outOfRange: boolean } {
  if (value === undefined || value === null) {
    return { value: null, invalidFormat: false, outOfRange: false };
  }
  const trimmed = String(value).trim();
  if (
    trimmed === '' ||
    trimmed.toLowerCase() === 'na' ||
    trimmed.toLowerCase() === 'nan' ||
    trimmed.toLowerCase() === 'null' ||
    trimmed.toLowerCase() === 'none'
  ) {
    return { value: null, invalidFormat: false, outOfRange: false };
  }
  const parsed = Number(trimmed);
  if (!Number.isFinite(parsed)) {
    return { value: null, invalidFormat: true, outOfRange: false };
  }
  const { min, max } = FEATURE_RANGES[key];
  if (parsed < min || parsed > max) {
    return { value: null, invalidFormat: false, outOfRange: true };
  }
  return { value: parsed, invalidFormat: false, outOfRange: false };
}

function parsePartTimeJob(value: unknown): { value: PartTimeJob | null; invalid: boolean } {
  if (value === undefined || value === null) return { value: null, invalid: false };
  const trimmed = String(value).trim().toLowerCase();
  if (trimmed === '' || trimmed === 'na' || trimmed === 'nan' || trimmed === 'null') {
    return { value: null, invalid: false };
  }
  if (trimmed === 'yes' || trimmed === 'y' || trimmed === '1' || trimmed === 'true') {
    return { value: 'Yes', invalid: false };
  }
  if (trimmed === 'no' || trimmed === 'n' || trimmed === '0' || trimmed === 'false') {
    return { value: 'No', invalid: false };
  }
  return { value: null, invalid: true };
}

function parseFinalResult(value: unknown): { value: FinalResult | null; invalid: boolean } {
  if (value === undefined || value === null) return { value: null, invalid: false };
  const trimmed = String(value).trim().toLowerCase();
  if (trimmed === '' || trimmed === 'na' || trimmed === 'nan' || trimmed === 'null') {
    return { value: null, invalid: false };
  }
  if (trimmed === 'pass' || trimmed === 'p' || trimmed === '0') {
    return { value: 'Pass', invalid: false };
  }
  if (trimmed === 'fail' || trimmed === 'f' || trimmed === '1') {
    return { value: 'Fail', invalid: false };
  }
  return { value: null, invalid: true };
}

export function parseAndValidateCsv(csvText: string, requireTarget: boolean): CsvParseResult {
  const trimmed = csvText.trim();
  if (!trimmed) {
    return {
      ok: false,
      rows: [],
      error: 'The uploaded CSV file is empty.',
      warnings: [],
      outOfRangeCount: 0,
      rangeViolations: [],
    };
  }

  const parsed = Papa.parse<Record<string, unknown>>(trimmed, {
    header: true,
    skipEmptyLines: 'greedy',
    transformHeader: (h) => h.trim().toLowerCase(),
  });

  if (parsed.errors.length > 0 && parsed.data.length === 0) {
    return {
      ok: false,
      rows: [],
      error: `Could not parse CSV file: ${parsed.errors[0].message}`,
      warnings: [],
      outOfRangeCount: 0,
      rangeViolations: [],
    };
  }

  const fields = parsed.meta.fields ?? [];
  const expected = requireTarget ? ALL_DATASET_COLUMNS : REQUIRED_FEATURE_COLUMNS;
  const missingCols = expected.filter((col) => !fields.includes(col));

  if (missingCols.length > 0) {
    return {
      ok: false,
      rows: [],
      error: `Missing required column(s): ${missingCols.join(', ')}. Expected: ${expected.join(', ')}.`,
      warnings: [],
      outOfRangeCount: 0,
      rangeViolations: [],
    };
  }

  if (parsed.data.length === 0) {
    return {
      ok: false,
      rows: [],
      error: 'CSV contains headers but zero student rows.',
      warnings: [],
      outOfRangeCount: 0,
      rangeViolations: [],
    };
  }

  const warnings: string[] = [];
  let invalidNumericCount = 0;
  let invalidCategoricalCount = 0;
  const rangeCounts: Record<NumericFeatureKey, number> = {
    attendance_pct: 0,
    quiz_avg: 0,
    assignment_avg: 0,
    mid_marks: 0,
    lab_marks: 0,
    study_hours_week: 0,
    prev_cgpa: 0,
  };

  const rows: RawStudentRow[] = parsed.data.map((record, idx) => {
    const rawId = record['student_id'] !== undefined ? String(record['student_id']).trim() : '';
    const studentId = rawId || `ROW-${idx + 1}`;

    const att = parseNumericCell(record['attendance_pct'], 'attendance_pct');
    const quiz = parseNumericCell(record['quiz_avg'], 'quiz_avg');
    const assign = parseNumericCell(record['assignment_avg'], 'assignment_avg');
    const mid = parseNumericCell(record['mid_marks'], 'mid_marks');
    const lab = parseNumericCell(record['lab_marks'], 'lab_marks');
    const hours = parseNumericCell(record['study_hours_week'], 'study_hours_week');
    const cgpa = parseNumericCell(record['prev_cgpa'], 'prev_cgpa');
    const job = parsePartTimeJob(record['part_time_job']);
    const result = parseFinalResult(record['final_result']);

    if (
      att.invalidFormat ||
      quiz.invalidFormat ||
      assign.invalidFormat ||
      mid.invalidFormat ||
      lab.invalidFormat ||
      hours.invalidFormat ||
      cgpa.invalidFormat
    ) {
      invalidNumericCount++;
    }

    if (att.outOfRange) rangeCounts.attendance_pct++;
    if (quiz.outOfRange) rangeCounts.quiz_avg++;
    if (assign.outOfRange) rangeCounts.assignment_avg++;
    if (mid.outOfRange) rangeCounts.mid_marks++;
    if (lab.outOfRange) rangeCounts.lab_marks++;
    if (hours.outOfRange) rangeCounts.study_hours_week++;
    if (cgpa.outOfRange) rangeCounts.prev_cgpa++;

    if (job.invalid || (requireTarget && result.invalid)) {
      invalidCategoricalCount++;
    }

    return {
      student_id: studentId,
      attendance_pct: att.value,
      quiz_avg: quiz.value,
      assignment_avg: assign.value,
      mid_marks: mid.value,
      lab_marks: lab.value,
      study_hours_week: hours.value,
      prev_cgpa: cgpa.value,
      part_time_job: job.value,
      final_result: result.value,
    };
  });

  const rangeViolations: RangeViolationItem[] = NUMERIC_FEATURE_KEYS.filter(
    (k) => rangeCounts[k] > 0
  ).map((k) => ({
    column: k,
    count: rangeCounts[k],
    expectedRange: `${FEATURE_RANGES[k].min}–${FEATURE_RANGES[k].max}`,
  }));

  const outOfRangeCount = rangeViolations.reduce((acc, item) => acc + item.count, 0);

  if (outOfRangeCount > 0) {
    const detailText = rangeViolations
      .map((v) => `${v.column}: ${v.count} outside ${v.expectedRange}`)
      .join('; ');
    warnings.push(
      `Flagged ${outOfRangeCount} out-of-range value(s) (${detailText}). Invalid values were excluded and imputed via pipeline medians.`
    );
  }

  if (invalidNumericCount > 0) {
    warnings.push(
      `${invalidNumericCount} row(s) had non-numeric feature values that were flagged as missing and imputed.`
    );
  }
  if (invalidCategoricalCount > 0) {
    warnings.push(
      `${invalidCategoricalCount} row(s) had unrecognized categorical labels flagged as missing.`
    );
  }

  return { ok: true, rows, warnings, outOfRangeCount, rangeViolations };
}

export function computeMedian(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

export function computeMode(values: PartTimeJob[]): PartTimeJob {
  let yesCount = 0;
  let noCount = 0;
  for (const v of values) {
    if (v === 'Yes') yesCount++;
    else if (v === 'No') noCount++;
  }
  return yesCount > noCount ? 'Yes' : 'No';
}

function rowSignature(row: RawStudentRow): string {
  return JSON.stringify([
    row.student_id,
    row.attendance_pct,
    row.quiz_avg,
    row.assignment_avg,
    row.mid_marks,
    row.lab_marks,
    row.study_hours_week,
    row.prev_cgpa,
    row.part_time_job,
    row.final_result,
  ]);
}

export interface CleanedDatasetOutput {
  summary: DataCleaningSummary;
  trainingRows: CleanedStudentRecord[];
  excludedRows: RawStudentRow[];
}

export function cleanStudentDataset(
  rawRows: RawStudentRow[],
  datasetName: string,
  outOfRangeCount = 0,
  rangeViolations: RangeViolationItem[] = []
): CleanedDatasetOutput {
  const originalRows = rawRows.length;

  // 1. Remove exact duplicate rows
  const seen = new Set<string>();
  const deduplicated: RawStudentRow[] = [];
  for (const row of rawRows) {
    const sig = rowSignature(row);
    if (!seen.has(sig)) {
      seen.add(sig);
      deduplicated.push(row);
    }
  }
  const duplicateRowsRemoved = originalRows - deduplicated.length;

  // 2. Count missing values across deduplicated rows
  const missingByColumn: MissingColumnCount[] = ALL_DATASET_COLUMNS.map((col) => {
    let missingCount = 0;
    for (const r of deduplicated) {
      if (col === 'student_id') {
        if (!r.student_id) missingCount++;
      } else if (r[col] === null || r[col] === undefined) {
        missingCount++;
      }
    }
    return { column: col, missingCount };
  });

  const totalMissingCells = missingByColumn.reduce((acc, item) => acc + item.missingCount, 0);

  // 3. Exclude records missing final_result from model training
  const withTarget = deduplicated.filter((r) => r.final_result !== null);
  const excludedRows = deduplicated.filter((r) => r.final_result === null);

  // 4. Compute cohort-level medians & mode for dataset audit & exploratory display
  const numericMedians = {} as Record<NumericFeatureKey, number>;
  for (const key of NUMERIC_FEATURE_KEYS) {
    const vals = withTarget
      .map((r) => r[key])
      .filter((v): v is number => v !== null && Number.isFinite(v));
    numericMedians[key] = Number(computeMedian(vals).toFixed(2));
  }

  const jobVals = withTarget
    .map((r) => r.part_time_job)
    .filter((v): v is PartTimeJob => v !== null);
  const categoricalMode = computeMode(jobVals);

  const trainingRows: CleanedStudentRecord[] = withTarget.map((r) => {
    const imputed: string[] = [];
    for (const key of NUMERIC_FEATURE_KEYS) {
      if (r[key] === null) imputed.push(key);
    }
    if (r.part_time_job === null) imputed.push('part_time_job');

    return {
      student_id: r.student_id,
      attendance_pct: r.attendance_pct ?? numericMedians.attendance_pct,
      quiz_avg: r.quiz_avg ?? numericMedians.quiz_avg,
      assignment_avg: r.assignment_avg ?? numericMedians.assignment_avg,
      mid_marks: r.mid_marks ?? numericMedians.mid_marks,
      lab_marks: r.lab_marks ?? numericMedians.lab_marks,
      study_hours_week: r.study_hours_week ?? numericMedians.study_hours_week,
      prev_cgpa: r.prev_cgpa ?? numericMedians.prev_cgpa,
      part_time_job: r.part_time_job ?? categoricalMode,
      final_result: r.final_result,
      imputed_fields: imputed,
    };
  });

  const passCount = trainingRows.filter((r) => r.final_result === 'Pass').length;
  const failCount = trainingRows.filter((r) => r.final_result === 'Fail').length;

  return {
    summary: {
      datasetName,
      originalRows,
      duplicateRowsRemoved,
      deduplicatedRows: deduplicated.length,
      totalMissingCells,
      missingByColumn,
      missingFinalResultExcluded: excludedRows.length,
      cleanedTrainingRows: trainingRows.length,
      passCount,
      failCount,
      outOfRangeCount,
      rangeViolations,
      numericMedians,
      categoricalMode,
    },
    trainingRows,
    excludedRows,
  };
}
