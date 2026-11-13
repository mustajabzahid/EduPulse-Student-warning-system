import { CleanedStudentRecord } from '../types/student';

export interface CgpaBandSpec {
  id: string;
  label: string;
  min: number;
  max: number;
  inclusiveMax: boolean;
}

export const CGPA_THREE_BUCKETS: CgpaBandSpec[] = [
  { id: 'bucket_1_2', label: '1.0–2.0', min: 1.0, max: 2.0, inclusiveMax: false },
  { id: 'bucket_2_3', label: '2.0–3.0', min: 2.0, max: 3.0, inclusiveMax: false },
  { id: 'bucket_3_4', label: '3.0–4.0', min: 3.0, max: 4.0, inclusiveMax: true },
];

export const CGPA_PROGRESSION_BANDS: CgpaBandSpec[] = [
  { id: 'b1', label: '1.00–1.49', min: 1.0, max: 1.5, inclusiveMax: false },
  { id: 'b2', label: '1.50–1.99', min: 1.5, max: 2.0, inclusiveMax: false },
  { id: 'b3', label: '2.00–2.39', min: 2.0, max: 2.4, inclusiveMax: false },
  { id: 'b4', label: '2.40–2.79', min: 2.4, max: 2.8, inclusiveMax: false },
  { id: 'b5', label: '2.80–3.19', min: 2.8, max: 3.2, inclusiveMax: false },
  { id: 'b6', label: '3.20–3.59', min: 3.2, max: 3.6, inclusiveMax: false },
  { id: 'b7', label: '3.60–4.00', min: 3.6, max: 4.0, inclusiveMax: true },
];

export interface CgpaTrendPoint {
  id: string;
  bandLabel: string;
  count: number;
  passRatePct: number;
  quizMean: number;
  quizPct: number;
  assignmentMean: number;
  assignmentPct: number;
  midMarksMean: number;
  midMarksPct: number;
  labMarksMean: number;
  labMarksPct: number;
  attendanceMean: number;
}

export interface CgpaTrendSummary {
  points: CgpaTrendPoint[];
  midGrowthPctPoints: number;
  quizGrowthPctPoints: number;
  lowestBandLabel: string;
  highestBandLabel: string;
}

export interface CgpaThreeBucketSummary {
  points: CgpaTrendPoint[];
  quizGrowthRaw: number;
  midGrowthRaw: number;
}

function avg(nums: number[]): number {
  if (nums.length === 0) return 0;
  return nums.reduce((acc, v) => acc + v, 0) / nums.length;
}

function aggregateBands(
  rows: CleanedStudentRecord[],
  bands: CgpaBandSpec[]
): CgpaTrendPoint[] {
  const points: CgpaTrendPoint[] = [];

  for (const band of bands) {
    const matched = rows.filter((r) => {
      if (band.inclusiveMax) {
        return r.prev_cgpa >= band.min && r.prev_cgpa <= band.max;
      }
      return r.prev_cgpa >= band.min && r.prev_cgpa < band.max;
    });

    if (matched.length === 0) continue;

    const passCount = matched.filter((r) => r.final_result === 'Pass').length;
    const quizMean = avg(matched.map((r) => r.quiz_avg));
    const assignmentMean = avg(matched.map((r) => r.assignment_avg));
    const midMarksMean = avg(matched.map((r) => r.mid_marks));
    const labMarksMean = avg(matched.map((r) => r.lab_marks));
    const attendanceMean = avg(matched.map((r) => r.attendance_pct));

    points.push({
      id: band.id,
      bandLabel: band.label,
      count: matched.length,
      passRatePct: (passCount / matched.length) * 100,
      quizMean,
      quizPct: (quizMean / 10) * 100,
      assignmentMean,
      assignmentPct: (assignmentMean / 10) * 100,
      midMarksMean,
      midMarksPct: (midMarksMean / 30) * 100,
      labMarksMean,
      labMarksPct: (labMarksMean / 20) * 100,
      attendanceMean,
    });
  }

  return points;
}

export function buildCgpaThreeBucketTrend(
  rows: CleanedStudentRecord[]
): CgpaThreeBucketSummary {
  const points = aggregateBands(rows, CGPA_THREE_BUCKETS);
  const first = points[0];
  const last = points[points.length - 1];
  return {
    points,
    quizGrowthRaw: first && last ? last.quizMean - first.quizMean : 0,
    midGrowthRaw: first && last ? last.midMarksMean - first.midMarksMean : 0,
  };
}

export function buildCgpaGrowthTrend(rows: CleanedStudentRecord[]): CgpaTrendSummary {
  const points = aggregateBands(rows, CGPA_PROGRESSION_BANDS);
  const first = points[0];
  const last = points[points.length - 1];

  return {
    points,
    midGrowthPctPoints: first && last ? last.midMarksPct - first.midMarksPct : 0,
    quizGrowthPctPoints: first && last ? last.quizPct - first.quizPct : 0,
    lowestBandLabel: first?.bandLabel ?? '-',
    highestBandLabel: last?.bandLabel ?? '-',
  };
}
