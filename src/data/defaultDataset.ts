// Deterministic generator producing the default "Track 4 - student_performance.csv" dataset
// aligned with the exact academic input scales:
// attendance_pct: 30-100, quiz_avg: 0-10, assignment_avg: 0-10, mid_marks: 0-30,
// lab_marks: 0-20, study_hours_week: 0-40, prev_cgpa: 1-4, part_time_job: Yes/No.

function createRng(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function normal(rng: () => number, mean: number, std: number): number {
  const u1 = Math.max(1e-7, rng());
  const u2 = rng();
  const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  return mean + z * std;
}

function clamp(val: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, val));
}

export function generateDefaultStudentCsv(): string {
  const rng = createRng(202604);
  const header =
    'student_id,attendance_pct,quiz_avg,assignment_avg,mid_marks,lab_marks,study_hours_week,prev_cgpa,part_time_job,final_result';

  const baseCount = 410;
  const dataLines: string[] = [];

  for (let i = 1; i <= baseCount; i++) {
    const id = `STU-${String(1000 + i)}`;
    const partTime = rng() < 0.34 ? 'Yes' : 'No';

    const baseEngagement = normal(rng, 0, 1);
    const jobPenalty = partTime === 'Yes' ? -0.35 : 0.1;

    const attendance = clamp(
      Math.round((78 + baseEngagement * 12 + jobPenalty * 6 + normal(rng, 0, 5)) * 10) / 10,
      30,
      100
    );
    const quizAvg = clamp(
      Math.round((6.9 + baseEngagement * 1.4 + normal(rng, 0, 0.7)) * 10) / 10,
      0,
      10
    );
    const assignmentAvg = clamp(
      Math.round((7.3 + baseEngagement * 1.3 + normal(rng, 0, 0.65)) * 10) / 10,
      0,
      10
    );
    const midMarks = clamp(
      Math.round((20.2 + baseEngagement * 4.4 + normal(rng, 0, 2.0)) * 10) / 10,
      0,
      30
    );
    const labMarks = clamp(
      Math.round((14.4 + baseEngagement * 2.6 + normal(rng, 0, 1.3)) * 10) / 10,
      0,
      20
    );
    const studyHours = clamp(
      Math.round((13.5 + baseEngagement * 4.8 + jobPenalty * 2.5 + normal(rng, 0, 2.2)) * 10) / 10,
      0,
      40
    );
    const prevCgpa = clamp(
      Math.round((2.82 + baseEngagement * 0.52 + normal(rng, 0, 0.25)) * 100) / 100,
      1.0,
      4.0
    );

    // Calibrated failure logit for ~27-30% Fail class imbalance on authentic scales
    const failLogit =
      14.2 -
      0.068 * attendance -
      0.16 * midMarks -
      0.24 * quizAvg -
      0.20 * assignmentAvg -
      0.14 * labMarks -
      0.075 * studyHours -
      1.05 * prevCgpa +
      (partTime === 'Yes' ? 0.55 : -0.15) +
      normal(rng, 0, 0.55);

    const failProb = 1 / (1 + Math.exp(-failLogit));
    let finalResult: string = rng() < failProb ? 'Fail' : 'Pass';

    const attStr = rng() < 0.024 ? '' : attendance.toFixed(1);
    const quizStr = rng() < 0.02 ? '' : quizAvg.toFixed(1);
    const assignStr = rng() < 0.02 ? '' : assignmentAvg.toFixed(1);
    const midStr = rng() < 0.022 ? '' : midMarks.toFixed(1);
    const labStr = rng() < 0.018 ? '' : labMarks.toFixed(1);
    const hoursStr = rng() < 0.022 ? '' : studyHours.toFixed(1);
    const cgpaStr = rng() < 0.02 ? '' : prevCgpa.toFixed(2);
    const jobStr = rng() < 0.022 ? '' : partTime;

    if (i > 402) {
      finalResult = '';
    }

    const row = [
      id,
      attStr,
      quizStr,
      assignStr,
      midStr,
      labStr,
      hoursStr,
      cgpaStr,
      jobStr,
      finalResult,
    ].join(',');

    dataLines.push(row);
  }

  // Inject 10 exact duplicate rows
  const duplicateIndices = [4, 19, 37, 64, 102, 148, 195, 241, 289, 333];
  for (const idx of duplicateIndices) {
    dataLines.push(dataLines[idx]);
  }

  return [header, ...dataLines].join('\n');
}

export const DEFAULT_DATASET_CSV = generateDefaultStudentCsv();
