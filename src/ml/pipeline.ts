import {
  BatchPredictionRow,
  CleanedStudentRecord,
  ConfusionMatrix,
  DecisionTreeNode,
  FeatureImportanceItem,
  FinalResult,
  FittedPreprocessor,
  LogisticRegressionModel,
  ModelEvaluationMetrics,
  ModelType,
  NumericFeatureKey,
  PartTimeJob,
  RandomForestModel,
  RawStudentRow,
  RiskTier,
  ScoredStudentRecord,
  SinglePredictionResult,
  StudentFeatureInput,
  TrainedModelBundle,
} from '../types/student';
import { computeMedian, computeMode, NUMERIC_FEATURE_KEYS } from './cleaning';

export const MODEL_FEATURE_NAMES = [
  'attendance_pct',
  'quiz_avg',
  'assignment_avg',
  'mid_marks',
  'lab_marks',
  'study_hours_week',
  'prev_cgpa',
  'part_time_job',
] as const;

export const FEATURE_LABELS: Record<string, string> = {
  attendance_pct: 'Attendance (%)',
  quiz_avg: 'Quiz Average',
  assignment_avg: 'Assignment Average',
  mid_marks: 'Midterm Marks',
  lab_marks: 'Lab Marks',
  study_hours_week: 'Study Hours / Week',
  prev_cgpa: 'Previous CGPA',
  part_time_job: 'Part-Time Job',
};

function createSeededRng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function shuffleWithRng<T>(arr: T[], rng: () => number): T[] {
  const copy = [...arr];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

/**
 * Stratified train/test split preserving Pass/Fail class ratios without leakage.
 */
export function stratifiedTrainTestSplit<T extends { final_result: FinalResult | null }>(
  rows: T[],
  testRatio = 0.2,
  seed = 42
): { train: T[]; test: T[] } {
  const rng = createSeededRng(seed);
  const failRows = shuffleWithRng(
    rows.filter((r) => r.final_result === 'Fail'),
    rng
  );
  const passRows = shuffleWithRng(
    rows.filter((r) => r.final_result === 'Pass'),
    rng
  );

  const failTestCount = Math.max(1, Math.round(failRows.length * testRatio));
  const passTestCount = Math.max(1, Math.round(passRows.length * testRatio));

  const test = shuffleWithRng(
    [...failRows.slice(0, failTestCount), ...passRows.slice(0, passTestCount)],
    rng
  );
  const train = shuffleWithRng(
    [...failRows.slice(failTestCount), ...passRows.slice(passTestCount)],
    rng
  );

  return { train, test };
}

/**
 * Fit imputer + standard scaler strictly on the training split to prevent data leakage.
 * Excludes student_id and final_result.
 */
export function fitPreprocessor(trainRows: RawStudentRow[]): FittedPreprocessor {
  const numericMedians = {} as Record<NumericFeatureKey, number>;
  const numericMeans = {} as Record<NumericFeatureKey, number>;
  const numericStds = {} as Record<NumericFeatureKey, number>;

  for (const key of NUMERIC_FEATURE_KEYS) {
    const nonNull = trainRows
      .map((r) => r[key])
      .filter((v): v is number => v !== null && Number.isFinite(v));
    const med = computeMedian(nonNull);
    numericMedians[key] = med;

    const imputedVals = trainRows.map((r) => r[key] ?? med);
    const mean = imputedVals.reduce((a, b) => a + b, 0) / Math.max(1, imputedVals.length);
    const variance =
      imputedVals.reduce((acc, v) => acc + (v - mean) ** 2, 0) /
      Math.max(1, imputedVals.length);
    const std = Math.sqrt(variance);

    numericMeans[key] = mean;
    numericStds[key] = std > 1e-8 ? std : 1;
  }

  const jobVals = trainRows
    .map((r) => r.part_time_job)
    .filter((v): v is PartTimeJob => v !== null);
  const partTimeMode = computeMode(jobVals);

  return {
    numericMedians,
    numericMeans,
    numericStds,
    partTimeMode,
    featureNames: [...MODEL_FEATURE_NAMES],
  };
}

export function transformSingleFeatureVector(
  row: {
    attendance_pct: number | null;
    quiz_avg: number | null;
    assignment_avg: number | null;
    mid_marks: number | null;
    lab_marks: number | null;
    study_hours_week: number | null;
    prev_cgpa: number | null;
    part_time_job: PartTimeJob | null;
  },
  preprocessor: FittedPreprocessor
): number[] {
  const vec: number[] = [];
  for (const key of NUMERIC_FEATURE_KEYS) {
    const val = row[key] ?? preprocessor.numericMedians[key];
    vec.push((val - preprocessor.numericMeans[key]) / preprocessor.numericStds[key]);
  }
  const job = row.part_time_job ?? preprocessor.partTimeMode;
  vec.push(job === 'Yes' ? 1 : 0);
  return vec;
}

export function transformFeatures(
  rows: Array<{
    attendance_pct: number | null;
    quiz_avg: number | null;
    assignment_avg: number | null;
    mid_marks: number | null;
    lab_marks: number | null;
    study_hours_week: number | null;
    prev_cgpa: number | null;
    part_time_job: PartTimeJob | null;
  }>,
  preprocessor: FittedPreprocessor
): number[][] {
  return rows.map((r) => transformSingleFeatureVector(r, preprocessor));
}

function sigmoid(z: number): number {
  if (z >= 0) {
    const ez = Math.exp(-z);
    return 1 / (1 + ez);
  }
  const ez = Math.exp(z);
  return ez / (1 + ez);
}

function solveLinearSystem(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) {
        pivot = r;
      }
    }
    [M[col], M[pivot]] = [M[pivot], M[col]];
    const diag = Math.abs(M[col][col]) < 1e-10 ? 1e-10 : M[col][col];

    for (let j = col; j <= n; j++) {
      M[col][j] /= diag;
    }

    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = M[r][col];
      for (let j = col; j <= n; j++) {
        M[r][j] -= factor * M[col][j];
      }
    }
  }

  return M.map((row) => row[n]);
}

function computeBalancedSampleWeights(y: number[]): number[] {
  const n = y.length;
  const nFail = y.filter((v) => v === 1).length;
  const nPass = n - nFail;
  const wFail = n / (2 * Math.max(1, nFail));
  const wPass = n / (2 * Math.max(1, nPass));
  return y.map((v) => (v === 1 ? wFail : wPass));
}

/**
 * L2-regularized Logistic Regression trained via Newton-Raphson (IRLS)
 * with class_weight='balanced' sample weights. Positive class = 1 (Fail).
 */
export function trainLogisticRegression(
  X: number[][],
  y: number[],
  featureNames: string[],
  l2Lambda = 1.0
): LogisticRegressionModel {
  const n = X.length;
  const p = featureNames.length;
  const d = p + 1; // +1 for intercept bias
  const sampleWeights = computeBalancedSampleWeights(y);

  const beta = new Array(d).fill(0);

  for (let iter = 0; iter < 30; iter++) {
    const grad = new Array(d).fill(0);
    const H: number[][] = Array.from({ length: d }, () => new Array(d).fill(0));

    for (let i = 0; i < n; i++) {
      const xi = [1, ...X[i]];
      let z = 0;
      for (let j = 0; j < d; j++) z += beta[j] * xi[j];
      const prob = sigmoid(z);
      const sw = sampleWeights[i];
      const diff = sw * (prob - y[i]);
      const wDiag = sw * Math.max(1e-6, prob * (1 - prob));

      for (let j = 0; j < d; j++) {
        grad[j] += diff * xi[j];
        for (let k = j; k < d; k++) {
          H[j][k] += wDiag * xi[j] * xi[k];
        }
      }
    }

    // Symmetrize Hessian & apply L2 regularization (do not penalize intercept j=0)
    for (let j = 0; j < d; j++) {
      if (j > 0) {
        grad[j] += l2Lambda * beta[j];
        H[j][j] += l2Lambda + 1e-5;
      } else {
        H[j][j] += 1e-5;
      }
      for (let k = 0; k < j; k++) {
        H[j][k] = H[k][j];
      }
    }

    const delta = solveLinearSystem(H, grad);
    let maxStep = 0;
    for (let j = 0; j < d; j++) {
      beta[j] -= delta[j];
      maxStep = Math.max(maxStep, Math.abs(delta[j]));
    }
    if (maxStep < 1e-6) break;
  }

  return {
    bias: beta[0],
    weights: beta.slice(1),
    featureNames,
  };
}

export function predictLogisticRegressionProba(
  X: number[][],
  model: LogisticRegressionModel
): number[] {
  return X.map((row) => {
    let z = model.bias;
    for (let j = 0; j < model.weights.length; j++) {
      z += model.weights[j] * row[j];
    }
    return sigmoid(z);
  });
}

function giniImpurity(sumWPos: number, sumWNeg: number): number {
  const total = sumWPos + sumWNeg;
  if (total <= 0) return 0;
  const p = sumWPos / total;
  return 2 * p * (1 - p);
}

function buildDecisionTree(
  X: number[][],
  y: number[],
  weights: number[],
  indices: number[],
  depth: number,
  maxDepth: number,
  minSamplesSplit: number,
  minSamplesLeaf: number,
  maxFeatures: number,
  rng: () => number,
  impurityAccumulator: number[]
): DecisionTreeNode {
  let sumPos = 0;
  let sumNeg = 0;
  for (const idx of indices) {
    if (y[idx] === 1) sumPos += weights[idx];
    else sumNeg += weights[idx];
  }

  const totalW = sumPos + sumNeg;
  const failProb = totalW > 0 ? sumPos / totalW : 0.5;
  const parentGini = giniImpurity(sumPos, sumNeg);

  if (
    depth >= maxDepth ||
    indices.length < minSamplesSplit ||
    sumPos <= 1e-9 ||
    sumNeg <= 1e-9
  ) {
    return { isLeaf: true, failProbability: failProb };
  }

  const numFeatures = X[0].length;
  const allFeatures = Array.from({ length: numFeatures }, (_, i) => i);
  const candidateFeatures = shuffleWithRng(allFeatures, rng).slice(0, maxFeatures);

  let bestGain = 1e-9;
  let bestFeature = -1;
  let bestThreshold = 0;
  let bestLeftIndices: number[] = [];
  let bestRightIndices: number[] = [];

  for (const f of candidateFeatures) {
    // Sort indices by feature value
    const sorted = [...indices].sort((a, b) => X[a][f] - X[b][f]);
    let leftPos = 0;
    let leftNeg = 0;

    for (let s = 0; s < sorted.length - 1; s++) {
      const idx = sorted[s];
      if (y[idx] === 1) leftPos += weights[idx];
      else leftNeg += weights[idx];

      const nextIdx = sorted[s + 1];
      if (X[idx][f] === X[nextIdx][f]) continue;

      const leftCount = s + 1;
      const rightCount = sorted.length - leftCount;
      if (leftCount < minSamplesLeaf || rightCount < minSamplesLeaf) continue;

      const rightPos = sumPos - leftPos;
      const rightNeg = sumNeg - leftNeg;
      const leftW = leftPos + leftNeg;
      const rightW = rightPos + rightNeg;

      const weightedChildrenGini =
        (leftW * giniImpurity(leftPos, leftNeg) + rightW * giniImpurity(rightPos, rightNeg)) /
        totalW;
      const gain = parentGini - weightedChildrenGini;

      if (gain > bestGain) {
        bestGain = gain;
        bestFeature = f;
        bestThreshold = (X[idx][f] + X[nextIdx][f]) / 2;
        bestLeftIndices = sorted.slice(0, s + 1);
        bestRightIndices = sorted.slice(s + 1);
      }
    }
  }

  if (bestFeature === -1) {
    return { isLeaf: true, failProbability: failProb };
  }

  impurityAccumulator[bestFeature] += totalW * bestGain;

  return {
    isLeaf: false,
    failProbability: failProb,
    featureIndex: bestFeature,
    threshold: bestThreshold,
    left: buildDecisionTree(
      X,
      y,
      weights,
      bestLeftIndices,
      depth + 1,
      maxDepth,
      minSamplesSplit,
      minSamplesLeaf,
      maxFeatures,
      rng,
      impurityAccumulator
    ),
    right: buildDecisionTree(
      X,
      y,
      weights,
      bestRightIndices,
      depth + 1,
      maxDepth,
      minSamplesSplit,
      minSamplesLeaf,
      maxFeatures,
      rng,
      impurityAccumulator
    ),
  };
}

function predictSingleTree(row: number[], node: DecisionTreeNode): number {
  let curr = node;
  while (!curr.isLeaf && curr.featureIndex !== undefined && curr.threshold !== undefined) {
    if (row[curr.featureIndex] <= curr.threshold) {
      if (!curr.left) break;
      curr = curr.left;
    } else {
      if (!curr.right) break;
      curr = curr.right;
    }
  }
  return curr.failProbability;
}

/**
 * Balanced Random Forest Classifier (bootstrap + random feature subsampling + Gini impurity)
 */
export function trainRandomForest(
  X: number[][],
  y: number[],
  featureNames: string[],
  nTrees = 55,
  maxDepth = 6,
  seed = 42
): RandomForestModel {
  const n = X.length;
  const p = featureNames.length;
  const maxFeatures = Math.max(1, Math.round(Math.sqrt(p)));
  const rng = createSeededRng(seed);
  const weights = computeBalancedSampleWeights(y);
  const impurityAccumulator = new Array(p).fill(0);
  const trees: DecisionTreeNode[] = [];

  for (let t = 0; t < nTrees; t++) {
    const bootstrapIndices: number[] = [];
    for (let i = 0; i < n; i++) {
      bootstrapIndices.push(Math.floor(rng() * n));
    }
    const tree = buildDecisionTree(
      X,
      y,
      weights,
      bootstrapIndices,
      0,
      maxDepth,
      6,
      3,
      maxFeatures,
      rng,
      impurityAccumulator
    );
    trees.push(tree);
  }

  const totalImp = impurityAccumulator.reduce((a, b) => a + b, 0);
  const featureImportances =
    totalImp > 0
      ? impurityAccumulator.map((v) => v / totalImp)
      : new Array(p).fill(1 / p);

  return {
    trees,
    featureImportances,
    featureNames,
  };
}

export function predictRandomForestProba(X: number[][], model: RandomForestModel): number[] {
  return X.map((row) => {
    let sum = 0;
    for (const tree of model.trees) {
      sum += predictSingleTree(row, tree);
    }
    return sum / Math.max(1, model.trees.length);
  });
}

function computeRocAuc(yTrue: number[], yProb: number[]): number {
  const pairs = yTrue.map((y, idx) => ({ y, p: yProb[idx] })).sort((a, b) => b.p - a.p);
  const posTotal = yTrue.filter((y) => y === 1).length;
  const negTotal = yTrue.length - posTotal;
  if (posTotal === 0 || negTotal === 0) return 0.5;

  let tp = 0;
  let fp = 0;
  let prevTpr = 0;
  let prevFpr = 0;
  let auc = 0;

  for (const item of pairs) {
    if (item.y === 1) tp++;
    else fp++;
    const tpr = tp / posTotal;
    const fpr = fp / negTotal;
    auc += ((fpr - prevFpr) * (tpr + prevTpr)) / 2;
    prevTpr = tpr;
    prevFpr = fpr;
  }
  return Number(auc.toFixed(4));
}

/**
 * Evaluate predictions treating Fail (1) as the positive class.
 */
export function evaluateModel(
  yTrue: number[],
  yProb: number[],
  modelType: ModelType,
  modelName: string,
  trainSize: number,
  threshold = 0.5
): ModelEvaluationMetrics {
  let tp = 0;
  let fp = 0;
  let fn = 0;
  let tn = 0;

  for (let i = 0; i < yTrue.length; i++) {
    const actual = yTrue[i];
    const pred = yProb[i] >= threshold ? 1 : 0;
    if (actual === 1 && pred === 1) tp++;
    else if (actual === 0 && pred === 1) fp++;
    else if (actual === 1 && pred === 0) fn++;
    else tn++;
  }

  const total = Math.max(1, yTrue.length);
  const accuracy = (tp + tn) / total;
  const failPrecision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const failRecall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const failF1 =
    failPrecision + failRecall > 0
      ? (2 * failPrecision * failRecall) / (failPrecision + failRecall)
      : 0;
  const passPrecision = tn + fn > 0 ? tn / (tn + fn) : 0;
  const passRecall = tn + fp > 0 ? tn / (tn + fp) : 0;
  const rocAuc = computeRocAuc(yTrue, yProb);

  const confusionMatrix: ConfusionMatrix = { tp, fp, fn, tn };

  return {
    modelType,
    modelName,
    trainSize,
    testSize: yTrue.length,
    accuracy,
    failPrecision,
    failRecall,
    failF1,
    passPrecision,
    passRecall,
    rocAuc,
    confusionMatrix,
  };
}

export function classifyRiskTier(failProbability: number): RiskTier {
  if (failProbability >= 0.65) return 'High';
  if (failProbability >= 0.35) return 'Medium';
  return 'Low';
}

function describeFeaturePlainly(
  feature: string,
  direction: 'increases_risk' | 'decreases_risk' | 'non_linear'
): string {
  switch (feature) {
    case 'attendance_pct':
      return direction === 'increases_risk'
        ? 'Higher attendance was associated with higher risk in this dataset.'
        : 'Lower lecture and lab attendance is strongly associated with higher failure probability in historical records.';
    case 'prev_cgpa':
      return 'Prior cumulative GPA reflects long-term academic readiness and strongly separates passing and failing cohorts.';
    case 'mid_marks':
      return 'Midterm exam performance serves as an early summative checkpoint indicating concept mastery before finals.';
    case 'quiz_avg':
      return 'Consistent quiz scores track week-to-week retention and steady engagement across course modules.';
    case 'lab_marks':
      return 'Practical lab performance captures hands-on execution and coursework completion.';
    case 'assignment_avg':
      return 'Assignment averages reflect structured homework completion and ongoing feedback incorporation.';
    case 'study_hours_week':
      return 'Weekly self-study hours show how much independent reinforcement time a student dedicates outside class.';
    case 'part_time_job':
      return direction === 'increases_risk'
        ? 'Working a part-time job coincides with elevated academic risk in this cohort, potentially reflecting time-management pressure.'
        : 'Part-time employment status interacts with study schedule capacity.';
    default:
      return 'Associated with historical academic outcome patterns.';
  }
}

export function generatePersonalizedAdvice(
  input: StudentFeatureInput,
  cohortMedians: Record<NumericFeatureKey, number>
): { recommendations: string[]; riskFactors: string[] } {
  const recommendations: string[] = [];
  const riskFactors: string[] = [];

  if (input.attendance_pct < 75) {
    riskFactors.push(
      `Low attendance (${input.attendance_pct.toFixed(1)}% vs ${cohortMedians.attendance_pct.toFixed(0)}% median)`
    );
    recommendations.push(
      `Prioritize raising attendance from ${input.attendance_pct.toFixed(0)}% above the 75% benchmark through weekly check-ins and barrier review.`
    );
  } else if (input.attendance_pct < 82) {
    recommendations.push(
      `Maintain or improve attendance (${input.attendance_pct.toFixed(0)}%) toward 85%+ to strengthen continuity before final assessments.`
    );
  }

  if (input.mid_marks < 18) {
    riskFactors.push(`Below-benchmark midterm (${input.mid_marks.toFixed(1)}/30)`);
    recommendations.push(
      `Schedule targeted midterm remediation and past-paper walkthroughs with the course TA (current midterm: ${input.mid_marks.toFixed(1)}/30, target ≥ 18/30).`
    );
  }

  if (input.quiz_avg < 6) {
    riskFactors.push(`Low quiz average (${input.quiz_avg.toFixed(1)}/10)`);
    recommendations.push(
      `Introduce spaced-repetition practice quizzes before weekly evaluations to lift quiz average from ${input.quiz_avg.toFixed(1)}/10 toward 6.0+/10.`
    );
  }

  if (input.assignment_avg < 6) {
    riskFactors.push(`Low assignment average (${input.assignment_avg.toFixed(1)}/10)`);
    recommendations.push(
      `Set structured 48-hour pre-deadline draft reviews to improve assignment consistency (currently ${input.assignment_avg.toFixed(1)}/10, target ≥ 6.0/10).`
    );
  }

  if (input.lab_marks < 12) {
    riskFactors.push(`Lab performance gap (${input.lab_marks.toFixed(1)}/20)`);
    recommendations.push(
      `Pair student with peer lab tutoring or guided office hours to close practical skill gaps (current lab score: ${input.lab_marks.toFixed(1)}/20, target ≥ 12/20).`
    );
  }

  if (input.study_hours_week < 10) {
    riskFactors.push(`Limited weekly study time (${input.study_hours_week.toFixed(1)} hrs/wk)`);
    recommendations.push(
      `Build a blocked 12–15 hr/week study timetable (currently ${input.study_hours_week.toFixed(1)} hrs/week, target ≥ 10 hrs/week) focused on weak modules.`
    );
  }

  if (input.prev_cgpa < 2.4) {
    riskFactors.push(`Prior CGPA below 2.40 (${input.prev_cgpa.toFixed(2)})`);
    recommendations.push(
      `Enroll in foundational academic coaching to stabilize study methods alongside a prior CGPA of ${input.prev_cgpa.toFixed(2)}/4.00.`
    );
  }

  if (input.part_time_job === 'Yes' && (input.study_hours_week < 12 || input.attendance_pct < 80)) {
    riskFactors.push('Part-time work alongside tight study/attendance margins');
    recommendations.push(
      'Review part-time shift scheduling around core lecture hours and exam weeks to protect study bandwidth.'
    );
  }

  if (recommendations.length === 0) {
    recommendations.push(
      'Student metrics are aligned with or above passing benchmarks. Continue current attendance and assessment cadence.'
    );
  }
  if (riskFactors.length === 0) {
    riskFactors.push('No major threshold deficits detected');
  }

  return { recommendations, riskFactors };
}

/**
 * End-to-end training pipeline using raw deduplicated rows (strictly fitting imputer & scaler
 * ONLY on the stratified training split so zero test leakage occurs).
 */
export function trainEduPulsePipeline(
  rawRowsWithTarget: RawStudentRow[],
  cleanedForScoring: CleanedStudentRecord[]
): TrainedModelBundle {
  const { train, test } = stratifiedTrainTestSplit(rawRowsWithTarget, 0.2, 42);

  // Fit preprocessing strictly on training split
  const preprocessor = fitPreprocessor(train);

  const XTrain = transformFeatures(train, preprocessor);
  const yTrain = train.map((r) => (r.final_result === 'Fail' ? 1 : 0));

  const XTest = transformFeatures(test, preprocessor);
  const yTest = test.map((r) => (r.final_result === 'Fail' ? 1 : 0));

  const featureNames = [...MODEL_FEATURE_NAMES];

  const lrModel = trainLogisticRegression(XTrain, yTrain, featureNames, 1.0);
  const rfModel = trainRandomForest(XTrain, yTrain, featureNames, 55, 6, 42);

  const lrTestProba = predictLogisticRegressionProba(XTest, lrModel);
  const rfTestProba = predictRandomForestProba(XTest, rfModel);

  const lrMetrics = evaluateModel(
    yTest,
    lrTestProba,
    'logistic_regression',
    'Logistic Regression',
    train.length
  );
  const rfMetrics = evaluateModel(
    yTest,
    rfTestProba,
    'random_forest',
    'Random Forest',
    train.length
  );

  // Build explainability lists
  const absLrSum = lrModel.weights.reduce((acc, w) => acc + Math.abs(w), 0) || 1;
  const lrFeatureImportance: FeatureImportanceItem[] = featureNames
    .map((f, idx) => {
      const coef = lrModel.weights[idx];
      const direction: 'increases_risk' | 'decreases_risk' =
        coef >= 0 ? 'increases_risk' : 'decreases_risk';
      return {
        feature: f,
        label: FEATURE_LABELS[f] ?? f,
        importance: Math.abs(coef) / absLrSum,
        direction,
        coefficient: coef,
        plainExplanation: describeFeaturePlainly(f, direction),
      };
    })
    .sort((a, b) => b.importance - a.importance);

  const rfFeatureImportance: FeatureImportanceItem[] = featureNames
    .map((f, idx) => {
      const lrCoef = lrModel.weights[idx];
      const direction: 'increases_risk' | 'decreases_risk' =
        lrCoef >= 0 ? 'increases_risk' : 'decreases_risk';
      return {
        feature: f,
        label: FEATURE_LABELS[f] ?? f,
        importance: rfModel.featureImportances[idx],
        direction,
        coefficient: lrCoef,
        plainExplanation: describeFeaturePlainly(f, direction),
      };
    })
    .sort((a, b) => b.importance - a.importance);

  const cohortMedians = preprocessor.numericMedians;

  // Score all cleaned students using the trained pipeline
  const XCohort = transformFeatures(cleanedForScoring, preprocessor);
  const lrCohortProba = predictLogisticRegressionProba(XCohort, lrModel);
  const rfCohortProba = predictRandomForestProba(XCohort, rfModel);

  const scoredCohort: ScoredStudentRecord[] = cleanedForScoring.map((student, idx) => {
    const lrP = lrCohortProba[idx];
    const rfP = rfCohortProba[idx];
    const { recommendations, riskFactors } = generatePersonalizedAdvice(student, cohortMedians);
    return {
      ...student,
      lrFailProbability: lrP,
      rfFailProbability: rfP,
      activeFailProbability: rfP,
      predictedResult: rfP >= 0.5 ? 'Fail' : 'Pass',
      riskTier: classifyRiskTier(rfP),
      topDrivers: riskFactors,
      recommendations,
    };
  });

  return {
    preprocessor,
    logisticRegression: lrModel,
    randomForest: rfModel,
    lrMetrics,
    rfMetrics,
    lrFeatureImportance,
    rfFeatureImportance,
    cohortMedians,
    scoredCohort,
  };
}

export function predictSingleStudent(
  input: StudentFeatureInput,
  bundle: TrainedModelBundle,
  modelType: ModelType
): SinglePredictionResult {
  const X = [transformSingleFeatureVector(input, bundle.preprocessor)];
  const failProbability =
    modelType === 'logistic_regression'
      ? predictLogisticRegressionProba(X, bundle.logisticRegression)[0]
      : predictRandomForestProba(X, bundle.randomForest)[0];

  const predictedResult: FinalResult = failProbability >= 0.5 ? 'Fail' : 'Pass';
  const riskTier = classifyRiskTier(failProbability);
  const { recommendations, riskFactors } = generatePersonalizedAdvice(input, bundle.cohortMedians);

  return {
    predictedResult,
    failProbability,
    passProbability: 1 - failProbability,
    riskTier,
    recommendations,
    riskFactors,
  };
}

export function predictBatchStudents(
  rawRows: RawStudentRow[],
  bundle: TrainedModelBundle,
  modelType: ModelType
): BatchPredictionRow[] {
  const X = transformFeatures(rawRows, bundle.preprocessor);
  const probs =
    modelType === 'logistic_regression'
      ? predictLogisticRegressionProba(X, bundle.logisticRegression)
      : predictRandomForestProba(X, bundle.randomForest);

  const results: BatchPredictionRow[] = rawRows.map((raw, idx) => {
    const imputed_fields: string[] = [];
    for (const key of NUMERIC_FEATURE_KEYS) {
      if (raw[key] === null) imputed_fields.push(key);
    }
    if (raw.part_time_job === null) imputed_fields.push('part_time_job');

    const filled: StudentFeatureInput = {
      attendance_pct: raw.attendance_pct ?? bundle.preprocessor.numericMedians.attendance_pct,
      quiz_avg: raw.quiz_avg ?? bundle.preprocessor.numericMedians.quiz_avg,
      assignment_avg: raw.assignment_avg ?? bundle.preprocessor.numericMedians.assignment_avg,
      mid_marks: raw.mid_marks ?? bundle.preprocessor.numericMedians.mid_marks,
      lab_marks: raw.lab_marks ?? bundle.preprocessor.numericMedians.lab_marks,
      study_hours_week:
        raw.study_hours_week ?? bundle.preprocessor.numericMedians.study_hours_week,
      prev_cgpa: raw.prev_cgpa ?? bundle.preprocessor.numericMedians.prev_cgpa,
      part_time_job: raw.part_time_job ?? bundle.preprocessor.partTimeMode,
    };

    const failProbability = probs[idx];
    const predictedResult: FinalResult = failProbability >= 0.5 ? 'Fail' : 'Pass';
    const riskTier = classifyRiskTier(failProbability);
    const { recommendations } = generatePersonalizedAdvice(filled, bundle.cohortMedians);

    return {
      student_id: raw.student_id,
      ...filled,
      actual_result: raw.final_result ?? 'Unknown',
      predicted_result: predictedResult,
      fail_probability: failProbability,
      risk_tier: riskTier,
      primary_recommendation: recommendations[0],
      imputed_fields,
    };
  });

  // Sort descending by highest failure risk
  return results.sort((a, b) => b.fail_probability - a.fail_probability);
}

export function batchPredictionsToCsv(rows: BatchPredictionRow[]): string {
  const headers = [
    'student_id',
    'fail_probability_pct',
    'risk_tier',
    'predicted_result',
    'actual_result',
    'attendance_pct',
    'quiz_avg',
    'assignment_avg',
    'mid_marks',
    'lab_marks',
    'study_hours_week',
    'prev_cgpa',
    'part_time_job',
    'primary_recommendation',
  ];

  const escapeCsv = (val: string | number) => {
    const str = String(val);
    if (str.includes(',') || str.includes('"') || str.includes('\n')) {
      return `"${str.replace(/"/g, '""')}"`;
    }
    return str;
  };

  const lines = rows.map((r) =>
    [
      r.student_id,
      (r.fail_probability * 100).toFixed(1),
      r.risk_tier,
      r.predicted_result,
      r.actual_result,
      r.attendance_pct.toFixed(1),
      r.quiz_avg.toFixed(1),
      r.assignment_avg.toFixed(1),
      r.mid_marks.toFixed(1),
      r.lab_marks.toFixed(1),
      r.study_hours_week.toFixed(1),
      r.prev_cgpa.toFixed(2),
      r.part_time_job,
      r.primary_recommendation,
    ]
      .map(escapeCsv)
      .join(',')
  );

  return [headers.join(','), ...lines].join('\n');
}

export function simulateAttendanceCurve(
  baseInput: StudentFeatureInput,
  bundle: TrainedModelBundle,
  modelType: ModelType
): Array<{ attendance: number; failProbability: number; riskTier: RiskTier }> {
  const points: Array<{ attendance: number; failProbability: number; riskTier: RiskTier }> = [];
  for (let att = 30; att <= 100; att += 5) {
    const pred = predictSingleStudent(
      { ...baseInput, attendance_pct: att },
      bundle,
      modelType
    );
    points.push({
      attendance: att,
      failProbability: pred.failProbability,
      riskTier: pred.riskTier,
    });
  }
  return points;
}
