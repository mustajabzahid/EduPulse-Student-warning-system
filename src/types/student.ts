export type FinalResult = 'Pass' | 'Fail';
export type PartTimeJob = 'Yes' | 'No';
export type RiskTier = 'Low' | 'Medium' | 'High';
export type ModelType = 'random_forest' | 'logistic_regression';

export interface RawStudentRow {
  student_id: string;
  attendance_pct: number | null;
  quiz_avg: number | null;
  assignment_avg: number | null;
  mid_marks: number | null;
  lab_marks: number | null;
  study_hours_week: number | null;
  prev_cgpa: number | null;
  part_time_job: PartTimeJob | null;
  final_result: FinalResult | null;
}

export interface StudentFeatureInput {
  attendance_pct: number;
  quiz_avg: number;
  assignment_avg: number;
  mid_marks: number;
  lab_marks: number;
  study_hours_week: number;
  prev_cgpa: number;
  part_time_job: PartTimeJob;
}

export interface CleanedStudentRecord extends StudentFeatureInput {
  student_id: string;
  final_result: FinalResult | null;
  imputed_fields: string[];
}

export interface MissingColumnCount {
  column: string;
  missingCount: number;
}

export interface RangeViolationItem {
  column: string;
  count: number;
  expectedRange: string;
}

export interface DataCleaningSummary {
  datasetName: string;
  originalRows: number;
  duplicateRowsRemoved: number;
  deduplicatedRows: number;
  totalMissingCells: number;
  missingByColumn: MissingColumnCount[];
  missingFinalResultExcluded: number;
  cleanedTrainingRows: number;
  passCount: number;
  failCount: number;
  outOfRangeCount: number;
  rangeViolations: RangeViolationItem[];
  numericMedians: Record<NumericFeatureKey, number>;
  categoricalMode: PartTimeJob;
}

export type NumericFeatureKey =
  | 'attendance_pct'
  | 'quiz_avg'
  | 'assignment_avg'
  | 'mid_marks'
  | 'lab_marks'
  | 'study_hours_week'
  | 'prev_cgpa';

export interface ConfusionMatrix {
  tp: number; // Actual Fail, Predicted Fail
  fp: number; // Actual Pass, Predicted Fail
  fn: number; // Actual Fail, Predicted Pass
  tn: number; // Actual Pass, Predicted Pass
}

export interface ModelEvaluationMetrics {
  modelType: ModelType;
  modelName: string;
  trainSize: number;
  testSize: number;
  accuracy: number;
  failPrecision: number;
  failRecall: number;
  failF1: number;
  passPrecision: number;
  passRecall: number;
  rocAuc: number;
  confusionMatrix: ConfusionMatrix;
}

export interface FeatureImportanceItem {
  feature: string;
  label: string;
  importance: number; // 0-1 normalized importance for RF or |coef| normalized for LR
  direction: 'increases_risk' | 'decreases_risk' | 'non_linear';
  coefficient?: number;
  plainExplanation: string;
}

export interface FittedPreprocessor {
  numericMedians: Record<NumericFeatureKey, number>;
  numericMeans: Record<NumericFeatureKey, number>;
  numericStds: Record<NumericFeatureKey, number>;
  partTimeMode: PartTimeJob;
  featureNames: string[];
}

export interface LogisticRegressionModel {
  weights: number[];
  bias: number;
  featureNames: string[];
}

export interface DecisionTreeNode {
  isLeaf: boolean;
  failProbability: number;
  featureIndex?: number;
  threshold?: number;
  left?: DecisionTreeNode;
  right?: DecisionTreeNode;
}

export interface RandomForestModel {
  trees: DecisionTreeNode[];
  featureImportances: number[];
  featureNames: string[];
}

export interface ScoredStudentRecord extends CleanedStudentRecord {
  lrFailProbability: number;
  rfFailProbability: number;
  activeFailProbability: number;
  predictedResult: FinalResult;
  riskTier: RiskTier;
  topDrivers: string[];
  recommendations: string[];
}

export interface TrainedModelBundle {
  preprocessor: FittedPreprocessor;
  logisticRegression: LogisticRegressionModel;
  randomForest: RandomForestModel;
  lrMetrics: ModelEvaluationMetrics;
  rfMetrics: ModelEvaluationMetrics;
  lrFeatureImportance: FeatureImportanceItem[];
  rfFeatureImportance: FeatureImportanceItem[];
  cohortMedians: Record<NumericFeatureKey, number>;
  scoredCohort: ScoredStudentRecord[];
}

export interface SinglePredictionResult {
  predictedResult: FinalResult;
  failProbability: number;
  passProbability: number;
  riskTier: RiskTier;
  recommendations: string[];
  riskFactors: string[];
}

export interface BatchPredictionRow extends StudentFeatureInput {
  student_id: string;
  actual_result: FinalResult | 'Unknown';
  predicted_result: FinalResult;
  fail_probability: number;
  risk_tier: RiskTier;
  primary_recommendation: string;
  imputed_fields: string[];
}

export interface CsvParseResult {
  ok: boolean;
  rows: RawStudentRow[];
  error?: string;
  warnings: string[];
  outOfRangeCount: number;
  rangeViolations: RangeViolationItem[];
}
