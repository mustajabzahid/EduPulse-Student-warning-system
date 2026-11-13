import { useCallback, useMemo, useRef, useState } from 'react';
import { MotionConfig } from 'motion/react';
import { AlertTriangle, CheckCircle2, X } from 'lucide-react';
import { DEFAULT_DATASET_CSV } from './data/defaultDataset';
import { cleanStudentDataset, parseAndValidateCsv } from './ml/cleaning';
import { trainEduPulsePipeline } from './ml/pipeline';
import { ModelType, StudentFeatureInput } from './types/student';
import { buildViewReportSpec, ViewReportContext } from './utils/reportData';
import { generateAndDownloadViewPdf } from './utils/pdfReport';
import { NavSection, Sidebar } from './components/Sidebar';
import { TopBar } from './components/TopBar';
import { OverviewView } from './components/views/OverviewView';
import { AnalyticsView } from './components/views/AnalyticsView';
import { PredictionView } from './components/views/PredictionView';
import { RiskRegisterView } from './components/views/RiskRegisterView';
import { BatchAnalysisView } from './components/views/BatchAnalysisView';
import { WhatIfView } from './components/views/WhatIfView';

const DEFAULT_STUDENT_DRAFT: StudentFeatureInput = {
  attendance_pct: 68,
  quiz_avg: 5.4,
  assignment_avg: 5.8,
  mid_marks: 15.5,
  lab_marks: 11.5,
  study_hours_week: 8.5,
  prev_cgpa: 2.35,
  part_time_job: 'Yes',
};

export default function App() {
  const [activeSection, setActiveSection] = useState<NavSection>('overview');
  const [activeModel, setActiveModel] = useState<ModelType>('random_forest');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const [datasetCsv, setDatasetCsv] = useState<string>(DEFAULT_DATASET_CSV);
  const [datasetName, setDatasetName] = useState<string>('Track 4 - student_performance.csv');
  const [isCustomDataset, setIsCustomDataset] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const [studentDraft, setStudentDraft] = useState<StudentFeatureInput>(DEFAULT_STUDENT_DRAFT);
  const [whatIfLabel, setWhatIfLabel] = useState<string>('Custom Student Draft');
  const [quickSearchQuery, setQuickSearchQuery] = useState<string>('');

  const mainRef = useRef<HTMLElement | null>(null);
  const reportContextRef = useRef<ViewReportContext>({});
  const [isExportingPdf, setIsExportingPdf] = useState(false);
  const [exportNotice, setExportNotice] = useState<{
    type: 'success' | 'error';
    message: string;
  } | null>(null);

  const workspace = useMemo(() => {
    const parsed = parseAndValidateCsv(datasetCsv, true);
    if (!parsed.ok) {
      // Fallback safely if corrupted
      const fallback = parseAndValidateCsv(DEFAULT_DATASET_CSV, true);
      const cleaned = cleanStudentDataset(
        fallback.rows,
        'Track 4 - student_performance.csv',
        fallback.outOfRangeCount,
        fallback.rangeViolations
      );
      const seen = new Set<string>();
      const rawWithTarget = fallback.rows.filter((r) => {
        const key = JSON.stringify(r);
        if (seen.has(key)) return false;
        seen.add(key);
        return r.final_result !== null;
      });
      return {
        warnings: fallback.warnings,
        cleaned,
        bundle: trainEduPulsePipeline(rawWithTarget, cleaned.trainingRows),
      };
    }

    const cleaned = cleanStudentDataset(
      parsed.rows,
      datasetName,
      parsed.outOfRangeCount,
      parsed.rangeViolations
    );

    // Deduplicate raw rows with target so imputation/scaling fit strictly inside train split
    const seen = new Set<string>();
    const rawWithTarget = parsed.rows.filter((r) => {
      const key = JSON.stringify(r);
      if (seen.has(key)) return false;
      seen.add(key);
      return r.final_result !== null;
    });

    const bundle = trainEduPulsePipeline(rawWithTarget, cleaned.trainingRows);
    return {
      warnings: parsed.warnings,
      cleaned,
      bundle,
    };
  }, [datasetCsv, datasetName]);

  const handleUploadDataset = (csvText: string, fileName: string) => {
    const parsed = parseAndValidateCsv(csvText, true);
    if (!parsed.ok) {
      setUploadError(parsed.error ?? 'Could not parse uploaded CSV file.');
      return;
    }

    const cleaned = cleanStudentDataset(
      parsed.rows,
      fileName,
      parsed.outOfRangeCount,
      parsed.rangeViolations
    );
    if (cleaned.summary.passCount < 4 || cleaned.summary.failCount < 4) {
      setUploadError(
        `Dataset "${fileName}" must contain at least 4 Pass and 4 Fail records with valid final_result labels for stratified training.`
      );
      return;
    }

    setUploadError(null);
    setDatasetCsv(csvText);
    setDatasetName(fileName);
    setIsCustomDataset(true);
  };

  const handleResetDataset = () => {
    setUploadError(null);
    setDatasetCsv(DEFAULT_DATASET_CSV);
    setDatasetName('Track 4 - student_performance.csv');
    setIsCustomDataset(false);
  };

  const handleSimulateFromRegister = (input: StudentFeatureInput, studentId: string) => {
    setStudentDraft(input);
    setWhatIfLabel(studentId);
    setActiveSection('whatif');
  };

  const handleQuickSearchStudent = useCallback((query: string) => {
    setQuickSearchQuery(query);
    setActiveSection('register');
  }, []);

  const handleReportContextChange = useCallback((patch: Partial<ViewReportContext>) => {
    reportContextRef.current = { ...reportContextRef.current, ...patch };
  }, []);

  const handleExportPdf = useCallback(async () => {
    if (isExportingPdf) return;
    setIsExportingPdf(true);
    setExportNotice(null);
    try {
      const spec = buildViewReportSpec({
        activeSection,
        activeModel,
        datasetName,
        summary: workspace.cleaned.summary,
        trainingRows: workspace.cleaned.trainingRows,
        bundle: workspace.bundle,
        context: reportContextRef.current,
      });
      const result = await generateAndDownloadViewPdf(spec, mainRef.current);
      setExportNotice({
        type: 'success',
        message: `Downloaded ${result.filename} (${result.pageCount} ${result.pageCount === 1 ? 'page' : 'pages'}, ${result.chartCount} ${result.chartCount === 1 ? 'chart' : 'charts'}).`,
      });
    } catch {
      setExportNotice({
        type: 'error',
        message: 'Could not generate the PDF report for this view. Please try again.',
      });
    } finally {
      setIsExportingPdf(false);
    }
  }, [isExportingPdf, activeSection, activeModel, datasetName, workspace]);

  return (
    <MotionConfig reducedMotion="user">
      <div className="flex min-h-screen text-slate-100">
      <Sidebar
        activeSection={activeSection}
        onSelectSection={setActiveSection}
        summary={workspace.cleaned.summary}
        mobileOpen={mobileMenuOpen}
        onCloseMobile={() => setMobileMenuOpen(false)}
      />

      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar
          activeSection={activeSection}
          datasetName={datasetName}
          isCustomDataset={isCustomDataset}
          activeModel={activeModel}
          isExportingPdf={isExportingPdf}
          onChangeModel={setActiveModel}
          onExportPdf={handleExportPdf}
          onUploadDataset={handleUploadDataset}
          onResetDataset={handleResetDataset}
          onOpenMobileMenu={() => setMobileMenuOpen(true)}
          onQuickSearchStudent={handleQuickSearchStudent}
        />

        <main ref={mainRef} className="mx-auto w-full max-w-[1320px] flex-1 px-4 py-6 sm:px-6 lg:px-8">
          {exportNotice && (
            <div
              role={exportNotice.type === 'error' ? 'alert' : 'status'}
              data-testid="pdf-export-notice"
              className={`mb-5 flex items-center justify-between gap-3 rounded-xl border px-4 py-2.5 text-xs backdrop-blur-md ${
                exportNotice.type === 'error'
                  ? 'border-rose-500/30 bg-rose-500/15 text-rose-200'
                  : 'border-emerald-400/30 bg-emerald-500/15 text-emerald-200'
              }`}
            >
              <div className="flex items-center gap-2">
                {exportNotice.type === 'error' ? (
                  <AlertTriangle className="h-4 w-4 shrink-0 text-rose-400" aria-hidden="true" />
                ) : (
                  <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-400" aria-hidden="true" />
                )}
                <span className="font-medium">{exportNotice.message}</span>
              </div>
              <button
                type="button"
                onClick={() => setExportNotice(null)}
                aria-label="Dismiss PDF export notification"
                className="rounded p-1 hover:bg-white/10"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </div>
          )}

          {uploadError && (
            <div
              role="alert"
              className="mb-6 flex items-start justify-between gap-3 rounded-xl border border-rose-500/30 bg-rose-500/15 px-4 py-3 text-sm text-rose-200 backdrop-blur-md"
            >
              <div className="flex items-start gap-2.5">
                <AlertTriangle
                  className="mt-0.5 h-4 w-4 shrink-0 text-rose-400"
                  aria-hidden="true"
                />
                <div>
                  <p className="font-semibold text-rose-100">Dataset Upload Rejected</p>
                  <p className="mt-0.5 text-xs text-rose-200/90">{uploadError}</p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setUploadError(null)}
                aria-label="Dismiss dataset upload error"
                className="rounded p-1 text-rose-300 hover:bg-white/10"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          )}

          {activeSection === 'overview' && (
            <OverviewView
              summary={workspace.cleaned.summary}
              trainingRows={workspace.cleaned.trainingRows}
              scoredCohort={workspace.bundle.scoredCohort}
              bundle={workspace.bundle}
              activeModel={activeModel}
              warnings={workspace.warnings}
              onNavigate={setActiveSection}
            />
          )}

          {activeSection === 'analytics' && (
            <AnalyticsView trainingRows={workspace.cleaned.trainingRows} />
          )}

          {activeSection === 'prediction' && (
            <PredictionView
              bundle={workspace.bundle}
              activeModel={activeModel}
              onChangeModel={setActiveModel}
              studentDraft={studentDraft}
              onUpdateDraft={(next) => {
                setStudentDraft(next);
                setWhatIfLabel('Custom Student Draft');
              }}
              onReportContextChange={handleReportContextChange}
            />
          )}

          {activeSection === 'register' && (
            <RiskRegisterView
              scoredCohort={workspace.bundle.scoredCohort}
              activeModel={activeModel}
              externalSearchQuery={quickSearchQuery}
              onSimulateStudent={handleSimulateFromRegister}
              onReportContextChange={handleReportContextChange}
            />
          )}

          {activeSection === 'batch' && (
            <BatchAnalysisView
              bundle={workspace.bundle}
              activeModel={activeModel}
              onReportContextChange={handleReportContextChange}
            />
          )}

          {activeSection === 'whatif' && (
            <WhatIfView
              bundle={workspace.bundle}
              activeModel={activeModel}
              baseStudent={studentDraft}
              studentLabel={whatIfLabel}
              onUpdateBaseStudent={(next) => {
                setStudentDraft(next);
                setWhatIfLabel('Sample At-Risk Profile');
              }}
              onReportContextChange={handleReportContextChange}
            />
          )}
        </main>
      </div>
    </div>
    </MotionConfig>
  );
}
