import { useEffect, useMemo, useState } from 'react';
import { ArrowDownUp, Search, SlidersHorizontal } from 'lucide-react';
import {
  ModelType,
  NumericFeatureKey,
  RiskTier,
  ScoredStudentRecord,
  StudentFeatureInput,
} from '../../types/student';
import { classifyRiskTier } from '../../ml/pipeline';
import { ViewReportContext } from '../../utils/reportData';
import { PlotlyChart } from '../PlotlyChart';
import { GlassMotionCard } from '../MotionPrimitives';

interface RiskRegisterViewProps {
  scoredCohort: ScoredStudentRecord[];
  activeModel: ModelType;
  externalSearchQuery?: string;
  onSimulateStudent: (input: StudentFeatureInput, studentId: string) => void;
  onReportContextChange?: (patch: Partial<ViewReportContext>) => void;
}

type SortKey = 'failProb' | 'attendance' | 'midterm' | 'cgpa';

export function RiskRegisterView({
  scoredCohort,
  activeModel,
  externalSearchQuery,
  onSimulateStudent,
  onReportContextChange,
}: RiskRegisterViewProps) {
  const [searchQuery, setSearchQuery] = useState(externalSearchQuery ?? '');
  const [tierFilter, setTierFilter] = useState<'All' | RiskTier>('All');
  const [sortKey, setSortKey] = useState<SortKey>('failProb');
  const [sortAsc, setSortAsc] = useState<boolean>(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  useEffect(() => {
    if (externalSearchQuery !== undefined && externalSearchQuery !== '') {
      setSearchQuery(externalSearchQuery);
    }
  }, [externalSearchQuery]);

  const enriched = useMemo(() => {
    const mapped = scoredCohort.map((s) => {
      const failProb =
        activeModel === 'logistic_regression' ? s.lrFailProbability : s.rfFailProbability;
      return {
        ...s,
        activeFailProb: failProb,
        activeTier: classifyRiskTier(failProb),
      };
    });

    return mapped.sort((a, b) => {
      let diff = 0;
      if (sortKey === 'failProb') diff = a.activeFailProb - b.activeFailProb;
      else if (sortKey === 'attendance') diff = a.attendance_pct - b.attendance_pct;
      else if (sortKey === 'midterm') diff = a.mid_marks - b.mid_marks;
      else diff = a.prev_cgpa - b.prev_cgpa;
      return sortAsc ? diff : -diff;
    });
  }, [scoredCohort, activeModel, sortKey, sortAsc]);

  const passMeans = useMemo(() => {
    const passOnly = scoredCohort.filter((s) => s.final_result === 'Pass');
    const source = passOnly.length > 0 ? passOnly : scoredCohort;
    const avg = (k: NumericFeatureKey) =>
      source.length > 0 ? source.reduce((acc, r) => acc + r[k], 0) / source.length : 0;
    return {
      attendance_pct: avg('attendance_pct'),
      quiz_avg: avg('quiz_avg'),
      assignment_avg: avg('assignment_avg'),
      mid_marks: avg('mid_marks'),
      lab_marks: avg('lab_marks'),
      study_hours_week: avg('study_hours_week'),
      prev_cgpa: avg('prev_cgpa'),
    };
  }, [scoredCohort]);

  const filtered = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    return enriched.filter((s) => {
      if (tierFilter !== 'All' && s.activeTier !== tierFilter) return false;
      if (q && !s.student_id.toLowerCase().includes(q)) return false;
      return true;
    });
  }, [enriched, searchQuery, tierFilter]);

  const selectedStudent = useMemo(() => {
    if (selectedId) {
      return enriched.find((s) => s.student_id === selectedId) ?? filtered[0] ?? null;
    }
    return filtered[0] ?? null;
  }, [enriched, filtered, selectedId]);

  useEffect(() => {
    onReportContextChange?.({
      register: {
        tierFilter,
        searchQuery,
        filteredRows: filtered,
        selectedStudent,
      },
    });
  }, [onReportContextChange, tierFilter, searchQuery, filtered, selectedStudent]);

  const tierCounts = useMemo(() => {
    return {
      All: enriched.length,
      High: enriched.filter((s) => s.activeTier === 'High').length,
      Medium: enriched.filter((s) => s.activeTier === 'Medium').length,
      Low: enriched.filter((s) => s.activeTier === 'Low').length,
    };
  }, [enriched]);

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortAsc((prev) => !prev);
    } else {
      setSortKey(key);
      setSortAsc(key === 'attendance' || key === 'midterm' || key === 'cgpa');
    }
  };

  const studentFingerprintData = useMemo(() => {
    if (!selectedStudent) return [];
    const specs: Array<{ label: string; key: NumericFeatureKey; max: number }> = [
      { label: 'CGPA', key: 'prev_cgpa', max: 4 },
      { label: 'Study Hrs', key: 'study_hours_week', max: 40 },
      { label: 'Lab', key: 'lab_marks', max: 20 },
      { label: 'Midterm', key: 'mid_marks', max: 30 },
      { label: 'Assign.', key: 'assignment_avg', max: 10 },
      { label: 'Quiz', key: 'quiz_avg', max: 10 },
      { label: 'Attend.', key: 'attendance_pct', max: 100 },
    ];

    return [
      {
        type: 'bar',
        orientation: 'h',
        name: 'Pass Cohort Avg',
        y: specs.map((s) => s.label),
        x: specs.map((s) => Number(((passMeans[s.key] / s.max) * 100).toFixed(1))),
        marker: { color: 'rgba(148, 163, 184, 0.55)' },
      },
      {
        type: 'bar',
        orientation: 'h',
        name: selectedStudent.student_id,
        y: specs.map((s) => s.label),
        x: specs.map((s) => Number(((selectedStudent[s.key] / s.max) * 100).toFixed(1))),
        marker: {
          color:
            selectedStudent.activeTier === 'High'
              ? '#fb7185'
              : selectedStudent.activeTier === 'Medium'
              ? '#fbbf24'
              : '#34d399',
        },
      },
    ];
  }, [selectedStudent, passMeans]);

  return (
    <div className="space-y-6">
      <p className="text-sm text-slate-300">
        Full cohort ranked by predicted failure risk using{' '}
        {activeModel === 'random_forest' ? 'Random Forest' : 'Logistic Regression'}. Select any
        student to inspect academic fingerprint diagnostics and advising actions.
      </p>

      {/* Filter & Search Toolbar */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="relative w-full sm:max-w-xs">
          <Search
            className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400"
            aria-hidden="true"
          />
          <input
            type="search"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Filter by Student ID (e.g. STU-1042)"
            aria-label="Filter students by ID"
            className="glass w-full rounded-xl py-2 pl-9 pr-3 text-sm text-white placeholder:text-slate-400 focus:outline-none"
          />
        </div>

        <div
          role="group"
          aria-label="Filter by risk tier"
          className="glass flex flex-wrap items-center gap-1 rounded-xl p-1"
        >
          {(['All', 'High', 'Medium', 'Low'] as const).map((tier) => (
            <button
              key={tier}
              type="button"
              onClick={() => setTierFilter(tier)}
              className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors whitespace-nowrap tabular-nums shrink-0 ${
                tierFilter === tier
                  ? 'bg-white text-slate-950 shadow-xs'
                  : 'text-slate-300 hover:text-white'
              }`}
            >
              {tier} ({tierCounts[tier]})
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-5 xl:grid-cols-12">
        {/* Main Register Table */}
        <GlassMotionCard delay={0.04} className="overflow-hidden p-0 xl:col-span-8">
          <div className="max-h-[600px] overflow-auto">
            <table className="w-full border-collapse text-left text-xs">
              <thead className="sticky top-0 z-10 border-b border-white/10 bg-[#0d1424]/90 text-slate-300 backdrop-blur-md">
                <tr>
                  <th className="py-2.5 pl-4 pr-3 font-medium">Student ID</th>
                  <th className="py-2.5 px-3 text-right font-medium">
                    <button
                      type="button"
                      onClick={() => toggleSort('failProb')}
                      className="inline-flex items-center gap-1 hover:text-white"
                    >
                      <span>Fail Prob.</span>
                      <ArrowDownUp className="h-3 w-3" aria-hidden="true" />
                    </button>
                  </th>
                  <th className="py-2.5 px-3 font-medium">Risk Tier</th>
                  <th className="py-2.5 px-3 text-right font-medium">
                    <button
                      type="button"
                      onClick={() => toggleSort('attendance')}
                      className="inline-flex items-center gap-1 hover:text-white"
                    >
                      <span>Attendance</span>
                      <ArrowDownUp className="h-3 w-3" aria-hidden="true" />
                    </button>
                  </th>
                  <th className="py-2.5 px-3 text-right font-medium">
                    <button
                      type="button"
                      onClick={() => toggleSort('midterm')}
                      className="inline-flex items-center gap-1 hover:text-white"
                    >
                      <span>Midterm</span>
                      <ArrowDownUp className="h-3 w-3" aria-hidden="true" />
                    </button>
                  </th>
                  <th className="py-2.5 px-3 text-right font-medium">
                    <button
                      type="button"
                      onClick={() => toggleSort('cgpa')}
                      className="inline-flex items-center gap-1 hover:text-white"
                    >
                      <span>CGPA</span>
                      <ArrowDownUp className="h-3 w-3" aria-hidden="true" />
                    </button>
                  </th>
                  <th className="py-2.5 pl-3 pr-4 text-right font-medium">Outcome</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/[0.06]">
                {filtered.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="py-10 text-center text-sm text-slate-400">
                      No students match the current filter.
                    </td>
                  </tr>
                ) : (
                  filtered.map((student) => {
                    const isSelected = selectedStudent?.student_id === student.student_id;
                    const barColor =
                      student.activeTier === 'High'
                        ? 'bg-rose-400'
                        : student.activeTier === 'Medium'
                        ? 'bg-amber-400'
                        : 'bg-emerald-400';

                    return (
                      <tr
                        key={student.student_id}
                        className={`cursor-pointer transition-colors ${
                          isSelected ? 'bg-white/[0.08]' : 'hover:bg-white/[0.025]'
                        }`}
                        onClick={() => setSelectedId(student.student_id)}
                      >
                        <td className="py-2.5 pl-4 pr-3 font-mono font-medium text-white">
                          <button
                            type="button"
                            onClick={() => setSelectedId(student.student_id)}
                            className="text-left hover:text-teal-300 focus:outline-none"
                          >
                            {student.student_id}
                          </button>
                        </td>
                        <td className="py-2.5 px-3 text-right">
                          <div className="inline-flex items-center justify-end gap-2">
                            <div className="hidden h-1.5 w-14 overflow-hidden rounded-full bg-white/10 sm:block">
                              <div
                                className={`h-full ${barColor}`}
                                style={{
                                  width: `${Math.min(100, Math.max(3, student.activeFailProb * 100)).toFixed(0)}%`,
                                }}
                              />
                            </div>
                            <span className="font-mono font-semibold text-white tabular-nums">
                              {(student.activeFailProb * 100).toFixed(1)}%
                            </span>
                          </div>
                        </td>
                        <td className="py-2.5 px-3 font-semibold">
                          <span
                            className={
                              student.activeTier === 'High'
                                ? 'text-rose-300'
                                : student.activeTier === 'Medium'
                                ? 'text-amber-300'
                                : 'text-emerald-300'
                            }
                          >
                            {student.activeTier}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 text-right font-mono text-slate-200 tabular-nums">
                          {student.attendance_pct.toFixed(1)}%
                        </td>
                        <td className="py-2.5 px-3 text-right font-mono text-slate-200 tabular-nums">
                          {student.mid_marks.toFixed(1)}/30
                        </td>
                        <td className="py-2.5 px-3 text-right font-mono text-slate-200 tabular-nums">
                          {student.prev_cgpa.toFixed(2)}
                        </td>
                        <td className="py-2.5 pl-3 pr-4 text-right font-medium">
                          <span
                            className={
                              student.final_result === 'Fail' ? 'text-rose-300' : 'text-slate-300'
                            }
                          >
                            {student.final_result ?? '-'}
                          </span>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </GlassMotionCard>

        {/* Selected Student Detail Drawer */}
        <GlassMotionCard variant="glass-strong" delay={0.1} className="p-5 xl:col-span-4">
          {selectedStudent ? (
            <div className="space-y-4">
              <div className="flex items-start justify-between border-b border-white/10 pb-3">
                <div>
                  <p className="text-xs text-slate-400">Selected Student Diagnostic</p>
                  <h2 className="mt-0.5 font-mono text-lg font-bold text-white">
                    {selectedStudent.student_id}
                  </h2>
                </div>
                <div className="text-right">
                  <p className="font-mono text-lg font-bold text-white tabular-nums">
                    {(selectedStudent.activeFailProb * 100).toFixed(1)}%
                  </p>
                  <p
                    className={`text-xs font-semibold ${
                      selectedStudent.activeTier === 'High'
                        ? 'text-rose-300'
                        : selectedStudent.activeTier === 'Medium'
                        ? 'text-amber-300'
                        : 'text-emerald-300'
                    }`}
                  >
                    {selectedStudent.activeTier} Risk
                  </p>
                </div>
              </div>

              <div>
                <p className="text-xs font-medium text-slate-300">
                  Student vs Passing Cohort (% of Scale)
                </p>
                <PlotlyChart
                  data={studentFingerprintData}
                  layout={{
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    barmode: 'group',
                    barcornerradius: 16,
                    bargap: 0.28,
                    bargroupgap: 0.14,
                    margin: { t: 28, r: 12, b: 28, l: 64 },
                    xaxis: {
                      range: [0, 105],
                      gridcolor: 'rgba(255,255,255,0.045)',
                      zeroline: false,
                      ticksuffix: '%',
                    },
                    yaxis: { automargin: true, showgrid: false },
                    legend: { orientation: 'h', y: 1.18, x: 0 },
                  }}
                  config={{ displayModeBar: false }}
                  height={215}
                  reportTitle={`Academic Fingerprint: ${selectedStudent.student_id} vs Passing Cohort`}
                  reportSubtitle={`Fail probability ${(selectedStudent.activeFailProb * 100).toFixed(1)}% (${selectedStudent.activeTier} Risk)`}
                  ariaLabel="Comparison bar chart of selected student versus passing cohort averages"
                />
              </div>

              <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
                <div className="flex justify-between border-b border-white/[0.06] py-1">
                  <span className="text-slate-400">Attendance</span>
                  <span className="font-mono font-medium text-white tabular-nums">
                    {selectedStudent.attendance_pct.toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between border-b border-white/[0.06] py-1">
                  <span className="text-slate-400">Prev. CGPA</span>
                  <span className="font-mono font-medium text-white tabular-nums">
                    {selectedStudent.prev_cgpa.toFixed(2)}
                  </span>
                </div>
                <div className="flex justify-between border-b border-white/[0.06] py-1">
                  <span className="text-slate-400">Quiz Avg</span>
                  <span className="font-mono font-medium text-white tabular-nums">
                    {selectedStudent.quiz_avg.toFixed(1)}/10
                  </span>
                </div>
                <div className="flex justify-between border-b border-white/[0.06] py-1">
                  <span className="text-slate-400">Assignment</span>
                  <span className="font-mono font-medium text-white tabular-nums">
                    {selectedStudent.assignment_avg.toFixed(1)}/10
                  </span>
                </div>
                <div className="flex justify-between border-b border-white/[0.06] py-1">
                  <span className="text-slate-400">Midterm</span>
                  <span className="font-mono font-medium text-white tabular-nums">
                    {selectedStudent.mid_marks.toFixed(1)}/30
                  </span>
                </div>
                <div className="flex justify-between border-b border-white/[0.06] py-1">
                  <span className="text-slate-400">Lab Marks</span>
                  <span className="font-mono font-medium text-white tabular-nums">
                    {selectedStudent.lab_marks.toFixed(1)}/20
                  </span>
                </div>
                <div className="flex justify-between border-b border-white/[0.06] py-1">
                  <span className="text-slate-400">Study Hrs/Wk</span>
                  <span className="font-mono font-medium text-white tabular-nums">
                    {selectedStudent.study_hours_week.toFixed(1)}
                  </span>
                </div>
                <div className="flex justify-between border-b border-white/[0.06] py-1">
                  <span className="text-slate-400">Part-Time Job</span>
                  <span className="font-medium text-white">
                    {selectedStudent.part_time_job}
                  </span>
                </div>
              </div>

              <div>
                <h3 className="text-xs font-semibold text-slate-200">Advisor Action Plan</h3>
                <ul className="mt-1.5 space-y-1.5 text-xs leading-relaxed text-slate-300">
                  {selectedStudent.recommendations.map((rec, i) => (
                    <li key={i}>
                      <span className="font-mono font-semibold text-teal-300">{i + 1}.</span> {rec}
                    </li>
                  ))}
                </ul>
              </div>

              <button
                type="button"
                onClick={() =>
                  onSimulateStudent(
                    {
                      attendance_pct: selectedStudent.attendance_pct,
                      quiz_avg: selectedStudent.quiz_avg,
                      assignment_avg: selectedStudent.assignment_avg,
                      mid_marks: selectedStudent.mid_marks,
                      lab_marks: selectedStudent.lab_marks,
                      study_hours_week: selectedStudent.study_hours_week,
                      prev_cgpa: selectedStudent.prev_cgpa,
                      part_time_job: selectedStudent.part_time_job,
                    },
                    selectedStudent.student_id
                  )
                }
                className="flex w-full items-center justify-center gap-2 rounded-xl bg-white px-4 py-2.5 text-xs font-semibold text-slate-950 shadow-[0_10px_28px_-8px_rgba(255,255,255,0.4)] transition-colors hover:bg-slate-100 whitespace-nowrap"
              >
                <SlidersHorizontal className="h-3.5 w-3.5" aria-hidden="true" />
                <span>Open in What-if Simulator</span>
              </button>
            </div>
          ) : (
            <p className="text-sm text-slate-400">Select a student from the register to inspect.</p>
          )}
        </GlassMotionCard>
      </div>
    </div>
  );
}
