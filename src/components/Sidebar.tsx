import {
  BarChart3,
  BrainCircuit,
  FileSpreadsheet,
  LayoutDashboard,
  SlidersHorizontal,
  Users,
  X,
} from 'lucide-react';
import { DataCleaningSummary } from '../types/student';

export type NavSection =
  | 'overview'
  | 'analytics'
  | 'prediction'
  | 'register'
  | 'batch'
  | 'whatif';

interface SidebarProps {
  activeSection: NavSection;
  onSelectSection: (section: NavSection) => void;
  summary: DataCleaningSummary;
  mobileOpen: boolean;
  onCloseMobile: () => void;
}

const NAV_ITEMS: Array<{
  id: NavSection;
  label: string;
  icon: typeof LayoutDashboard;
}> = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard },
  { id: 'analytics', label: 'Analytics', icon: BarChart3 },
  { id: 'prediction', label: 'AI Prediction', icon: BrainCircuit },
  { id: 'register', label: 'Student Risk Register', icon: Users },
  { id: 'batch', label: 'Batch Analysis', icon: FileSpreadsheet },
  { id: 'whatif', label: 'What-if Simulator', icon: SlidersHorizontal },
];

export function Sidebar({
  activeSection,
  onSelectSection,
  summary,
  mobileOpen,
  onCloseMobile,
}: SidebarProps) {
  const passPct =
    summary.cleanedTrainingRows > 0
      ? (summary.passCount / summary.cleanedTrainingRows) * 100
      : 0;

  const navContent = (
    <div className="glass-soft flex h-full w-[280px] flex-col border-r border-white/[0.08] px-5 py-6 text-slate-100">
      {/* Top logo area: 44px square white box with black character */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3.5">
          <div
            aria-hidden="true"
            className="flex h-[44px] w-[44px] shrink-0 items-center justify-center rounded-2xl bg-white font-display text-xl font-bold text-black shadow-[0_10px_28px_-8px_rgba(255,255,255,0.35)]"
          >
            E
          </div>
          <div>
            <span className="font-display text-lg font-bold tracking-tight text-white whitespace-nowrap">
              EduPulse
            </span>
            <p className="text-[11px] font-medium tracking-wide text-slate-400">
              Early Warning AI
            </p>
          </div>
        </div>

        <button
          type="button"
          onClick={onCloseMobile}
          aria-label="Close navigation menu"
          className="rounded-lg p-1.5 text-slate-400 hover:bg-white/10 hover:text-white lg:hidden"
        >
          <X className="h-5 w-5" />
        </button>
      </div>

      {/* Navigation links with 17px icons and 14px text */}
      <nav aria-label="Primary navigation" className="mt-8 flex-1 space-y-1.5">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const isActive = activeSection === item.id;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => {
                onSelectSection(item.id);
                onCloseMobile();
              }}
              aria-current={isActive ? 'page' : undefined}
              className={`flex w-full items-center gap-3.5 rounded-xl px-3.5 py-2.5 text-left text-[14px] font-medium transition-all duration-150 whitespace-nowrap ${
                isActive
                  ? 'glass-strong text-white'
                  : 'text-slate-400 hover:bg-white/[0.045] hover:text-slate-100'
              }`}
            >
              <Icon
                className={`h-[17px] w-[17px] shrink-0 transition-colors ${
                  isActive ? 'text-white' : 'text-slate-400'
                }`}
                aria-hidden="true"
              />
              <span className="truncate">{item.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Floating Cohort Card at bottom with 5s float animation (translateY: -6px) */}
      <div className="pt-4">
        <div className="glass-strong animate-float rounded-[24px] p-4">
          <div className="flex items-center justify-between text-[11px] uppercase tracking-wider text-slate-400">
            <span>Active Cohort Card</span>
            <span className="font-mono text-emerald-300 tabular-nums">
              {passPct.toFixed(0)}% Pass
            </span>
          </div>
          <p
            className="mt-2 truncate font-display text-sm font-semibold text-white"
            title={summary.datasetName}
          >
            {summary.datasetName}
          </p>
          <p className="mt-0.5 font-mono text-xs text-slate-300 tabular-nums">
            {summary.cleanedTrainingRows} cleaned records
          </p>
          <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-rose-500/35">
            <div
              className="h-full bg-emerald-400 transition-all duration-300"
              style={{ width: `${passPct.toFixed(1)}%` }}
            />
          </div>
          <div className="mt-2 flex items-center justify-between font-mono text-[11px] text-slate-300 tabular-nums">
            <span>{summary.passCount} Pass</span>
            <span>{summary.failCount} Fail</span>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <>
      <aside className="sticky top-0 hidden h-screen w-[280px] shrink-0 lg:block">
        {navContent}
      </aside>

      {mobileOpen && (
        <div className="fixed inset-0 z-50 flex lg:hidden">
          <div
            className="fixed inset-0 bg-black/70 backdrop-blur-xs"
            onClick={onCloseMobile}
            aria-hidden="true"
          />
          <aside className="relative z-10 w-[280px] max-w-[85vw]">{navContent}</aside>
        </div>
      )}
    </>
  );
}
