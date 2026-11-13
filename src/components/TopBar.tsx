import { ChangeEvent, FormEvent, useRef, useState } from 'react';
import { FileDown, Loader2, Menu, RotateCcw, Search, Upload } from 'lucide-react';
import { ModelType } from '../types/student';
import { NavSection } from './Sidebar';

interface TopBarProps {
  activeSection: NavSection;
  datasetName: string;
  isCustomDataset: boolean;
  activeModel: ModelType;
  isExportingPdf: boolean;
  onChangeModel: (model: ModelType) => void;
  onExportPdf: () => void;
  onUploadDataset: (csvText: string, fileName: string) => void;
  onResetDataset: () => void;
  onOpenMobileMenu: () => void;
  onQuickSearchStudent?: (query: string) => void;
}

const SECTION_TITLES: Record<NavSection, string> = {
  overview: 'Overview',
  analytics: 'Analytics',
  prediction: 'AI Prediction',
  register: 'Risk Register',
  batch: 'Batch Analysis',
  whatif: 'What-if Simulator',
};

export function TopBar({
  activeSection,
  datasetName,
  isCustomDataset,
  activeModel,
  isExportingPdf,
  onChangeModel,
  onExportPdf,
  onUploadDataset,
  onResetDataset,
  onOpenMobileMenu,
  onQuickSearchStudent,
}: TopBarProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [searchValue, setSearchValue] = useState('');

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        onUploadDataset(reader.result, file.name);
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  const handleSearchSubmit = (e: FormEvent) => {
    e.preventDefault();
    onQuickSearchStudent?.(searchValue.trim());
  };

  return (
    <header className="sticky top-0 z-30 border-b border-white/[0.08] bg-[#060913]/65 px-4 py-4 backdrop-blur-xl sm:px-6 lg:px-8">
      <div className="mx-auto flex max-w-[1360px] flex-wrap items-center justify-between gap-4">
        {/* Left side: 30px display heading with 14px date/cohort subtext */}
        <div className="flex min-w-0 items-center gap-3">
          <button
            type="button"
            onClick={onOpenMobileMenu}
            aria-label="Open navigation menu"
            className="glass rounded-xl p-2.5 text-slate-200 hover:bg-white/10 lg:hidden"
          >
            <Menu className="h-5 w-5" />
          </button>

          <div className="min-w-0">
            <h1 className="font-display truncate text-[30px] leading-tight font-bold tracking-tight text-white">
              {SECTION_TITLES[activeSection]}
            </h1>
            <p className="truncate text-[14px] text-slate-400">
              Nov 12, 2026 ·{' '}
              <span className="font-mono text-slate-300">{datasetName}</span>
            </p>
          </div>
        </div>

        {/* Right side: 224px glass search bar, model toggle, glowing white primary button, upload CSV, 40px slate gradient avatar */}
        <div className="flex flex-wrap items-center gap-2.5 shrink-0">
          <form onSubmit={handleSearchSubmit} className="relative hidden w-[224px] md:block">
            <Search
              className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400"
              aria-hidden="true"
            />
            <input
              type="search"
              value={searchValue}
              onChange={(e) => setSearchValue(e.target.value)}
              placeholder="Search student ID..."
              aria-label="Quick search student ID"
              className="glass w-full rounded-xl py-2 pl-9 pr-3 text-xs text-white placeholder:text-slate-400 focus:outline-none"
            />
          </form>

          <div
            role="group"
            aria-label="Active ML model"
            className="glass hidden items-center gap-1 rounded-xl p-1 xl:flex"
          >
            <button
              type="button"
              onClick={() => onChangeModel('random_forest')}
              className={`rounded-lg px-2.5 py-1.5 text-xs font-medium transition-all duration-150 whitespace-nowrap shrink-0 ${
                activeModel === 'random_forest'
                  ? 'bg-white text-slate-950 shadow-xs'
                  : 'text-slate-300 hover:text-white'
              }`}
            >
              Random Forest
            </button>
            <button
              type="button"
              onClick={() => onChangeModel('logistic_regression')}
              className={`rounded-lg px-2.5 py-1.5 text-xs font-medium transition-all duration-150 whitespace-nowrap shrink-0 ${
                activeModel === 'logistic_regression'
                  ? 'bg-white text-slate-950 shadow-xs'
                  : 'text-slate-300 hover:text-white'
              }`}
            >
              Logistic Regression
            </button>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,text/csv"
            onChange={handleFileChange}
            aria-label="Upload training dataset CSV"
            className="sr-only"
          />

          {isCustomDataset && (
            <button
              type="button"
              onClick={onResetDataset}
              className="glass inline-flex items-center gap-1.5 rounded-xl px-3 py-2 text-xs font-medium text-slate-200 hover:bg-white/10 whitespace-nowrap shrink-0"
            >
              <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
              <span className="hidden sm:inline">Reset</span>
            </button>
          )}

          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="glass inline-flex items-center gap-1.5 rounded-xl px-3.5 py-2 text-xs font-medium text-slate-100 hover:bg-white/10 whitespace-nowrap shrink-0"
          >
            <Upload className="h-3.5 w-3.5" aria-hidden="true" />
            <span className="hidden sm:inline">Upload CSV</span>
          </button>

          {/* White primary button with heavy glow */}
          <button
            type="button"
            onClick={onExportPdf}
            disabled={isExportingPdf}
            aria-label={`Export PDF summary report for ${SECTION_TITLES[activeSection]}`}
            className="inline-flex items-center gap-1.5 rounded-xl bg-white px-4 py-2 text-xs font-semibold text-slate-950 shadow-[0_10px_28px_-8px_rgba(255,255,255,0.4)] transition-all hover:bg-slate-100 disabled:cursor-wait disabled:opacity-65 whitespace-nowrap shrink-0"
          >
            {isExportingPdf ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
            ) : (
              <FileDown className="h-3.5 w-3.5" aria-hidden="true" />
            )}
            <span>{isExportingPdf ? 'Exporting...' : 'Export PDF'}</span>
          </button>

          {/* Circular avatar (40px) with slate gradient */}
          <div
            aria-label="Academic advisor profile"
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full border border-white/25 bg-gradient-to-br from-slate-500 via-slate-700 to-slate-900 font-display text-xs font-bold text-white shadow-inner"
          >
            EP
          </div>
        </div>
      </div>
    </header>
  );
}
