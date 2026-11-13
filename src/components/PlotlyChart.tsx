import { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist-min';

interface PlotlyChartProps {
  data: Array<Record<string, unknown>>;
  layout?: Record<string, unknown>;
  config?: Record<string, unknown>;
  className?: string;
  height?: number;
  ariaLabel?: string;
  reportTitle?: string;
  reportSubtitle?: string;
  includeInReport?: boolean;
}

export function PlotlyChart({
  data,
  layout = {},
  config = {},
  className = '',
  height = 320,
  ariaLabel,
  reportTitle,
  reportSubtitle,
  includeInReport = true,
}: PlotlyChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const mergedLayout: Record<string, unknown> = {
      font: {
        family: 'Plus Jakarta Sans, -apple-system, BlinkMacSystemFont, sans-serif',
        color: '#cbd5e1',
        size: 11.5,
      },
      barcornerradius: 16,
      margin: { t: 42, r: 20, b: 48, l: 54 },
      hoverlabel: {
        bgcolor: 'rgba(9, 14, 28, 0.95)',
        bordercolor: 'rgba(255,255,255,0.18)',
        font: { color: '#f8fafc', family: 'Plus Jakarta Sans, sans-serif', size: 12 },
      },
      ...layout,
      paper_bgcolor: (layout.paper_bgcolor as string | undefined) ?? 'transparent',
      plot_bgcolor: (layout.plot_bgcolor as string | undefined) ?? 'transparent',
      height,
      autosize: true,
    };

    const mergedConfig: Record<string, unknown> = {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d', 'toggleSpikelines'],
      ...config,
    };

    void Plotly.react(el, data, mergedLayout, mergedConfig);

    const observer = new ResizeObserver(() => {
      if (el) {
        Plotly.Plots.resize(el);
      }
    });
    observer.observe(el);

    return () => {
      observer.disconnect();
    };
  }, [data, layout, config, height]);

  useEffect(() => {
    const el = containerRef.current;
    return () => {
      if (el) {
        Plotly.purge(el);
      }
    };
  }, []);

  return (
    <div
      ref={containerRef}
      role="img"
      aria-label={ariaLabel ?? 'Interactive analytics chart'}
      data-report-chart={includeInReport ? 'true' : undefined}
      data-report-title={reportTitle ?? ariaLabel ?? 'Interactive analytics chart'}
      data-report-subtitle={reportSubtitle ?? ''}
      className={`w-full overflow-hidden rounded-[16px] ${className}`}
      style={{ minHeight: `${height}px`, borderRadius: '16px' }}
    />
  );
}
