import { jsPDF } from 'jspdf';
import Plotly from 'plotly.js-dist-min';
import { ReportVectorBar, ViewReportSpec, toPdfSafeText } from './reportData';

export interface CapturedReportChart {
  title: string;
  subtitle?: string;
  dataUrl: string;
  aspectRatio: number; // width / height
}

export interface GeneratedPdfResult {
  filename: string;
  chartCount: number;
  pageCount: number;
}

const PAGE_W = 210;
const PAGE_H = 297;
const MARGIN_X = 14;
const CONTENT_W = PAGE_W - MARGIN_X * 2; // 182 mm
const MAX_Y = PAGE_H - 18; // 279 mm

const COLORS = {
  navy: [11, 22, 44] as const,
  teal: [13, 148, 136] as const,
  tealLight: [94, 234, 212] as const,
  red: [220, 38, 38] as const,
  amber: [217, 119, 6] as const,
  slate900: [15, 23, 42] as const,
  slate700: [51, 65, 85] as const,
  slate500: [100, 116, 139] as const,
  slate300: [203, 213, 225] as const,
  slate200: [226, 232, 240] as const,
  slate100: [241, 245, 249] as const,
  slate50: [248, 250, 252] as const,
  white: [255, 255, 255] as const,
};

function hexToRgb(hex: string): [number, number, number] {
  const clean = hex.replace('#', '');
  if (clean.length !== 6) return [11, 22, 44];
  const r = parseInt(clean.slice(0, 2), 16);
  const g = parseInt(clean.slice(2, 4), 16);
  const b = parseInt(clean.slice(4, 6), 16);
  return [r, g, b];
}

function getAccentRgb(accent?: 'navy' | 'teal' | 'red' | 'amber'): readonly [number, number, number] {
  if (accent === 'teal') return COLORS.teal;
  if (accent === 'red') return COLORS.red;
  if (accent === 'amber') return COLORS.amber;
  return COLORS.navy;
}

/**
 * Captures all rendered Plotly charts inside the active workspace container as crisp PNG images.
 */
export async function captureVisiblePlotlyCharts(
  rootElement: HTMLElement | null = document.querySelector('main')
): Promise<CapturedReportChart[]> {
  if (!rootElement) return [];
  const chartEls = Array.from(
    rootElement.querySelectorAll<HTMLElement>('[data-report-chart="true"]')
  );
  const captured: CapturedReportChart[] = [];

  for (const el of chartEls) {
    const rect = el.getBoundingClientRect();
    if (rect.width < 40 || rect.height < 40) continue;

    const title = toPdfSafeText(
      el.dataset.reportTitle || el.getAttribute('aria-label') || 'Analytics Chart'
    );
    const subtitle = el.dataset.reportSubtitle
      ? toPdfSafeText(el.dataset.reportSubtitle)
      : undefined;

    const exportWidth = Math.max(760, Math.round(rect.width * 1.6));
    const exportHeight = Math.max(380, Math.round(rect.height * 1.6));

    try {
      const dataUrl = await Plotly.toImage(el, {
        format: 'png',
        width: exportWidth,
        height: exportHeight,
        scale: 2,
      });
      if (dataUrl && dataUrl.startsWith('data:image/')) {
        captured.push({
          title,
          subtitle,
          dataUrl,
          aspectRatio: exportWidth / exportHeight,
        });
      }
    } catch {
      // Skip any chart whose WebGL/SVG snapshot failed without breaking export
    }
  }

  return captured;
}

function drawContinuationHeader(doc: jsPDF, spec: ViewReportSpec): number {
  doc.setFillColor(...COLORS.navy);
  doc.rect(0, 0, PAGE_W, 13, 'F');
  doc.setFillColor(...COLORS.teal);
  doc.rect(0, 13, PAGE_W, 0.8, 'F');

  doc.setTextColor(...COLORS.white);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(9.5);
  doc.text(toPdfSafeText(`EduPulse | ${spec.title}`), MARGIN_X, 8.5);

  doc.setTextColor(...COLORS.tealLight);
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(8);
  doc.text(toPdfSafeText(spec.modelLabel), PAGE_W - MARGIN_X, 8.5, { align: 'right' });

  return 20;
}

function drawVectorCompositionBar(
  doc: jsPDF,
  bar: ReportVectorBar,
  startY: number
): number {
  const total = bar.segments.reduce((acc, s) => acc + s.count, 0);
  const cardH = 29;

  doc.setFillColor(...COLORS.white);
  doc.setDrawColor(...COLORS.slate300);
  doc.setLineWidth(0.3);
  doc.roundedRect(MARGIN_X, startY, CONTENT_W, cardH, 2.2, 2.2, 'FD');

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(9.5);
  doc.setTextColor(...COLORS.slate900);
  doc.text(toPdfSafeText(bar.title), MARGIN_X + 4, startY + 6);

  if (bar.subtitle) {
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(7.5);
    doc.setTextColor(...COLORS.slate500);
    doc.text(toPdfSafeText(bar.subtitle), MARGIN_X + 4, startY + 10.5);
  }

  const barX = MARGIN_X + 4;
  const barY = startY + 13.5;
  const barW = CONTENT_W - 8;
  const barH = 5.5;

  doc.setFillColor(...COLORS.slate100);
  doc.roundedRect(barX, barY, barW, barH, 1.5, 1.5, 'F');

  if (total > 0) {
    let curX = barX;
    bar.segments.forEach((seg, idx) => {
      if (seg.count <= 0) return;
      const segW =
        idx === bar.segments.length - 1
          ? Math.max(0, barX + barW - curX)
          : (seg.count / total) * barW;
      doc.setFillColor(...hexToRgb(seg.colorHex));
      doc.rect(curX, barY, segW, barH, 'F');
      curX += segW;
    });
  }

  let legendX = barX;
  const legendY = startY + 24.5;
  doc.setFontSize(8);
  bar.segments.forEach((seg) => {
    const pct = total > 0 ? ((seg.count / total) * 100).toFixed(1) : '0.0';
    const [r, g, b] = hexToRgb(seg.colorHex);
    doc.setFillColor(r, g, b);
    doc.rect(legendX, legendY - 2.6, 3, 3, 'F');
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(...COLORS.slate900);
    const labelStr = `${toPdfSafeText(seg.label)}: ${seg.count} (${pct}%)`;
    doc.text(labelStr, legendX + 4.5, legendY);
    legendX += doc.getTextWidth(labelStr) + 12;
  });

  return startY + cardH + 5;
}

export async function generateAndDownloadViewPdf(
  spec: ViewReportSpec,
  rootElement?: HTMLElement | null
): Promise<GeneratedPdfResult> {
  const charts = await captureVisiblePlotlyCharts(rootElement);
  const doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });

  let y = 0;

  const ensureSpace = (neededMm: number) => {
    if (y + neededMm > MAX_Y) {
      doc.addPage();
      y = drawContinuationHeader(doc, spec);
    }
  };

  // 1. Page 1 Executive Midnight Navy Banner
  doc.setFillColor(...COLORS.navy);
  doc.rect(0, 0, PAGE_W, 32, 'F');
  doc.setFillColor(...COLORS.teal);
  doc.rect(0, 32, PAGE_W, 1.5, 'F');

  doc.setTextColor(...COLORS.tealLight);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(8);
  doc.text('EDUPULSE  |  AT-RISK STUDENT EARLY WARNING SYSTEM', MARGIN_X, 9.5);

  doc.setTextColor(...COLORS.white);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(16);
  doc.text(toPdfSafeText(spec.title), MARGIN_X, 17.5);

  doc.setTextColor(203, 213, 225);
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(8.5);
  const subtitleLines = doc.splitTextToSize(toPdfSafeText(spec.subtitle), CONTENT_W);
  doc.text(subtitleLines[0] ?? '', MARGIN_X, 24.5);

  // Metadata Strip
  y = 33.5;
  doc.setFillColor(...COLORS.slate50);
  doc.setDrawColor(...COLORS.slate200);
  doc.setLineWidth(0.25);
  doc.rect(0, y, PAGE_W, 9, 'FD');

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(7.8);
  doc.setTextColor(...COLORS.slate700);
  const metaLeft = toPdfSafeText(`Dataset: ${spec.datasetName}   |   Model: ${spec.modelLabel}`);
  const metaRight = toPdfSafeText(`Generated: ${spec.generatedAt}`);
  doc.text(metaLeft, MARGIN_X, y + 5.8);
  doc.text(metaRight, PAGE_W - MARGIN_X, y + 5.8, { align: 'right' });

  y += 14.5;

  // 2. Key Metrics Section (2x2 Grid)
  if (spec.metrics.length > 0) {
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(9);
    doc.setTextColor(...COLORS.navy);
    doc.text('KEY METRICS SUMMARY', MARGIN_X, y);
    y += 3.5;

    const colGap = 4;
    const cardW = (CONTENT_W - colGap) / 2; // 89mm
    const cardH = 21;

    for (let i = 0; i < spec.metrics.length; i += 2) {
      ensureSpace(cardH + 4);
      for (let col = 0; col < 2; col += 1) {
        const metric = spec.metrics[i + col];
        if (!metric) continue;
        const cardX = MARGIN_X + col * (cardW + colGap);

        doc.setFillColor(...COLORS.white);
        doc.setDrawColor(...COLORS.slate300);
        doc.setLineWidth(0.3);
        doc.roundedRect(cardX, y, cardW, cardH, 2, 2, 'FD');

        // Colored left bar
        const accent = getAccentRgb(metric.accent);
        doc.setFillColor(...accent);
        doc.rect(cardX, y + 2, 1.6, cardH - 4, 'F');

        doc.setFont('helvetica', 'normal');
        doc.setFontSize(7.8);
        doc.setTextColor(...COLORS.slate500);
        doc.text(toPdfSafeText(metric.label), cardX + 4.5, y + 6);

        doc.setFont('helvetica', 'bold');
        doc.setFontSize(14);
        doc.setTextColor(...COLORS.slate900);
        doc.text(toPdfSafeText(metric.value), cardX + 4.5, y + 13);

        if (metric.note) {
          doc.setFont('helvetica', 'normal');
          doc.setFontSize(7.2);
          doc.setTextColor(...COLORS.slate700);
          const noteLines = doc.splitTextToSize(toPdfSafeText(metric.note), cardW - 8);
          doc.text(noteLines[0] ?? '', cardX + 4.5, y + 18);
        }
      }
      y += cardH + 3.5;
    }
    y += 2.5;
  }

  // 3. Vector Composition Chart (if present, e.g. Register or Batch)
  if (spec.vectorChart) {
    ensureSpace(34);
    y = drawVectorCompositionBar(doc, spec.vectorChart, y);
  }

  // 4. Captured Plotly Charts
  if (charts.length > 0) {
    ensureSpace(58);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(9);
    doc.setTextColor(...COLORS.navy);
    doc.text('ANALYSIS CHARTS', MARGIN_X, y);
    y += 3.5;

    const useSingleWide = charts.length === 1;
    if (useSingleWide) {
      const chart = charts[0];
      const cardW = CONTENT_W;
      const imgW = cardW - 6;
      const rawImgH = imgW / Math.max(0.8, chart.aspectRatio);
      const imgH = Math.min(82, Math.max(48, rawImgH));
      const headerH = chart.subtitle ? 12.5 : 8.5;
      const cardH = headerH + imgH + 3;

      ensureSpace(cardH + 4);
      doc.setFillColor(...COLORS.navy);
      doc.setDrawColor(30, 41, 59);
      doc.setLineWidth(0.35);
      doc.roundedRect(MARGIN_X, y, cardW, cardH, 2.2, 2.2, 'FD');

      doc.setFont('helvetica', 'bold');
      doc.setFontSize(9);
      doc.setTextColor(...COLORS.white);
      doc.text(chart.title, MARGIN_X + 3.5, y + 5.5);

      if (chart.subtitle) {
        doc.setFont('helvetica', 'normal');
        doc.setFontSize(7.3);
        doc.setTextColor(...COLORS.slate300);
        const subLines = doc.splitTextToSize(chart.subtitle, cardW - 7);
        doc.text(subLines[0] ?? '', MARGIN_X + 3.5, y + 9.5);
      }

      doc.addImage(chart.dataUrl, 'PNG', MARGIN_X + 3, y + headerH, imgW, imgH, undefined, 'FAST');
      y += cardH + 5;
    } else {
      const colGap = 4;
      const cardW = (CONTENT_W - colGap) / 2; // 89mm
      const imgW = cardW - 5;

      for (let i = 0; i < charts.length; i += 2) {
        const pair = [charts[i], charts[i + 1]].filter(Boolean) as CapturedReportChart[];
        const rowImgH = Math.max(
          ...pair.map((c) => Math.min(58, Math.max(40, imgW / Math.max(0.75, c.aspectRatio))))
        );
        const headerH = pair.some((c) => Boolean(c.subtitle)) ? 12 : 8.5;
        const cardH = headerH + rowImgH + 3;

        ensureSpace(cardH + 4);

        pair.forEach((chart, colIdx) => {
          const cardX = pair.length === 1 ? MARGIN_X : MARGIN_X + colIdx * (cardW + colGap);
          const actualCardW = pair.length === 1 ? CONTENT_W : cardW;
          const actualImgW = actualCardW - 5;
          const actualImgH = Math.min(
            rowImgH,
            Math.max(38, actualImgW / Math.max(0.75, chart.aspectRatio))
          );

          doc.setFillColor(...COLORS.navy);
          doc.setDrawColor(30, 41, 59);
          doc.setLineWidth(0.35);
          doc.roundedRect(cardX, y, actualCardW, cardH, 2.2, 2.2, 'FD');

          doc.setFont('helvetica', 'bold');
          doc.setFontSize(8.6);
          doc.setTextColor(...COLORS.white);
          const titleLines = doc.splitTextToSize(chart.title, actualCardW - 7);
          doc.text(titleLines[0] ?? '', cardX + 3, y + 5.2);

          if (chart.subtitle) {
            doc.setFont('helvetica', 'normal');
            doc.setFontSize(6.9);
            doc.setTextColor(...COLORS.slate300);
            const subLines = doc.splitTextToSize(chart.subtitle, actualCardW - 7);
            doc.text(subLines[0] ?? '', cardX + 3, y + 9.2);
          }

          doc.addImage(
            chart.dataUrl,
            'PNG',
            cardX + 2.5,
            y + headerH,
            actualImgW,
            actualImgH,
            undefined,
            'FAST'
          );
        });

        y += cardH + 4;
      }
      y += 2;
    }
  }

  // 5. Highlights & Recommendations
  for (const block of spec.highlights) {
    ensureSpace(20);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(9);
    doc.setTextColor(...COLORS.navy);
    doc.text(toPdfSafeText(block.title.toUpperCase()), MARGIN_X, y);
    y += 4;

    doc.setFont('helvetica', 'normal');
    doc.setFontSize(8.2);
    doc.setTextColor(...COLORS.slate700);

    for (const item of block.items) {
      const wrapped = doc.splitTextToSize(`- ${toPdfSafeText(item)}`, CONTENT_W - 4);
      const blockH = wrapped.length * 4 + 1.5;
      ensureSpace(blockH);
      doc.text(wrapped, MARGIN_X + 2, y);
      y += blockH;
    }
    y += 3;
  }

  // 6. Summary Table
  if (spec.table && spec.table.rows.length > 0) {
    const colCount = spec.table.headers.length;
    const colW = CONTENT_W / colCount;
    const rowH = 6.5;

    const drawTableHeader = () => {
      doc.setFillColor(...COLORS.navy);
      doc.rect(MARGIN_X, y, CONTENT_W, 7.2, 'F');
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(7.6);
      doc.setTextColor(...COLORS.white);
      spec.table!.headers.forEach((h, idx) => {
        const cellText = doc.splitTextToSize(toPdfSafeText(h), colW - 3)[0] ?? '';
        doc.text(cellText, MARGIN_X + idx * colW + 2, y + 4.8);
      });
      y += 7.2;
    };

    ensureSpace(22);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(9);
    doc.setTextColor(...COLORS.navy);
    doc.text(toPdfSafeText(spec.table.title.toUpperCase()), MARGIN_X, y);
    y += 3.5;

    drawTableHeader();

    spec.table.rows.forEach((row, rIdx) => {
      if (y + rowH > MAX_Y) {
        doc.addPage();
        y = drawContinuationHeader(doc, spec);
        drawTableHeader();
      }

      if (rIdx % 2 === 1) {
        doc.setFillColor(...COLORS.slate50);
        doc.rect(MARGIN_X, y, CONTENT_W, rowH, 'F');
      }

      doc.setDrawColor(...COLORS.slate200);
      doc.setLineWidth(0.2);
      doc.line(MARGIN_X, y + rowH, MARGIN_X + CONTENT_W, y + rowH);

      doc.setFont('helvetica', 'normal');
      doc.setFontSize(7.4);
      doc.setTextColor(...COLORS.slate700);

      row.forEach((cell, cIdx) => {
        const cellText = doc.splitTextToSize(toPdfSafeText(cell), colW - 3)[0] ?? '';
        doc.text(cellText, MARGIN_X + cIdx * colW + 2, y + 4.5);
      });

      y += rowH;
    });

    y += 5;
  }

  // 7. Methodological Notes / Disclaimers
  if (spec.notes.length > 0) {
    const joined = spec.notes.map((n) => toPdfSafeText(n)).join(' ');
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(7.8);
    const wrapped = doc.splitTextToSize(joined, CONTENT_W - 10);
    const boxH = Math.max(11, wrapped.length * 3.8 + 5);

    ensureSpace(boxH + 4);
    doc.setFillColor(...COLORS.slate100);
    doc.setDrawColor(...COLORS.slate300);
    doc.setLineWidth(0.25);
    doc.roundedRect(MARGIN_X, y, CONTENT_W, boxH, 1.8, 1.8, 'FD');

    doc.setFillColor(...COLORS.teal);
    doc.rect(MARGIN_X, y + 1.5, 1.4, boxH - 3, 'F');

    doc.setTextColor(...COLORS.slate700);
    doc.text(wrapped, MARGIN_X + 5, y + 5.2);
    y += boxH + 4;
  }

  // 8. Page Footers on all pages
  const totalPages = doc.getNumberOfPages();
  for (let p = 1; p <= totalPages; p += 1) {
    doc.setPage(p);
    doc.setDrawColor(...COLORS.slate300);
    doc.setLineWidth(0.25);
    doc.line(MARGIN_X, PAGE_H - 12, PAGE_W - MARGIN_X, PAGE_H - 12);

    doc.setFont('helvetica', 'normal');
    doc.setFontSize(7.5);
    doc.setTextColor(...COLORS.slate500);
    doc.text(
      'EduPulse - At-Risk Student Early Warning System',
      MARGIN_X,
      PAGE_H - 7.5
    );
    doc.text(`Page ${p} of ${totalPages}`, PAGE_W - MARGIN_X, PAGE_H - 7.5, {
      align: 'right',
    });
  }

  doc.save(spec.filename);
  return {
    filename: spec.filename,
    chartCount: charts.length + (spec.vectorChart ? 1 : 0),
    pageCount: totalPages,
  };
}
