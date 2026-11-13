declare module 'plotly.js-dist-min' {
  const Plotly: {
    newPlot: (
      root: HTMLElement,
      data: Array<Record<string, unknown>>,
      layout?: Record<string, unknown>,
      config?: Record<string, unknown>
    ) => Promise<HTMLElement>;
    react: (
      root: HTMLElement,
      data: Array<Record<string, unknown>>,
      layout?: Record<string, unknown>,
      config?: Record<string, unknown>
    ) => Promise<HTMLElement>;
    purge: (root: HTMLElement) => void;
    toImage: (
      root: HTMLElement,
      opts: {
        format: 'png' | 'jpeg' | 'webp' | 'svg';
        width?: number;
        height?: number;
        scale?: number;
      }
    ) => Promise<string>;
    Plots: {
      resize: (root: HTMLElement) => void;
    };
  };
  export default Plotly;
}
