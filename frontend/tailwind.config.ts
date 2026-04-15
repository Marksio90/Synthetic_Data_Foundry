import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        'bg-base': '#0d1117',
        'bg-surface': '#161b22',
        'bg-surface2': '#21262d',
        border: '#30363d',
        accent: '#58a6ff',
        success: '#3fb950',
        warning: '#d29922',
        error: '#f85149',
        text: '#c9d1d9',
        'text-muted': '#8b949e',
      },
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'SF Mono', 'Menlo', 'Consolas', 'Liberation Mono', 'monospace'],
      },
    },
  },
  plugins: [],
};

export default config;
