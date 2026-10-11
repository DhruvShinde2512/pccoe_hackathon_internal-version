import defaultColors from 'tailwindcss/colors';

/**
 * Light theme.
 *
 * Components were authored against a dark palette (bg-slate-950 page, text-slate-200 body,
 * red-300 accent text, ...). Rather than rewriting every class, the shade scales are remapped
 * here: slate becomes a light surface/dark ink scale, and accent hues are mirrored around 500
 * so light-on-dark accents (e.g. text-red-300) render as dark-on-light (red-700).
 */
const slate = defaultColors.slate;
const lightSlate = {
  50: slate[950], // strongest ink
  100: slate[900],
  200: slate[800], // body text
  300: slate[700],
  400: slate[600],
  500: slate[500],
  600: slate[400],
  700: slate[300], // strong borders
  800: slate[200], // borders, subtle fills
  900: '#FFFFFF', // panels
  950: slate[100], // page background
};

const mirror = (scale) => ({
  50: scale[950],
  100: scale[900],
  200: scale[800],
  300: scale[700],
  400: scale[600],
  500: scale[500],
  600: scale[400],
  700: scale[300],
  800: scale[200],
  900: scale[100],
  950: scale[50],
});

const ACCENTS = ['red', 'rose', 'orange', 'amber', 'yellow', 'emerald', 'green', 'sky', 'blue', 'violet', 'purple', 'indigo', 'cyan', 'teal'];

/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        slate: lightSlate,
        ...Object.fromEntries(ACCENTS.map((name) => [name, mirror(defaultColors[name])])),
        entity: {
          employee: '#3B82F6',
          account: '#A855F7',
          transaction: '#10B981',
          anomaly: '#EF4444',
        },
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      keyframes: {
        'toast-in': {
          '0%': { opacity: '0', transform: 'translateY(8px) scale(0.98)' },
          '100%': { opacity: '1', transform: 'translateY(0) scale(1)' },
        },
        'drawer-in': {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(0)' },
        },
      },
      animation: {
        'toast-in': 'toast-in 180ms ease-out',
        'drawer-in': 'drawer-in 220ms ease-out',
      },
    },
  },
  plugins: [],
};
