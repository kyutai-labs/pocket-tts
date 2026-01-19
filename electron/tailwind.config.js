/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/renderer/**/*.{html,js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        'bg-primary': '#1a1a2e',
        'bg-secondary': '#16213e',
        'bg-tertiary': '#0f3460',
        'text-primary': '#e8e8e8',
        'text-secondary': '#a0a0a0',
        'accent': '#ff6b35',
        'accent-hover': '#ff8c5a',
        'border-color': '#2a2a4a',
      },
    },
  },
  plugins: [],
};
