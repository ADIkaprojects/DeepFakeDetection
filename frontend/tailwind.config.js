/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        display: ["Space Grotesk", "Segoe UI", "sans-serif"],
        mono: ["IBM Plex Mono", "ui-monospace", "monospace"],
      },
      boxShadow: {
        panel: "0 10px 30px rgba(15, 23, 42, 0.09)",
      },
    },
  },
  plugins: [],
};

