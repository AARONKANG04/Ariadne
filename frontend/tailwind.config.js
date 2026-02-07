/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translate(0, 0)' },
          '50%': { transform: 'translate(20px, -20px)' }
        },
        drift: {
          '0%, 100%': { transform: 'translate(0, 0)' },
          '50%': { transform: 'translate(15px, 15px)' }
        },
        drift2: {
          '0%, 100%': { transform: 'translate(0, 0)' },
          '50%': { transform: 'translate(-12px, 18px)' }
        },
        drift3: {
          '0%, 100%': { transform: 'translate(0, 0)' },
          '50%': { transform: 'translate(10px, -15px)' }
        }
      },
      animation: {
        'float': 'float 8s ease-in-out infinite',
        'float-delay-1': 'float 10s ease-in-out 1s infinite',
        'float-delay-2': 'float 12s ease-in-out 2s infinite',
        'drift': 'drift 20s ease-in-out infinite',
        'drift2': 'drift2 25s ease-in-out infinite',
        'drift3': 'drift3 30s ease-in-out infinite'
      }
    }
  },
  plugins: [],
}