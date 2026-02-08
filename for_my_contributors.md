# <3

Frontend server and backend server should run at the same time for testing.
I'm sure theres a better way to do this but I'm too sleep deprived to figure it out.


1. Setup a virtual environment and install all packages. (`npm install` and `pip install -r backend/requirements.txt`)
2. From the project root, run `uvicorn backend.main:app --reload --port 8000`
3. In the frontend folder, run `npx vite`