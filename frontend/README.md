# Cleo web app

The Cleo model-launch interface is a Vite + React + TypeScript application built from shadcn/ui components and Tailwind CSS. It reads the Cleo 1 release profile from `/api/profile` and streams local inference from `/api/generate` as newline-delimited JSON.

## Development

Run the Python API on port 7860, then start Vite:

```bash
uv run cleo-1 web --checkpoint artifacts/best.pt --device mps --no-browser
cd frontend
npm install
npm run dev
```

Vite proxies `/api` to `http://127.0.0.1:7860`.

## Production build

```bash
npm run build
```

The build is emitted to `../cleo1/static/`, where FastAPI serves it through the existing `cleo-1 web` command.
