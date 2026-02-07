# Mission Control Frontend (`frontend/`)

This package is the **Next.js** web UI for OpenClaw Mission Control.

- Talks to the Mission Control **backend** over HTTP (typically `http://localhost:8000`).
- Uses **React Query** for data fetching.
- Can optionally enable **Clerk** authentication (disabled by default unless you provide a *real* Clerk publishable key).

## Prerequisites

- Node.js (recommend **18+**) and npm
- Backend running locally (see `../backend/README.md` if present) **or** run the stack via Docker Compose from repo root.

## Local development

From `frontend/`:

```bash
npm install

# set env vars (see below)
cp .env.example .env.local

npm run dev
```

Open http://localhost:3000.

### LAN development

To bind Next dev server to all interfaces:

```bash
npm run dev:lan
```

## Environment variables

The frontend reads configuration from standard Next.js env files (`.env.local`, `.env`, etc.).

### Required

#### `NEXT_PUBLIC_API_URL`

Base URL of the backend API.

- Default for local dev: `http://localhost:8000`
- Used by the generated API client and helpers (see `src/lib/api-base.ts` and `src/api/mutator.ts`).

Example:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Optional: Clerk authentication

Clerk is **optional**.

The app only enables Clerk when `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` looks like a real key.
Implementation detail: we gate on a conservative regex (`pk_test_...` / `pk_live_...`) in `src/auth/clerkKey.ts`.

#### Env vars

- `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`
  - If **unset/blank/placeholder**, Clerk is treated as **disabled**.
- `CLERK_SECRET_KEY`
  - Required only if you enable Clerk features that need server-side verification.
- Redirect URLs (optional; used by Clerk UI flows):
  - `NEXT_PUBLIC_CLERK_SIGN_IN_FORCE_REDIRECT_URL`
  - `NEXT_PUBLIC_CLERK_SIGN_UP_FORCE_REDIRECT_URL`
  - `NEXT_PUBLIC_CLERK_SIGN_IN_FALLBACK_REDIRECT_URL`
  - `NEXT_PUBLIC_CLERK_SIGN_UP_FALLBACK_REDIRECT_URL`

**Important:** `frontend/.env.example` contains placeholder values like `YOUR_PUBLISHABLE_KEY`.
Those placeholders are *not* valid keys and are intentionally treated as “Clerk disabled”.

## How the frontend talks to the backend

### API base URL

The client builds URLs using `NEXT_PUBLIC_API_URL` (normalized to remove trailing slashes).

### Generated API client (Orval + React Query)

We generate a typed client from the backend OpenAPI schema using **Orval**:

- Config: `orval.config.ts`
- Output: `src/api/generated/*`
- Script: `npm run api:gen`

By default, Orval reads:

- `ORVAL_INPUT` (if set), otherwise
- `http://127.0.0.1:8000/openapi.json`

Example:

```bash
# from frontend/
ORVAL_INPUT=http://localhost:8000/openapi.json npm run api:gen
```

### Auth header / Clerk token injection

All Orval-generated requests go through the custom mutator (`src/api/mutator.ts`).
It will:

- set `Content-Type: application/json` when there is a body and you didn’t specify a content type
- add `Authorization: Bearer <token>` automatically **if** Clerk is enabled and there is an active Clerk session in the browser
- parse errors into an `ApiError` with status + parsed response body

## Common commands

From `frontend/`:

```bash
npm run dev        # start dev server
npm run build      # production build
npm run start      # run the built app
npm run lint       # eslint
npm run test       # vitest (with coverage)
npm run test:watch # watch mode
npm run api:gen    # regenerate typed API client via Orval
```

## Docker

There is a `frontend/Dockerfile` used by the root `compose.yml`.

If you’re working on self-hosting, prefer running compose from the repo root so the backend/db/redis are aligned with the documented ports/env.

## Troubleshooting

### `NEXT_PUBLIC_API_URL is not set`

The API client throws if `NEXT_PUBLIC_API_URL` is missing.

Fix:

```bash
cp .env.example .env.local
# then edit .env.local if your backend URL differs
```

### Frontend loads, but API calls fail (CORS / network errors)

- Confirm backend is up: http://localhost:8000/healthz
- Confirm `NEXT_PUBLIC_API_URL` points to the correct host/port.
- If accessing from another device (LAN), use a reachable backend URL (not `localhost`).

### Clerk redirects / auth UI shows unexpectedly

Clerk should be **off** unless you set a real `pk_test_...` or `pk_live_...` publishable key.

- Remove/blank `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` in your `.env.local` to force Clerk off.

### Dev server blocked by origin restrictions

`next.config.ts` sets:

- `allowedDevOrigins: ["192.168.1.101"]`

If you’re developing from a different hostname/IP, you may need to update `allowedDevOrigins` (or use `npm run dev` on localhost).