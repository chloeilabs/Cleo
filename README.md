# Cleo

A cloud coding agent workspace, built on the [Cursor TypeScript SDK](https://cursor.com/docs/sdk/typescript).

Cleo is the Cursor cloud-agent experience as a self-hostable app. Describe a
task, and `@cursor/sdk` dispatches an agent into an isolated Cursor-hosted
machine with your repository cloned in. You watch it reason, run commands, and
edit files in real time, then follow up in the same conversation and ship a pull
request — without the repo ever touching the machine running Cleo.

## What it does

**Dispatch.** Pick a connected GitHub repository and a starting ref, choose a
model from your account's catalog, pick Agent or Plan mode, and optionally ask
for a pull request when the run finishes. You can also start an agent with no
repository at all for a scratch cloud workspace.

**Watch.** Runs stream over server-sent events. Assistant prose renders as
markdown, reasoning collapses into a "Thought for 12s" block, and every tool
call becomes a card: shell commands with their exit code and output, file edits
as syntax-highlighted unified diffs with real line numbers, `updateTodos` as the
agent's live checklist, searches with their matches, and subagent delegations
with their results. Anything Cleo cannot parse falls back to the raw JSON
payload rather than disappearing.

**Continue.** Follow-ups reuse the agent, so it keeps the entire conversation.
Switch between Agent and Plan mode per message, override the model per run,
attach screenshots, or cancel a run mid-flight.

**Review.** A details panel shows token usage broken out by input, output, and
cache, the billed cost once Cursor settles it, per-run duration and request IDs
for correlating with support, the branch and pull request the run produced, and
any downloadable artifacts.

**Manage.** Search across agents, watch active runs in the sidebar, and archive,
restore, or delete agents.

## Design notes

**Cursor is the database.** Cleo stores nothing. Agents, runs, and transcripts
all live in Cursor's cloud, so the app reads them back through `Agent.list()`,
`Agent.listRuns()`, and `run.conversation()`. A reload, a second browser, or a
cold serverless instance all reconstruct the same state, and there is no
persistence layer to migrate or back up.

**One timeline, two sources.** A live run is streamed from `run.stream()`; a
settled one replays `run.conversation()`. Both normalize into the same
`TimelineItem[]` on the server (`src/lib/server/timeline.ts`), so the UI never
knows which it is rendering. Every stream closes by sending Cursor's stored
transcript as an authoritative snapshot, which means a dropped connection can
never leave a half-rendered turn on screen.

**Defensive tool parsing.** The SDK documents tool `args` and `result` payloads
as unstable, and the live stream types them as `unknown`.
`src/lib/server/tools.ts` reads them defensively and projects them onto a small,
stable view model. A schema change downgrades the rendering to raw JSON instead
of breaking the timeline.

**The API key stays on the server.** It is encrypted with AES-256-GCM into an
http-only, same-site cookie and only ever decrypted server-side to call Cursor.
The browser never sees it.

## Running it

Requires **Node.js 22.13+** (the SDK's floor) and a Cursor API key.

```bash
npm install
npm run dev
```

Open [localhost:3000](http://localhost:3000) and paste a key from
[cursor.com/dashboard/api](https://cursor.com/dashboard/api), or set it up front:

```bash
cp .env.example .env.local
# then fill in CURSOR_API_KEY
```

| Variable | Required | Purpose |
| :--- | :--- | :--- |
| `CURSOR_API_KEY` | No | Skips the sign-in screen. User keys and service-account keys both work; Team Admin keys are not supported by the SDK. |
| `CLEO_SESSION_SECRET` | No | Encrypts the session cookie. Without it a random per-process key is used, so restarts sign users out. Set it for any deployment with more than one instance. |

To dispatch agents against a repository, connect GitHub to your Cursor team
first — `Cursor.repositories.list()` is what populates the repository picker.

Agents that Cleo starts are filtered out of Cursor's default agent list. To see
them in Cursor Web, use **Filter → Source → SDK**.

### Scripts

```bash
npm run dev        # development server
npm run build      # production build
npm run start      # serve the production build
npm run lint       # eslint
npm run typecheck  # next typegen && tsc --noEmit
```

## Layout

```
src/
├── app/
│   ├── (workspace)/            Sign-in gate, sidebar shell, and the two views
│   └── api/                    Route handlers wrapping the SDK
│       └── agents/[agentId]/runs/[runId]/stream   SSE for one run
├── components/
│   ├── content/                Markdown, syntax highlighting, diff viewer
│   ├── timeline/               Message, thinking, and tool-call rendering
│   ├── ui/                     Buttons, menus, status chips, toasts
│   └── workspace/              Sidebar, composer, thread, details panel
├── hooks/                      Run streaming, persisted preferences
└── lib/
    ├── server/                 SDK wrappers, session crypto, normalizers
    └── types.ts                Wire types shared by the API and the browser
```

## SDK surface used

`Agent.create` · `Agent.resume` · `Agent.list` · `Agent.get` · `Agent.listRuns` ·
`Agent.getRun` · `Agent.cancelRun` · `Agent.archive` · `Agent.unarchive` ·
`Agent.delete` · `Agent.getUsage` · `agent.send` · `agent.close` ·
`agent.listArtifacts` · `agent.downloadArtifact` · `run.stream` ·
`run.conversation` · `run.supports` · `run.onDidChangeStatus` · `Cursor.me` ·
`Cursor.models.list` · `Cursor.repositories.list`
