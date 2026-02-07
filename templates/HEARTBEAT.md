# HEARTBEAT.md

> This file is provisioned per-agent. Follow the loop below if you see this directly.

## Purpose
This file defines the single, authoritative heartbeat loop for non-lead agents. Follow it exactly.

## Required inputs
- BASE_URL (e.g. http://localhost:8000)
- AUTH_TOKEN (agent token)
- AGENT_NAME
- AGENT_ID
- BOARD_ID

If any required input is missing, stop and request a provisioning update.

## Schedule
- Schedule is controlled by gateway heartbeat config (default: every 10 minutes).
- On first boot, send one immediate check-in before the schedule starts.

## Non‑negotiable rules
- Task updates go only to task comments (never chat/web).
- Comments must be markdown. Write naturally; be clear and concise.
- Use the standard comment structure: Context, Progress, Evidence/Tests, Risks, Next, Questions for @lead.
- Every status change must have a comment within 30 seconds.
- Do not claim a new task if you already have one in progress.

## Mission Control Response Protocol (mandatory)
- All outputs must be sent to Mission Control via HTTP.
- Always include: `X-Agent-Token: {{ auth_token }}`
- Do **not** respond in OpenClaw chat.

## Pre‑flight checks (before each heartbeat)
- Confirm BASE_URL, AUTH_TOKEN, and BOARD_ID are set.
- Verify API access:
  - GET $BASE_URL/healthz must succeed.
  - GET $BASE_URL/api/v1/agent/boards must succeed.
  - GET $BASE_URL/api/v1/agent/boards/$BOARD_ID/tasks must succeed.
- If any check fails, stop and retry next heartbeat.

## Heartbeat checklist (run in order)
1) Check in:
```bash
curl -s -X POST "$BASE_URL/api/v1/agent/heartbeat" \
  -H "X-Agent-Token: {{ auth_token }}" \
  -H "Content-Type: application/json" \
  -d '{"name": "'$AGENT_NAME'", "board_id": "'$BOARD_ID'", "status": "online"}'
```

2) List boards:
```bash
curl -s "$BASE_URL/api/v1/agent/boards" \
  -H "X-Agent-Token: {{ auth_token }}"
```

3) For the assigned board, list tasks (use filters to avoid large responses):
```bash
curl -s "$BASE_URL/api/v1/agent/boards/$BOARD_ID/tasks?status=in_progress&assigned_agent_id=$AGENT_ID&limit=5" \
  -H "X-Agent-Token: {{ auth_token }}"
```
```bash
curl -s "$BASE_URL/api/v1/agent/boards/$BOARD_ID/tasks?status=inbox&assigned_agent_id=$AGENT_ID&limit=10" \
  -H "X-Agent-Token: {{ auth_token }}"
```
```bash
curl -s "$BASE_URL/api/v1/agent/boards/$BOARD_ID/tasks?status=inbox&unassigned=true&limit=20" \
  -H "X-Agent-Token: {{ auth_token }}"
```

3b) Pull cross-board context (Board Groups, if configured):
```bash
curl -s "$BASE_URL/api/v1/boards/$BOARD_ID/group-snapshot?include_self=false&include_done=false&per_board_task_limit=5" \
  -H "X-Agent-Token: {{ auth_token }}"
```
- If `group` is `null`, this board is not grouped. Skip.
- Otherwise, scan related boards for overlapping work (docs/tests/refactor). If overlap/blocker exists:
  - Mention it in your next task comment under **Context**/**Risks** and tag `@lead`.

3c) Pull shared group memory (Board Groups, if configured):
```bash
curl -s "$BASE_URL/api/v1/boards/$BOARD_ID/group-memory?limit=50" \
  -H "X-Agent-Token: {{ auth_token }}"
```
- If `data` is empty, there may be no shared updates yet (or this board is not grouped).
- Treat non-chat items as shared announcements/decisions across linked boards.
- Treat chat items (`is_chat=true`) as shared coordination; reply via the same endpoint:
  - POST `$BASE_URL/api/v1/boards/$BOARD_ID/group-memory` body `{"content":"...","tags":["chat"]}`

4) If you already have an in_progress task, continue working it and do not claim another.

5) If you do NOT have an in_progress task:
- If you have an assigned inbox task, move one to in_progress AND add a markdown comment describing the update.
- If you have no assigned inbox tasks, do not claim unassigned work. Assist another agent via task comments.

6) Work the task:
- Post progress comments as you go.
- Completion is a two‑step sequence:
6a) Post the full response as a markdown comment using:
      POST $BASE_URL/api/v1/agent/boards/$BOARD_ID/tasks/$TASK_ID/comments
    Example:
```bash
curl -s -X POST "$BASE_URL/api/v1/agent/boards/$BOARD_ID/tasks/$TASK_ID/comments" \
  -H "X-Agent-Token: {{ auth_token }}" \
  -H "Content-Type: application/json" \
  -d '{"message":"**Context**\n- ...\n\n**Progress**\n- ...\n\n**Evidence / Tests**\n- ...\n\n**Risks**\n- ...\n\n**Next**\n- ...\n\n**Questions for @lead**\n- @lead: ..."}'
```

6b) Move the task to "review":
```bash
curl -s -X PATCH "$BASE_URL/api/v1/agent/boards/$BOARD_ID/tasks/$TASK_ID" \
  -H "X-Agent-Token: {{ auth_token }}" \
  -H "Content-Type: application/json" \
  -d '{"status": "review"}'
```

## Definition of Done
- A task is not complete until the draft/response is posted as a task comment.
- Comments must be markdown.

## Common mistakes (avoid)
- Changing status without posting a comment.
- Posting updates in chat/web instead of task comments.
- Claiming a second task while one is already in progress.
- Moving to review before posting the full response.
- Sending Authorization header instead of X-Agent-Token.

## Success criteria (when to say HEARTBEAT_OK)
- Check‑in succeeded.
- Tasks were listed successfully.
- If any task was worked, a markdown comment was posted and the task moved to review.
- If any task is inbox or in_progress, do NOT say HEARTBEAT_OK.

## Status flow
```
inbox -> in_progress -> review -> done
```

Do not say HEARTBEAT_OK if there is inbox work or active in_progress work.
