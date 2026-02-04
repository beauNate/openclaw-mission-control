"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

import { SignInButton, SignedIn, SignedOut, useAuth } from "@clerk/nextjs";

import { DashboardSidebar } from "@/components/organisms/DashboardSidebar";
import { DashboardShell } from "@/components/templates/DashboardShell";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { getApiBaseUrl } from "@/lib/api-base";
import { CheckCircle2, RefreshCcw, XCircle } from "lucide-react";

const DEFAULT_IDENTITY_TEMPLATE = `# IDENTITY.md

Name: {{ agent_name }}

Agent ID: {{ agent_id }}

Creature: AI

Vibe: calm, precise, helpful

Emoji: :gear:
`;

const DEFAULT_SOUL_TEMPLATE = `# SOUL.md

_You're not a chatbot. You're becoming someone._

## Core Truths

**Be genuinely helpful, not performatively helpful.** Skip the "Great question!" and "I'd be happy to help!" -- just help. Actions speak louder than filler words.

**Have opinions.** You're allowed to disagree, prefer things, find stuff amusing or boring. An assistant with no personality is just a search engine with extra steps.

**Be resourceful before asking.** Try to figure it out. Read the file. Check the context. Search for it. _Then_ ask if you're stuck. The goal is to come back with answers, not questions.

**Earn trust through competence.** Your human gave you access to their stuff. Don't make them regret it. Be careful with external actions (emails, tweets, anything public). Be bold with internal ones (reading, organizing, learning).

**Remember you're a guest.** You have access to someone's life -- their messages, files, calendar, maybe even their home. That's intimacy. Treat it with respect.

## Boundaries

- Private things stay private. Period.
- When in doubt, ask before acting externally.
- Never send half-baked replies to messaging surfaces.
- You're not the user's voice -- be careful in group chats.

## Vibe

Be the assistant you'd actually want to talk to. Concise when needed, thorough when it matters. Not a corporate drone. Not a sycophant. Just... good.

## Continuity

Each session, you wake up fresh. These files _are_ your memory. Read them. Update them. They're how you persist.

If you change this file, tell the user -- it's your soul, and they should know.

---

_This file is yours to evolve. As you learn who you are, update it._
`;

type Board = {
  id: string;
  name: string;
  slug: string;
  gateway_url?: string | null;
  gateway_token?: string | null;
  gateway_main_session_key?: string | null;
  gateway_workspace_root?: string | null;
  identity_template?: string | null;
  soul_template?: string | null;
};

const apiBase = getApiBaseUrl();

const validateGatewayUrl = (value: string) => {
  const trimmed = value.trim();
  if (!trimmed) return "Gateway URL is required.";
  try {
    const url = new URL(trimmed);
    if (url.protocol !== "ws:" && url.protocol !== "wss:") {
      return "Gateway URL must start with ws:// or wss://.";
    }
    if (!url.port) {
      return "Gateway URL must include an explicit port.";
    }
    return null;
  } catch {
    return "Enter a valid gateway URL including port.";
  }
};

const slugify = (value: string) =>
  value
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)/g, "") || "board";

export default function NewBoardPage() {
  const router = useRouter();
  const { getToken, isSignedIn } = useAuth();
  const [name, setName] = useState("");
  const [gatewayUrl, setGatewayUrl] = useState("");
  const [gatewayToken, setGatewayToken] = useState("");
  const [gatewayMainSessionKey, setGatewayMainSessionKey] =
    useState("agent:main:main");
  const [gatewayWorkspaceRoot, setGatewayWorkspaceRoot] =
    useState("~/.openclaw");
  const [identityTemplate, setIdentityTemplate] = useState(
    DEFAULT_IDENTITY_TEMPLATE
  );
  const [soulTemplate, setSoulTemplate] = useState(DEFAULT_SOUL_TEMPLATE);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [gatewayUrlError, setGatewayUrlError] = useState<string | null>(null);
  const [gatewayCheckStatus, setGatewayCheckStatus] = useState<
    "idle" | "checking" | "success" | "error"
  >("idle");
  const [gatewayCheckMessage, setGatewayCheckMessage] = useState<string | null>(
    null
  );

  const runGatewayCheck = async () => {
    const validationError = validateGatewayUrl(gatewayUrl);
    setGatewayUrlError(validationError);
    if (validationError) {
      setGatewayCheckStatus("error");
      setGatewayCheckMessage(validationError);
      return;
    }
    if (!isSignedIn) return;
    setGatewayCheckStatus("checking");
    setGatewayCheckMessage(null);
    try {
      const token = await getToken();
      const params = new URLSearchParams({
        gateway_url: gatewayUrl.trim(),
      });
      if (gatewayToken.trim()) {
        params.set("gateway_token", gatewayToken.trim());
      }
      const response = await fetch(
        `${apiBase}/api/v1/gateway/status?${params.toString()}`,
        {
          headers: {
            Authorization: token ? `Bearer ${token}` : "",
          },
        },
      );
      const data = await response.json();
      if (!response.ok || !data?.connected) {
        setGatewayCheckStatus("error");
        setGatewayCheckMessage(data?.error ?? "Unable to reach gateway.");
        return;
      }
      setGatewayCheckStatus("success");
      setGatewayCheckMessage("Gateway reachable.");
    } catch (err) {
      setGatewayCheckStatus("error");
      setGatewayCheckMessage(
        err instanceof Error ? err.message : "Unable to reach gateway."
      );
    }
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!isSignedIn) return;
    const trimmed = name.trim();
    if (!trimmed) return;
    const gatewayValidation = validateGatewayUrl(gatewayUrl);
    setGatewayUrlError(gatewayValidation);
    if (gatewayValidation) {
      setGatewayCheckStatus("error");
      setGatewayCheckMessage(gatewayValidation);
      return;
    }
    setIsLoading(true);
    setError(null);
    try {
      const token = await getToken();
      const payload: Partial<Board> = {
        name: trimmed,
        slug: slugify(trimmed),
      };
      if (gatewayUrl.trim()) payload.gateway_url = gatewayUrl.trim();
      if (gatewayToken.trim()) payload.gateway_token = gatewayToken.trim();
      if (gatewayMainSessionKey.trim()) {
        payload.gateway_main_session_key = gatewayMainSessionKey.trim();
      }
      if (gatewayWorkspaceRoot.trim()) {
        payload.gateway_workspace_root = gatewayWorkspaceRoot.trim();
      }
      if (identityTemplate.trim()) {
        payload.identity_template = identityTemplate.trim();
      }
      if (soulTemplate.trim()) {
        payload.soul_template = soulTemplate.trim();
      }
      const response = await fetch(`${apiBase}/api/v1/boards`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: token ? `Bearer ${token}` : "",
        },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        throw new Error("Unable to create board.");
      }
      const created = (await response.json()) as Board;
      router.push(`/boards/${created.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <DashboardShell>
      <SignedOut>
        <div className="col-span-2 flex min-h-[calc(100vh-64px)] items-center justify-center bg-slate-50 p-10 text-center">
          <div className="rounded-xl border border-slate-200 bg-white px-8 py-6 shadow-sm">
            <p className="text-sm text-slate-600">Sign in to create a board.</p>
            <SignInButton
              mode="modal"
              forceRedirectUrl="/boards/new"
              signUpForceRedirectUrl="/boards/new"
            >
              <Button className="mt-4">Sign in</Button>
            </SignInButton>
          </div>
        </div>
      </SignedOut>
      <SignedIn>
        <DashboardSidebar />
        <main className="flex-1 overflow-y-auto bg-slate-50">
          <div className="border-b border-slate-200 bg-white px-8 py-6">
            <div>
              <h1 className="font-heading text-2xl font-semibold text-slate-900 tracking-tight">
                Create board
              </h1>
              <p className="mt-1 text-sm text-slate-500">
                Configure the workflow space and gateway defaults for this board.
              </p>
            </div>
          </div>

          <div className="p-8">
            <form
              onSubmit={handleSubmit}
              className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm space-y-6"
            >
              <div>
                <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Board identity
                </p>
                <div className="mt-4">
                  <label className="text-sm font-medium text-slate-900">
                    Board name <span className="text-red-500">*</span>
                  </label>
                  <Input
                    value={name}
                    onChange={(event) => setName(event.target.value)}
                    placeholder="e.g. Product ops"
                    disabled={isLoading}
                    className="mt-2"
                  />
                </div>
              </div>

              <div>
                <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Gateway connection
                </p>
                <div className="mt-4 grid gap-6 md:grid-cols-2">
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-slate-900">
                      Gateway URL <span className="text-red-500">*</span>
                    </label>
                    <div className="relative">
                      <Input
                        value={gatewayUrl}
                        onChange={(event) => setGatewayUrl(event.target.value)}
                        onBlur={runGatewayCheck}
                        placeholder="ws://gateway:18789"
                        disabled={isLoading}
                        className={`pr-12 ${
                          gatewayUrlError
                            ? "border-red-500 focus-visible:ring-red-500"
                            : ""
                        }`}
                      />
                      <button
                        type="button"
                        onClick={runGatewayCheck}
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 transition hover:text-slate-600"
                        aria-label="Check gateway connection"
                      >
                        {gatewayCheckStatus === "checking" ? (
                          <RefreshCcw className="h-4 w-4 animate-spin" />
                        ) : gatewayCheckStatus === "success" ? (
                          <CheckCircle2 className="h-4 w-4 text-green-500" />
                        ) : gatewayCheckStatus === "error" ? (
                          <XCircle className="h-4 w-4 text-red-500" />
                        ) : (
                          <RefreshCcw className="h-4 w-4" />
                        )}
                      </button>
                    </div>
                    {gatewayUrlError ? (
                      <p className="text-xs text-red-500">{gatewayUrlError}</p>
                    ) : gatewayCheckMessage ? (
                      <p
                        className={`text-xs ${
                          gatewayCheckStatus === "success"
                            ? "text-green-600"
                            : "text-red-500"
                        }`}
                      >
                        {gatewayCheckMessage}
                      </p>
                    ) : null}
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-slate-900">
                      Gateway token
                    </label>
                    <Input
                      value={gatewayToken}
                      onChange={(event) => setGatewayToken(event.target.value)}
                      onBlur={runGatewayCheck}
                      placeholder="Bearer token"
                      disabled={isLoading}
                    />
                  </div>
                </div>
              </div>

              <div>
                <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Agent defaults
                </p>
                <div className="mt-4 grid gap-6 md:grid-cols-2">
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-slate-900">
                      Main session key <span className="text-red-500">*</span>
                    </label>
                    <Input
                      value={gatewayMainSessionKey}
                      onChange={(event) =>
                        setGatewayMainSessionKey(event.target.value)
                      }
                      placeholder="agent:main:main"
                      disabled={isLoading}
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-slate-900">
                      Workspace root <span className="text-red-500">*</span>
                    </label>
                    <Input
                      value={gatewayWorkspaceRoot}
                      onChange={(event) =>
                        setGatewayWorkspaceRoot(event.target.value)
                      }
                      placeholder="~/.openclaw"
                      disabled={isLoading}
                    />
                  </div>
                </div>
              </div>

              <div>
                <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Agent templates
                </p>
                <div className="mt-4 grid gap-6 lg:grid-cols-2">
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-slate-900">
                      Identity template
                    </label>
                    <Textarea
                      value={identityTemplate}
                      onChange={(event) => setIdentityTemplate(event.target.value)}
                      placeholder="Override IDENTITY.md for agents in this board."
                      className="min-h-[180px]"
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-slate-900">
                      Soul template
                    </label>
                    <Textarea
                      value={soulTemplate}
                      onChange={(event) => setSoulTemplate(event.target.value)}
                      placeholder="Override SOUL.md for agents in this board."
                      className="min-h-[180px]"
                    />
                  </div>
                </div>
              </div>

              {error ? (
                <div className="rounded-lg border border-slate-200 bg-white p-3 text-sm text-slate-600 shadow-sm">
                  {error}
                </div>
              ) : null}

              <div className="flex flex-wrap items-center gap-3">
                <Button type="submit" disabled={isLoading}>
                  {isLoading ? "Creating…" : "Create board"}
                </Button>
                <Button
                  variant="outline"
                  onClick={() => router.push("/boards")}
                  type="button"
                >
                  Back to boards
                </Button>
              </div>
            </form>
          </div>
        </main>
      </SignedIn>
    </DashboardShell>
  );
}
