"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";

import { SignInButton, SignedIn, SignedOut, useAuth } from "@clerk/nextjs";

import { DashboardSidebar } from "@/components/organisms/DashboardSidebar";
import { DashboardShell } from "@/components/templates/DashboardShell";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { getApiBaseUrl } from "@/lib/api-base";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const apiBase = getApiBaseUrl();

type Agent = {
  id: string;
  name: string;
  board_id?: string | null;
  heartbeat_config?: {
    every?: string;
    target?: string;
  } | null;
};

type Board = {
  id: string;
  name: string;
  slug: string;
};

export default function EditAgentPage() {
  const { getToken, isSignedIn } = useAuth();
  const router = useRouter();
  const params = useParams();
  const agentIdParam = params?.agentId;
  const agentId = Array.isArray(agentIdParam) ? agentIdParam[0] : agentIdParam;

  const [agent, setAgent] = useState<Agent | null>(null);
  const [name, setName] = useState("");
  const [boards, setBoards] = useState<Board[]>([]);
  const [boardId, setBoardId] = useState("");
  const [heartbeatEvery, setHeartbeatEvery] = useState("10m");
  const [heartbeatTarget, setHeartbeatTarget] = useState("none");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadBoards = async () => {
    if (!isSignedIn) return;
    try {
      const token = await getToken();
      const response = await fetch(`${apiBase}/api/v1/boards`, {
        headers: { Authorization: token ? `Bearer ${token}` : "" },
      });
      if (!response.ok) {
        throw new Error("Unable to load boards.");
      }
      const data = (await response.json()) as Board[];
      setBoards(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    }
  };

  const loadAgent = async () => {
    if (!isSignedIn || !agentId) return;
    setIsLoading(true);
    setError(null);
    try {
      const token = await getToken();
      const response = await fetch(`${apiBase}/api/v1/agents/${agentId}`, {
        headers: { Authorization: token ? `Bearer ${token}` : "" },
      });
      if (!response.ok) {
        throw new Error("Unable to load agent.");
      }
      const data = (await response.json()) as Agent;
      setAgent(data);
      setName(data.name);
      if (data.board_id) {
        setBoardId(data.board_id);
      }
      if (data.heartbeat_config?.every) {
        setHeartbeatEvery(data.heartbeat_config.every);
      }
      if (data.heartbeat_config?.target) {
        setHeartbeatTarget(data.heartbeat_config.target);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadBoards();
    loadAgent();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isSignedIn, agentId]);

  useEffect(() => {
    if (boardId) return;
    if (agent?.board_id) {
      setBoardId(agent.board_id);
      return;
    }
    if (boards.length > 0) {
      setBoardId(boards[0].id);
    }
  }, [agent, boards, boardId]);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!isSignedIn || !agentId) return;
    const trimmed = name.trim();
    if (!trimmed) {
      setError("Agent name is required.");
      return;
    }
    if (!boardId) {
      setError("Select a board before saving.");
      return;
    }
    setIsLoading(true);
    setError(null);
    try {
      const token = await getToken();
      const response = await fetch(`${apiBase}/api/v1/agents/${agentId}`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
          Authorization: token ? `Bearer ${token}` : "",
        },
        body: JSON.stringify({
          name: trimmed,
          board_id: boardId,
          heartbeat_config: {
            every: heartbeatEvery.trim() || "10m",
            target: heartbeatTarget,
          },
        }),
      });
      if (!response.ok) {
        throw new Error("Unable to update agent.");
      }
      router.push(`/agents/${agentId}`);
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
            <p className="text-sm text-slate-600">Sign in to edit agents.</p>
            <SignInButton
              mode="modal"
              forceRedirectUrl={`/agents/${agentId}/edit`}
              signUpForceRedirectUrl={`/agents/${agentId}/edit`}
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
                {agent?.name ?? "Edit agent"}
              </h1>
              <p className="mt-1 text-sm text-slate-500">
                Status is controlled by agent heartbeat.
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
                  Agent identity
                </p>
                <div className="mt-4 grid gap-6 md:grid-cols-2">
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-slate-900">
                      Agent name
                    </label>
                    <Input
                      value={name}
                      onChange={(event) => setName(event.target.value)}
                      placeholder="e.g. Deploy bot"
                      disabled={isLoading}
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-slate-900">
                      Board
                    </label>
                    <Select
                      value={boardId}
                      onValueChange={(value) => setBoardId(value)}
                      disabled={boards.length === 0}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select board" />
                      </SelectTrigger>
                      <SelectContent>
                        {boards.map((board) => (
                          <SelectItem key={board.id} value={board.id}>
                            {board.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    {boards.length === 0 ? (
                      <p className="text-xs text-slate-500">
                        Create a board before assigning agents.
                      </p>
                    ) : null}
                  </div>
                </div>
              </div>

              <div>
                <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Heartbeat settings
                </p>
                <div className="mt-4 grid gap-6 md:grid-cols-2">
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-slate-900">
                      Interval
                    </label>
                    <Input
                      value={heartbeatEvery}
                      onChange={(event) => setHeartbeatEvery(event.target.value)}
                      placeholder="e.g. 10m"
                      disabled={isLoading}
                    />
                    <p className="text-xs text-slate-500">
                      Set how often this agent runs HEARTBEAT.md.
                    </p>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-slate-900">
                      Target
                    </label>
                    <Select
                      value={heartbeatTarget}
                      onValueChange={(value) => setHeartbeatTarget(value)}
                      disabled={isLoading}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select target" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="none">
                          None (no outbound message)
                        </SelectItem>
                        <SelectItem value="last">Last channel</SelectItem>
                      </SelectContent>
                    </Select>
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
                  {isLoading ? "Saving…" : "Save changes"}
                </Button>
                <Button
                  variant="outline"
                  type="button"
                  onClick={() => router.push(`/agents/${agentId}`)}
                >
                  Back to agent
                </Button>
              </div>
            </form>
          </div>
        </main>
      </SignedIn>
    </DashboardShell>
  );
}
