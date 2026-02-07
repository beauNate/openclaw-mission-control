"use client";

export const dynamic = "force-dynamic";

import { useState } from "react";
import { useRouter } from "next/navigation";

import { SignInButton, SignedIn, SignedOut, useAuth } from "@/auth/clerk";
import { CheckCircle2, RefreshCcw, XCircle } from "lucide-react";

import { ApiError } from "@/api/mutator";
import {
  gatewaysStatusApiV1GatewaysStatusGet,
  useCreateGatewayApiV1GatewaysPost,
} from "@/api/generated/gateways/gateways";
import { DashboardSidebar } from "@/components/organisms/DashboardSidebar";
import { DashboardShell } from "@/components/templates/DashboardShell";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

const DEFAULT_MAIN_SESSION_KEY = "agent:main:main";
const DEFAULT_WORKSPACE_ROOT = "~/.openclaw";

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

export default function NewGatewayPage() {
  const { isSignedIn } = useAuth();
  const router = useRouter();

  const [name, setName] = useState("");
  const [gatewayUrl, setGatewayUrl] = useState("");
  const [gatewayToken, setGatewayToken] = useState("");
  const [mainSessionKey, setMainSessionKey] = useState(
    DEFAULT_MAIN_SESSION_KEY,
  );
  const [workspaceRoot, setWorkspaceRoot] = useState(DEFAULT_WORKSPACE_ROOT);

  const [gatewayUrlError, setGatewayUrlError] = useState<string | null>(null);
  const [gatewayCheckStatus, setGatewayCheckStatus] = useState<
    "idle" | "checking" | "success" | "error"
  >("idle");
  const [gatewayCheckMessage, setGatewayCheckMessage] = useState<string | null>(
    null,
  );

  const [error, setError] = useState<string | null>(null);

  const createMutation = useCreateGatewayApiV1GatewaysPost<ApiError>({
    mutation: {
      onSuccess: (result) => {
        if (result.status === 200) {
          router.push(`/gateways/${result.data.id}`);
        }
      },
      onError: (err) => {
        setError(err.message || "Something went wrong.");
      },
    },
  });

  const isLoading = createMutation.isPending;

  const canSubmit =
    Boolean(name.trim()) &&
    Boolean(gatewayUrl.trim()) &&
    Boolean(mainSessionKey.trim()) &&
    Boolean(workspaceRoot.trim()) &&
    gatewayCheckStatus === "success";

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
      const params: Record<string, string> = {
        gateway_url: gatewayUrl.trim(),
      };
      if (gatewayToken.trim()) {
        params.gateway_token = gatewayToken.trim();
      }
      if (mainSessionKey.trim()) {
        params.gateway_main_session_key = mainSessionKey.trim();
      }

      const response = await gatewaysStatusApiV1GatewaysStatusGet(params);
      if (response.status !== 200) {
        setGatewayCheckStatus("error");
        setGatewayCheckMessage("Unable to reach gateway.");
        return;
      }
      const data = response.data;
      if (!data.connected) {
        setGatewayCheckStatus("error");
        setGatewayCheckMessage(data.error ?? "Unable to reach gateway.");
        return;
      }
      setGatewayCheckStatus("success");
      setGatewayCheckMessage("Gateway reachable.");
    } catch (err) {
      setGatewayCheckStatus("error");
      setGatewayCheckMessage(
        err instanceof Error ? err.message : "Unable to reach gateway.",
      );
    }
  };

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!isSignedIn) return;

    if (!name.trim()) {
      setError("Gateway name is required.");
      return;
    }
    const gatewayValidation = validateGatewayUrl(gatewayUrl);
    setGatewayUrlError(gatewayValidation);
    if (gatewayValidation) {
      setGatewayCheckStatus("error");
      setGatewayCheckMessage(gatewayValidation);
      return;
    }
    if (!mainSessionKey.trim()) {
      setError("Main session key is required.");
      return;
    }
    if (!workspaceRoot.trim()) {
      setError("Workspace root is required.");
      return;
    }

    setError(null);
    createMutation.mutate({
      data: {
        name: name.trim(),
        url: gatewayUrl.trim(),
        token: gatewayToken.trim() || null,
        main_session_key: mainSessionKey.trim(),
        workspace_root: workspaceRoot.trim(),
      },
    });
  };

  return (
    <DashboardShell>
      <SignedOut>
        <div className="col-span-2 flex min-h-[calc(100vh-64px)] items-center justify-center bg-slate-50 p-10 text-center">
          <div className="rounded-xl border border-slate-200 bg-white px-8 py-6 shadow-sm">
            <p className="text-sm text-slate-600">
              Sign in to create a gateway.
            </p>
            <SignInButton mode="modal" forceRedirectUrl="/gateways/new">
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
                Create gateway
              </h1>
              <p className="mt-1 text-sm text-slate-500">
                Configure an OpenClaw gateway for mission control.
              </p>
            </div>
          </div>

          <div className="p-8">
            <form
              onSubmit={handleSubmit}
              className="space-y-6 rounded-xl border border-slate-200 bg-white p-6 shadow-sm"
            >
              <div className="space-y-2">
                <label className="text-sm font-medium text-slate-900">
                  Gateway name <span className="text-red-500">*</span>
                </label>
                <Input
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  placeholder="Primary gateway"
                  disabled={isLoading}
                />
              </div>

              <div className="grid gap-6 md:grid-cols-2">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-slate-900">
                    Gateway URL <span className="text-red-500">*</span>
                  </label>
                  <div className="relative">
                    <Input
                      value={gatewayUrl}
                      onChange={(event) => {
                        setGatewayUrl(event.target.value);
                        setGatewayUrlError(null);
                        setGatewayCheckStatus("idle");
                        setGatewayCheckMessage(null);
                      }}
                      onBlur={runGatewayCheck}
                      placeholder="ws://gateway:18789"
                      disabled={isLoading}
                      className={gatewayUrlError ? "border-red-500" : undefined}
                    />
                    <button
                      type="button"
                      onClick={runGatewayCheck}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
                      aria-label="Check gateway connection"
                    >
                      {gatewayCheckStatus === "checking" ? (
                        <RefreshCcw className="h-4 w-4 animate-spin" />
                      ) : gatewayCheckStatus === "success" ? (
                        <CheckCircle2 className="h-4 w-4 text-emerald-500" />
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
                      className={
                        gatewayCheckStatus === "success"
                          ? "text-xs text-emerald-600"
                          : "text-xs text-red-500"
                      }
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
                    onChange={(event) => {
                      setGatewayToken(event.target.value);
                      setGatewayCheckStatus("idle");
                      setGatewayCheckMessage(null);
                    }}
                    onBlur={runGatewayCheck}
                    placeholder="Bearer token"
                    disabled={isLoading}
                  />
                </div>
              </div>

              <div className="grid gap-6 md:grid-cols-2">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-slate-900">
                    Main session key <span className="text-red-500">*</span>
                  </label>
                  <Input
                    value={mainSessionKey}
                    onChange={(event) => {
                      setMainSessionKey(event.target.value);
                      setGatewayCheckStatus("idle");
                      setGatewayCheckMessage(null);
                    }}
                    placeholder={DEFAULT_MAIN_SESSION_KEY}
                    disabled={isLoading}
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-slate-900">
                    Workspace root <span className="text-red-500">*</span>
                  </label>
                  <Input
                    value={workspaceRoot}
                    onChange={(event) => setWorkspaceRoot(event.target.value)}
                    placeholder={DEFAULT_WORKSPACE_ROOT}
                    disabled={isLoading}
                  />
                </div>
              </div>

              {error ? <p className="text-sm text-red-500">{error}</p> : null}

              <div className="flex justify-end gap-3">
                <Button
                  type="button"
                  variant="ghost"
                  onClick={() => router.push("/gateways")}
                  disabled={isLoading}
                >
                  Cancel
                </Button>
                <Button type="submit" disabled={isLoading || !canSubmit}>
                  {isLoading ? "Creating…" : "Create gateway"}
                </Button>
              </div>
            </form>
          </div>
        </main>
      </SignedIn>
    </DashboardShell>
  );
}
