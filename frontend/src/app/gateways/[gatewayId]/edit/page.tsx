"use client";

export const dynamic = "force-dynamic";

import { useState } from "react";
import { useParams, useRouter } from "next/navigation";

import { SignInButton, SignedIn, SignedOut, useAuth } from "@/auth/clerk";
import { CheckCircle2, RefreshCcw, XCircle } from "lucide-react";

import { ApiError } from "@/api/mutator";
import {
  gatewaysStatusApiV1GatewaysStatusGet,
  type getGatewayApiV1GatewaysGatewayIdGetResponse,
  useGetGatewayApiV1GatewaysGatewayIdGet,
  useUpdateGatewayApiV1GatewaysGatewayIdPatch,
} from "@/api/generated/gateways/gateways";
import type { GatewayUpdate } from "@/api/generated/model";
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

export default function EditGatewayPage() {
  const { isSignedIn } = useAuth();
  const router = useRouter();
  const params = useParams();
  const gatewayIdParam = params?.gatewayId;
  const gatewayId = Array.isArray(gatewayIdParam)
    ? gatewayIdParam[0]
    : gatewayIdParam;

  const [name, setName] = useState<string | undefined>(undefined);
  const [gatewayUrl, setGatewayUrl] = useState<string | undefined>(undefined);
  const [gatewayToken, setGatewayToken] = useState<string | undefined>(
    undefined,
  );
  const [mainSessionKey, setMainSessionKey] = useState<string | undefined>(
    undefined,
  );
  const [workspaceRoot, setWorkspaceRoot] = useState<string | undefined>(
    undefined,
  );

  const [gatewayUrlError, setGatewayUrlError] = useState<string | null>(null);
  const [gatewayCheckStatus, setGatewayCheckStatus] = useState<
    "idle" | "checking" | "success" | "error"
  >("idle");
  const [gatewayCheckMessage, setGatewayCheckMessage] = useState<string | null>(
    null,
  );

  const [error, setError] = useState<string | null>(null);

  const gatewayQuery = useGetGatewayApiV1GatewaysGatewayIdGet<
    getGatewayApiV1GatewaysGatewayIdGetResponse,
    ApiError
  >(gatewayId ?? "", {
    query: {
      enabled: Boolean(isSignedIn && gatewayId),
      refetchOnMount: "always",
      retry: false,
    },
  });

  const updateMutation = useUpdateGatewayApiV1GatewaysGatewayIdPatch<ApiError>({
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

  const loadedGateway =
    gatewayQuery.data?.status === 200 ? gatewayQuery.data.data : null;
  const resolvedName = name ?? loadedGateway?.name ?? "";
  const resolvedGatewayUrl = gatewayUrl ?? loadedGateway?.url ?? "";
  const resolvedGatewayToken = gatewayToken ?? loadedGateway?.token ?? "";
  const resolvedMainSessionKey =
    mainSessionKey ??
    loadedGateway?.main_session_key ??
    DEFAULT_MAIN_SESSION_KEY;
  const resolvedWorkspaceRoot =
    workspaceRoot ?? loadedGateway?.workspace_root ?? DEFAULT_WORKSPACE_ROOT;

  const isLoading = gatewayQuery.isLoading || updateMutation.isPending;
  const errorMessage = error ?? gatewayQuery.error?.message ?? null;

  const canSubmit =
    Boolean(resolvedName.trim()) &&
    Boolean(resolvedGatewayUrl.trim()) &&
    Boolean(resolvedMainSessionKey.trim()) &&
    Boolean(resolvedWorkspaceRoot.trim()) &&
    gatewayCheckStatus === "success";

  const runGatewayCheck = async () => {
    const validationError = validateGatewayUrl(resolvedGatewayUrl);
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
        gateway_url: resolvedGatewayUrl.trim(),
      };
      if (resolvedGatewayToken.trim()) {
        params.gateway_token = resolvedGatewayToken.trim();
      }
      if (resolvedMainSessionKey.trim()) {
        params.gateway_main_session_key = resolvedMainSessionKey.trim();
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
    if (!isSignedIn || !gatewayId) return;

    if (!resolvedName.trim()) {
      setError("Gateway name is required.");
      return;
    }
    const gatewayValidation = validateGatewayUrl(resolvedGatewayUrl);
    setGatewayUrlError(gatewayValidation);
    if (gatewayValidation) {
      setGatewayCheckStatus("error");
      setGatewayCheckMessage(gatewayValidation);
      return;
    }
    if (!resolvedMainSessionKey.trim()) {
      setError("Main session key is required.");
      return;
    }
    if (!resolvedWorkspaceRoot.trim()) {
      setError("Workspace root is required.");
      return;
    }

    setError(null);

    const payload: GatewayUpdate = {
      name: resolvedName.trim(),
      url: resolvedGatewayUrl.trim(),
      token: resolvedGatewayToken.trim() || null,
      main_session_key: resolvedMainSessionKey.trim(),
      workspace_root: resolvedWorkspaceRoot.trim(),
    };

    updateMutation.mutate({ gatewayId, data: payload });
  };

  return (
    <DashboardShell>
      <SignedOut>
        <div className="col-span-2 flex min-h-[calc(100vh-64px)] items-center justify-center bg-slate-50 p-10 text-center">
          <div className="rounded-xl border border-slate-200 bg-white px-8 py-6 shadow-sm">
            <p className="text-sm text-slate-600">Sign in to edit a gateway.</p>
            <SignInButton
              mode="modal"
              forceRedirectUrl={`/gateways/${gatewayId}/edit`}
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
                {resolvedName.trim()
                  ? `Edit gateway — ${resolvedName.trim()}`
                  : "Edit gateway"}
              </h1>
              <p className="mt-1 text-sm text-slate-500">
                Update connection settings for this OpenClaw gateway.
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
                  value={resolvedName}
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
                      value={resolvedGatewayUrl}
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
                    value={resolvedGatewayToken}
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
                    value={resolvedMainSessionKey}
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
                    value={resolvedWorkspaceRoot}
                    onChange={(event) => setWorkspaceRoot(event.target.value)}
                    placeholder={DEFAULT_WORKSPACE_ROOT}
                    disabled={isLoading}
                  />
                </div>
              </div>

              {errorMessage ? (
                <p className="text-sm text-red-500">{errorMessage}</p>
              ) : null}

              <div className="flex justify-end gap-3">
                <Button
                  type="button"
                  variant="ghost"
                  onClick={() => router.push("/gateways")}
                  disabled={isLoading}
                >
                  Back
                </Button>
                <Button type="submit" disabled={isLoading || !canSubmit}>
                  {isLoading ? "Saving…" : "Save changes"}
                </Button>
              </div>
            </form>
          </div>
        </main>
      </SignedIn>
    </DashboardShell>
  );
}
