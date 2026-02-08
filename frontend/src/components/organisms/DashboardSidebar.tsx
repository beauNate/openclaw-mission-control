"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  BarChart3,
  Bot,
  CheckCircle2,
  Folder,
  Building2,
  LayoutGrid,
  Network,
} from "lucide-react";

import { useAuth } from "@/auth/clerk";
import { ApiError } from "@/api/mutator";
import {
  type getMyMembershipApiV1OrganizationsMeMemberGetResponse,
  useGetMyMembershipApiV1OrganizationsMeMemberGet,
} from "@/api/generated/organizations/organizations";
import {
  type healthzHealthzGetResponse,
  useHealthzHealthzGet,
} from "@/api/generated/default/default";
import { cn } from "@/lib/utils";

export function DashboardSidebar() {
  const pathname = usePathname();
  const { isSignedIn } = useAuth();
  const membershipQuery = useGetMyMembershipApiV1OrganizationsMeMemberGet<
    getMyMembershipApiV1OrganizationsMeMemberGetResponse,
    ApiError
  >({
    query: {
      enabled: Boolean(isSignedIn),
      refetchOnMount: "always",
      retry: false,
    },
  });
  const member =
    membershipQuery.data?.status === 200 ? membershipQuery.data.data : null;
  const isAdmin = member ? ["owner", "admin"].includes(member.role) : false;
  const healthQuery = useHealthzHealthzGet<healthzHealthzGetResponse, ApiError>(
    {
      query: {
        refetchInterval: 30_000,
        refetchOnMount: "always",
        retry: false,
      },
      request: { cache: "no-store" },
    },
  );

  const okValue = healthQuery.data?.data?.ok;
  const systemStatus: "unknown" | "operational" | "degraded" =
    okValue === true
      ? "operational"
      : okValue === false
        ? "degraded"
        : healthQuery.isError
          ? "degraded"
          : "unknown";
  const statusLabel =
    systemStatus === "operational"
      ? "All systems operational"
      : systemStatus === "unknown"
        ? "System status unavailable"
        : "System degraded";

  return (
    <aside className="flex h-full w-64 flex-col border-r border-slate-200 bg-white">
      <div className="flex-1 px-3 py-4">
        <p className="px-3 text-xs font-semibold uppercase tracking-wider text-slate-500">
          Navigation
        </p>
        <nav className="mt-3 space-y-1 text-sm">
          <Link
            href="/dashboard"
            className={cn(
              "flex items-center gap-3 rounded-lg px-3 py-2.5 text-slate-700 transition",
              pathname === "/dashboard"
                ? "bg-blue-100 text-blue-800 font-medium"
                : "hover:bg-slate-100",
            )}
          >
            <BarChart3 className="h-4 w-4" />
            Dashboard
          </Link>
          {isAdmin ? (
            <Link
              href="/gateways"
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2.5 text-slate-700 transition",
                pathname.startsWith("/gateways")
                  ? "bg-blue-100 text-blue-800 font-medium"
                  : "hover:bg-slate-100",
              )}
            >
              <Network className="h-4 w-4" />
              Gateways
            </Link>
          ) : null}
          <Link
            href="/board-groups"
            className={cn(
              "flex items-center gap-3 rounded-lg px-3 py-2.5 text-slate-700 transition",
              pathname.startsWith("/board-groups")
                ? "bg-blue-100 text-blue-800 font-medium"
                : "hover:bg-slate-100",
            )}
          >
            <Folder className="h-4 w-4" />
            Board groups
          </Link>
          <Link
            href="/boards"
            className={cn(
              "flex items-center gap-3 rounded-lg px-3 py-2.5 text-slate-700 transition",
              pathname.startsWith("/boards")
                ? "bg-blue-100 text-blue-800 font-medium"
                : "hover:bg-slate-100",
            )}
          >
            <LayoutGrid className="h-4 w-4" />
            Boards
          </Link>
          <Link
            href="/organization"
            className={cn(
              "flex items-center gap-3 rounded-lg px-3 py-2.5 text-slate-700 transition",
              pathname.startsWith("/organization")
                ? "bg-blue-100 text-blue-800 font-medium"
                : "hover:bg-slate-100",
            )}
          >
            <Building2 className="h-4 w-4" />
            Organization
          </Link>
          <Link
            href="/approvals"
            className={cn(
              "flex items-center gap-3 rounded-lg px-3 py-2.5 text-slate-700 transition",
              pathname.startsWith("/approvals")
                ? "bg-blue-100 text-blue-800 font-medium"
                : "hover:bg-slate-100",
            )}
          >
            <CheckCircle2 className="h-4 w-4" />
            Approvals
          </Link>
          <Link
            href="/activity"
            className={cn(
              "flex items-center gap-3 rounded-lg px-3 py-2.5 text-slate-700 transition",
              pathname.startsWith("/activity")
                ? "bg-blue-100 text-blue-800 font-medium"
                : "hover:bg-slate-100",
            )}
          >
            <Activity className="h-4 w-4" />
            Live feed
          </Link>
          {isAdmin ? (
            <Link
              href="/agents"
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2.5 text-slate-700 transition",
                pathname.startsWith("/agents")
                  ? "bg-blue-100 text-blue-800 font-medium"
                  : "hover:bg-slate-100",
              )}
            >
              <Bot className="h-4 w-4" />
              Agents
            </Link>
          ) : null}
        </nav>
      </div>
      <div className="border-t border-slate-200 p-4">
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <span
            className={cn(
              "h-2 w-2 rounded-full",
              systemStatus === "operational" && "bg-emerald-500",
              systemStatus === "degraded" && "bg-rose-500",
              systemStatus === "unknown" && "bg-slate-300",
            )}
          />
          {statusLabel}
        </div>
      </div>
    </aside>
  );
}
