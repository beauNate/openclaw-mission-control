"use client";

export const dynamic = "force-dynamic";

import { useMemo, useState } from "react";
import Link from "next/link";

import { SignInButton, SignedIn, SignedOut, useAuth } from "@/auth/clerk";
import {
  type ColumnDef,
  type SortingState,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
} from "@tanstack/react-table";
import { useQueryClient } from "@tanstack/react-query";

import { DashboardSidebar } from "@/components/organisms/DashboardSidebar";
import { DashboardShell } from "@/components/templates/DashboardShell";
import { Button, buttonVariants } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

import { ApiError } from "@/api/mutator";
import {
  type listGatewaysApiV1GatewaysGetResponse,
  getListGatewaysApiV1GatewaysGetQueryKey,
  useDeleteGatewayApiV1GatewaysGatewayIdDelete,
  useListGatewaysApiV1GatewaysGet,
} from "@/api/generated/gateways/gateways";
import {
  type getMyMembershipApiV1OrganizationsMeMemberGetResponse,
  useGetMyMembershipApiV1OrganizationsMeMemberGet,
} from "@/api/generated/organizations/organizations";
import type { GatewayRead } from "@/api/generated/model";

const truncate = (value?: string | null, max = 24) => {
  if (!value) return "—";
  if (value.length <= max) return value;
  return `${value.slice(0, max)}…`;
};

const formatTimestamp = (value?: string | null) => {
  if (!value) return "—";
  const date = new Date(`${value}${value.endsWith("Z") ? "" : "Z"}`);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
};

export default function GatewaysPage() {
  const { isSignedIn } = useAuth();
  const queryClient = useQueryClient();

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
  const [sorting, setSorting] = useState<SortingState>([
    { id: "name", desc: false },
  ]);
  const [deleteTarget, setDeleteTarget] = useState<GatewayRead | null>(null);

  const gatewaysKey = getListGatewaysApiV1GatewaysGetQueryKey();
  const gatewaysQuery = useListGatewaysApiV1GatewaysGet<
    listGatewaysApiV1GatewaysGetResponse,
    ApiError
  >(undefined, {
    query: {
      enabled: Boolean(isSignedIn && isAdmin),
      refetchInterval: 30_000,
      refetchOnMount: "always",
    },
  });

  const gateways = useMemo(
    () =>
      gatewaysQuery.data?.status === 200
        ? (gatewaysQuery.data.data.items ?? [])
        : [],
    [gatewaysQuery.data],
  );
  const sortedGateways = useMemo(() => [...gateways], [gateways]);

  const deleteMutation = useDeleteGatewayApiV1GatewaysGatewayIdDelete<
    ApiError,
    { previous?: listGatewaysApiV1GatewaysGetResponse }
  >(
    {
      mutation: {
        onMutate: async ({ gatewayId }) => {
          await queryClient.cancelQueries({ queryKey: gatewaysKey });
          const previous =
            queryClient.getQueryData<listGatewaysApiV1GatewaysGetResponse>(
              gatewaysKey,
            );
          if (previous && previous.status === 200) {
            const nextItems = previous.data.items.filter(
              (gateway) => gateway.id !== gatewayId,
            );
            const removedCount = previous.data.items.length - nextItems.length;
            queryClient.setQueryData<listGatewaysApiV1GatewaysGetResponse>(
              gatewaysKey,
              {
                ...previous,
                data: {
                  ...previous.data,
                  items: nextItems,
                  total: Math.max(0, previous.data.total - removedCount),
                },
              },
            );
          }
          return { previous };
        },
        onError: (_error, _gateway, context) => {
          if (context?.previous) {
            queryClient.setQueryData(gatewaysKey, context.previous);
          }
        },
        onSuccess: () => {
          setDeleteTarget(null);
        },
        onSettled: () => {
          queryClient.invalidateQueries({ queryKey: gatewaysKey });
        },
      },
    },
    queryClient,
  );

  const handleDelete = () => {
    if (!deleteTarget) return;
    deleteMutation.mutate({ gatewayId: deleteTarget.id });
  };

  const columns = useMemo<ColumnDef<GatewayRead>[]>(
    () => [
      {
        accessorKey: "name",
        header: "Gateway",
        cell: ({ row }) => (
          <Link href={`/gateways/${row.original.id}`} className="group block">
            <p className="text-sm font-medium text-slate-900 group-hover:text-blue-600">
              {row.original.name}
            </p>
            <p className="text-xs text-slate-500">
              {truncate(row.original.url, 36)}
            </p>
          </Link>
        ),
      },
      {
        accessorKey: "main_session_key",
        header: "Main session",
        cell: ({ row }) => (
          <span className="text-sm text-slate-700">
            {truncate(row.original.main_session_key, 24)}
          </span>
        ),
      },
      {
        accessorKey: "workspace_root",
        header: "Workspace root",
        cell: ({ row }) => (
          <span className="text-sm text-slate-700">
            {truncate(row.original.workspace_root, 28)}
          </span>
        ),
      },
      {
        accessorKey: "updated_at",
        header: "Updated",
        cell: ({ row }) => (
          <span className="text-sm text-slate-700">
            {formatTimestamp(row.original.updated_at)}
          </span>
        ),
      },
      {
        id: "actions",
        header: "",
        cell: ({ row }) => (
          <div className="flex justify-end gap-2">
            <Link
              href={`/gateways/${row.original.id}/edit`}
              className={buttonVariants({ variant: "ghost", size: "sm" })}
            >
              Edit
            </Link>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setDeleteTarget(row.original)}
            >
              Delete
            </Button>
          </div>
        ),
      },
    ],
    [],
  );

  // eslint-disable-next-line react-hooks/incompatible-library
  const table = useReactTable({
    data: sortedGateways,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  return (
    <DashboardShell>
      <SignedOut>
        <div className="col-span-2 flex min-h-[calc(100vh-64px)] items-center justify-center bg-slate-50 p-10 text-center">
          <div className="rounded-xl border border-slate-200 bg-white px-8 py-6 shadow-sm">
            <p className="text-sm text-slate-600">Sign in to view gateways.</p>
            <SignInButton mode="modal" forceRedirectUrl="/gateways">
              <Button className="mt-4">Sign in</Button>
            </SignInButton>
          </div>
        </div>
      </SignedOut>
      <SignedIn>
        <DashboardSidebar />
        <main className="flex-1 overflow-y-auto bg-slate-50">
          <div className="sticky top-0 z-30 border-b border-slate-200 bg-white">
            <div className="px-8 py-6">
              <div className="flex items-center justify-between">
                <div>
                  <h1 className="text-2xl font-semibold tracking-tight text-slate-900">
                    Gateways
                  </h1>
                  <p className="mt-1 text-sm text-slate-500">
                    Manage OpenClaw gateway connections used by boards
                  </p>
                </div>
                {isAdmin && gateways.length > 0 ? (
                  <Link
                    href="/gateways/new"
                    className={buttonVariants({
                      size: "md",
                      variant: "primary",
                    })}
                  >
                    Create gateway
                  </Link>
                ) : null}
              </div>
            </div>
          </div>

          <div className="p-8">
            {!isAdmin ? (
              <div className="rounded-xl border border-slate-200 bg-white px-6 py-5 text-sm text-slate-600 shadow-sm">
                Only organization owners and admins can access gateways.
              </div>
            ) : (
              <>
                <div className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-sm">
                      <thead className="sticky top-0 z-10 bg-slate-50 text-xs font-semibold uppercase tracking-wider text-slate-500">
                        {table.getHeaderGroups().map((headerGroup) => (
                          <tr key={headerGroup.id}>
                            {headerGroup.headers.map((header) => (
                              <th key={header.id} className="px-6 py-3">
                                {header.isPlaceholder
                                  ? null
                                  : flexRender(
                                      header.column.columnDef.header,
                                      header.getContext(),
                                    )}
                              </th>
                            ))}
                          </tr>
                        ))}
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {gatewaysQuery.isLoading ? (
                          <tr>
                            <td colSpan={columns.length} className="px-6 py-8">
                              <span className="text-sm text-slate-500">
                                Loading…
                              </span>
                            </td>
                          </tr>
                        ) : table.getRowModel().rows.length ? (
                          table.getRowModel().rows.map((row) => (
                            <tr key={row.id} className="hover:bg-slate-50">
                              {row.getVisibleCells().map((cell) => (
                                <td key={cell.id} className="px-6 py-4">
                                  {flexRender(
                                    cell.column.columnDef.cell,
                                    cell.getContext(),
                                  )}
                                </td>
                              ))}
                            </tr>
                          ))
                        ) : (
                          <tr>
                            <td colSpan={columns.length} className="px-6 py-16">
                              <div className="flex flex-col items-center justify-center text-center">
                                <div className="mb-4 rounded-full bg-slate-50 p-4">
                                  <svg
                                    className="h-16 w-16 text-slate-300"
                                    viewBox="0 0 24 24"
                                    fill="none"
                                    stroke="currentColor"
                                    strokeWidth="1.5"
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                  >
                                    <rect
                                      x="2"
                                      y="7"
                                      width="20"
                                      height="14"
                                      rx="2"
                                      ry="2"
                                    />
                                    <path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16" />
                                  </svg>
                                </div>
                                <h3 className="mb-2 text-lg font-semibold text-slate-900">
                                  No gateways yet
                                </h3>
                                <p className="mb-6 max-w-md text-sm text-slate-500">
                                  Create your first gateway to connect boards
                                  and start managing your OpenClaw connections.
                                </p>
                                <Link
                                  href="/gateways/new"
                                  className={buttonVariants({
                                    size: "md",
                                    variant: "primary",
                                  })}
                                >
                                  Create your first gateway
                                </Link>
                              </div>
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>

                {gatewaysQuery.error ? (
                  <p className="mt-4 text-sm text-red-500">
                    {gatewaysQuery.error.message}
                  </p>
                ) : null}
              </>
            )}
          </div>
        </main>
      </SignedIn>

      <Dialog
        open={Boolean(deleteTarget)}
        onOpenChange={() => setDeleteTarget(null)}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete gateway?</DialogTitle>
            <DialogDescription>
              This removes the gateway connection from Mission Control. Boards
              using it will need a new gateway assigned.
            </DialogDescription>
          </DialogHeader>
          {deleteMutation.error ? (
            <p className="text-sm text-red-500">
              {deleteMutation.error.message}
            </p>
          ) : null}
          <DialogFooter>
            <Button variant="ghost" onClick={() => setDeleteTarget(null)}>
              Cancel
            </Button>
            <Button onClick={handleDelete} disabled={deleteMutation.isPending}>
              {deleteMutation.isPending ? "Deleting…" : "Delete"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </DashboardShell>
  );
}
