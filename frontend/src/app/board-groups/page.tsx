"use client";

export const dynamic = "force-dynamic";

import { useMemo, useState } from "react";
import Link from "next/link";

import { SignInButton, SignedIn, SignedOut, useAuth } from "@/auth/clerk";
import {
  type ColumnDef,
  flexRender,
  getCoreRowModel,
  useReactTable,
} from "@tanstack/react-table";
import { useQueryClient } from "@tanstack/react-query";

import { ApiError } from "@/api/mutator";
import {
  type listBoardGroupsApiV1BoardGroupsGetResponse,
  getListBoardGroupsApiV1BoardGroupsGetQueryKey,
  useDeleteBoardGroupApiV1BoardGroupsGroupIdDelete,
  useListBoardGroupsApiV1BoardGroupsGet,
} from "@/api/generated/board-groups/board-groups";
import type { BoardGroupRead } from "@/api/generated/model";
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

export default function BoardGroupsPage() {
  const { isSignedIn } = useAuth();
  const queryClient = useQueryClient();
  const [deleteTarget, setDeleteTarget] = useState<BoardGroupRead | null>(null);

  const groupsKey = getListBoardGroupsApiV1BoardGroupsGetQueryKey();
  const groupsQuery = useListBoardGroupsApiV1BoardGroupsGet<
    listBoardGroupsApiV1BoardGroupsGetResponse,
    ApiError
  >(undefined, {
    query: {
      enabled: Boolean(isSignedIn),
      refetchInterval: 30_000,
      refetchOnMount: "always",
    },
  });

  const groups = useMemo(
    () =>
      groupsQuery.data?.status === 200
        ? (groupsQuery.data.data.items ?? [])
        : [],
    [groupsQuery.data],
  );

  const deleteMutation = useDeleteBoardGroupApiV1BoardGroupsGroupIdDelete<
    ApiError,
    { previous?: listBoardGroupsApiV1BoardGroupsGetResponse }
  >(
    {
      mutation: {
        onMutate: async ({ groupId }) => {
          await queryClient.cancelQueries({ queryKey: groupsKey });
          const previous =
            queryClient.getQueryData<listBoardGroupsApiV1BoardGroupsGetResponse>(
              groupsKey,
            );
          if (previous && previous.status === 200) {
            const nextItems = previous.data.items.filter(
              (group) => group.id !== groupId,
            );
            const removedCount = previous.data.items.length - nextItems.length;
            queryClient.setQueryData<listBoardGroupsApiV1BoardGroupsGetResponse>(
              groupsKey,
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
        onError: (_error, _group, context) => {
          if (context?.previous) {
            queryClient.setQueryData(groupsKey, context.previous);
          }
        },
        onSuccess: () => {
          setDeleteTarget(null);
        },
        onSettled: () => {
          queryClient.invalidateQueries({ queryKey: groupsKey });
        },
      },
    },
    queryClient,
  );

  const handleDelete = () => {
    if (!deleteTarget) return;
    deleteMutation.mutate({ groupId: deleteTarget.id });
  };

  const columns = useMemo<ColumnDef<BoardGroupRead>[]>(
    () => [
      {
        accessorKey: "name",
        header: "Group",
        cell: ({ row }) => (
          <Link
            href={`/board-groups/${row.original.id}`}
            className="group block"
          >
            <p className="text-sm font-medium text-slate-900 group-hover:text-blue-600">
              {row.original.name}
            </p>
            {row.original.description ? (
              <p className="mt-1 text-xs text-slate-500 line-clamp-2">
                {row.original.description}
              </p>
            ) : (
              <p className="mt-1 text-xs text-slate-400">No description</p>
            )}
          </Link>
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
          <div className="flex items-center justify-end gap-2">
            <Link
              href={`/board-groups/${row.original.id}/edit`}
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
    data: groups,
    columns,
    getCoreRowModel: getCoreRowModel(),
  });

  return (
    <DashboardShell>
      <SignedOut>
        <div className="col-span-2 flex min-h-[calc(100vh-64px)] items-center justify-center bg-slate-50 p-10 text-center">
          <div className="rounded-xl border border-slate-200 bg-white px-8 py-6 shadow-sm">
            <p className="text-sm text-slate-600">
              Sign in to view board groups.
            </p>
            <SignInButton mode="modal" forceRedirectUrl="/board-groups">
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
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <h1 className="text-2xl font-semibold tracking-tight text-slate-900">
                    Board groups
                  </h1>
                  <p className="mt-1 text-sm text-slate-500">
                    Group boards so agents can see related work. {groups.length}{" "}
                    group{groups.length === 1 ? "" : "s"} total.
                  </p>
                </div>
                <Link
                  href="/board-groups/new"
                  className={buttonVariants({ size: "md", variant: "primary" })}
                >
                  Create group
                </Link>
              </div>
            </div>
          </div>

          <div className="p-8">
            <div className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead className="sticky top-0 z-10 bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
                    {table.getHeaderGroups().map((headerGroup) => (
                      <tr key={headerGroup.id}>
                        {headerGroup.headers.map((header) => (
                          <th
                            key={header.id}
                            className="px-6 py-3 text-left font-semibold"
                          >
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
                    {groupsQuery.isLoading ? (
                      <tr>
                        <td colSpan={columns.length} className="px-6 py-8">
                          <span className="text-sm text-slate-500">
                            Loading…
                          </span>
                        </td>
                      </tr>
                    ) : table.getRowModel().rows.length ? (
                      table.getRowModel().rows.map((row) => (
                        <tr
                          key={row.id}
                          className="transition hover:bg-slate-50"
                        >
                          {row.getVisibleCells().map((cell) => (
                            <td key={cell.id} className="px-6 py-4 align-top">
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
                                <path d="M3 7h8" />
                                <path d="M3 17h8" />
                                <path d="M13 7h8" />
                                <path d="M13 17h8" />
                                <path d="M3 12h18" />
                              </svg>
                            </div>
                            <h3 className="mb-2 text-lg font-semibold text-slate-900">
                              No groups yet
                            </h3>
                            <p className="mb-6 max-w-md text-sm text-slate-500">
                              Create a board group to increase cross-board
                              visibility for agents.
                            </p>
                            <Link
                              href="/board-groups/new"
                              className={buttonVariants({
                                size: "md",
                                variant: "primary",
                              })}
                            >
                              Create your first group
                            </Link>
                          </div>
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            {groupsQuery.error ? (
              <p className="mt-4 text-sm text-red-500">
                {groupsQuery.error.message}
              </p>
            ) : null}
          </div>
        </main>
      </SignedIn>

      <Dialog
        open={!!deleteTarget}
        onOpenChange={(nextOpen) => {
          if (!nextOpen) {
            setDeleteTarget(null);
          }
        }}
      >
        <DialogContent aria-label="Delete board group">
          <DialogHeader>
            <DialogTitle>Delete board group</DialogTitle>
            <DialogDescription>
              This will remove {deleteTarget?.name}. Boards will be ungrouped.
              This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          {deleteMutation.error ? (
            <div className="rounded-lg border border-[color:var(--border)] bg-[color:var(--surface-muted)] p-3 text-xs text-muted">
              {deleteMutation.error.message}
            </div>
          ) : null}
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteTarget(null)}>
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
