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
  type listBoardsApiV1BoardsGetResponse,
  getListBoardsApiV1BoardsGetQueryKey,
  useDeleteBoardApiV1BoardsBoardIdDelete,
  useListBoardsApiV1BoardsGet,
} from "@/api/generated/boards/boards";
import type { BoardRead } from "@/api/generated/model";
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

export default function BoardsPage() {
  const { isSignedIn } = useAuth();
  const queryClient = useQueryClient();
  const [deleteTarget, setDeleteTarget] = useState<BoardRead | null>(null);

  const boardsKey = getListBoardsApiV1BoardsGetQueryKey();
  const boardsQuery = useListBoardsApiV1BoardsGet<
    listBoardsApiV1BoardsGetResponse,
    ApiError
  >(undefined, {
    query: {
      enabled: Boolean(isSignedIn),
      refetchInterval: 30_000,
      refetchOnMount: "always",
    },
  });

  const boards = useMemo(
    () =>
      boardsQuery.data?.status === 200
        ? (boardsQuery.data.data.items ?? [])
        : [],
    [boardsQuery.data],
  );

  const deleteMutation = useDeleteBoardApiV1BoardsBoardIdDelete<
    ApiError,
    { previous?: listBoardsApiV1BoardsGetResponse }
  >(
    {
      mutation: {
        onMutate: async ({ boardId }) => {
          await queryClient.cancelQueries({ queryKey: boardsKey });
          const previous =
            queryClient.getQueryData<listBoardsApiV1BoardsGetResponse>(
              boardsKey,
            );
          if (previous && previous.status === 200) {
            const nextItems = previous.data.items.filter(
              (board) => board.id !== boardId,
            );
            const removedCount = previous.data.items.length - nextItems.length;
            queryClient.setQueryData<listBoardsApiV1BoardsGetResponse>(
              boardsKey,
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
        onError: (_error, _board, context) => {
          if (context?.previous) {
            queryClient.setQueryData(boardsKey, context.previous);
          }
        },
        onSuccess: () => {
          setDeleteTarget(null);
        },
        onSettled: () => {
          queryClient.invalidateQueries({ queryKey: boardsKey });
        },
      },
    },
    queryClient,
  );

  const handleDelete = () => {
    if (!deleteTarget) return;
    deleteMutation.mutate({ boardId: deleteTarget.id });
  };

  const columns = useMemo<ColumnDef<BoardRead>[]>(
    () => [
      {
        accessorKey: "name",
        header: "Board",
        cell: ({ row }) => (
          <Link href={`/boards/${row.original.id}`} className="group block">
            <p className="text-sm font-medium text-slate-900 group-hover:text-blue-600">
              {row.original.name}
            </p>
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
              href={`/boards/${row.original.id}/edit`}
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
    data: boards,
    columns,
    getCoreRowModel: getCoreRowModel(),
  });

  return (
    <DashboardShell>
      <SignedOut>
        <div className="col-span-2 flex min-h-[calc(100vh-64px)] items-center justify-center bg-slate-50 p-10 text-center">
          <div className="rounded-xl border border-slate-200 bg-white px-8 py-6 shadow-sm">
            <p className="text-sm text-slate-600">Sign in to view boards.</p>
            <SignInButton
              mode="modal"
              forceRedirectUrl="/boards"
              signUpForceRedirectUrl="/boards"
            >
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
                    Boards
                  </h1>
                  <p className="mt-1 text-sm text-slate-500">
                    Manage boards and task workflows. {boards.length} board
                    {boards.length === 1 ? "" : "s"} total.
                  </p>
                </div>
                {boards.length > 0 ? (
                  <Link
                    href="/boards/new"
                    className={buttonVariants({
                      size: "md",
                      variant: "primary",
                    })}
                  >
                    Create board
                  </Link>
                ) : null}
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
                    {boardsQuery.isLoading ? (
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
                                <rect x="3" y="3" width="7" height="7" />
                                <rect x="14" y="3" width="7" height="7" />
                                <rect x="14" y="14" width="7" height="7" />
                                <rect x="3" y="14" width="7" height="7" />
                              </svg>
                            </div>
                            <h3 className="mb-2 text-lg font-semibold text-slate-900">
                              No boards yet
                            </h3>
                            <p className="mb-6 max-w-md text-sm text-slate-500">
                              Create your first board to start routing tasks and
                              monitoring work across agents.
                            </p>
                            <Link
                              href="/boards/new"
                              className={buttonVariants({
                                size: "md",
                                variant: "primary",
                              })}
                            >
                              Create your first board
                            </Link>
                          </div>
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            {boardsQuery.error ? (
              <p className="mt-4 text-sm text-red-500">
                {boardsQuery.error.message}
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
        <DialogContent aria-label="Delete board">
          <DialogHeader>
            <DialogTitle>Delete board</DialogTitle>
            <DialogDescription>
              This will remove {deleteTarget?.name}. This action cannot be
              undone.
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
