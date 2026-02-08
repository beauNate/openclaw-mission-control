"use client";

export const dynamic = "force-dynamic";

import { useEffect, useMemo, useState } from "react";

import { SignInButton, SignedIn, SignedOut, useAuth } from "@/auth/clerk";
import { useQueryClient } from "@tanstack/react-query";
import { Building2, Copy, UserPlus, Users } from "lucide-react";

import { ApiError } from "@/api/mutator";
import {
  type listBoardsApiV1BoardsGetResponse,
  useListBoardsApiV1BoardsGet,
} from "@/api/generated/boards/boards";
import {
  type getMyOrgApiV1OrganizationsMeGetResponse,
  type getMyMembershipApiV1OrganizationsMeMemberGetResponse,
  type getOrgMemberApiV1OrganizationsMeMembersMemberIdGetResponse,
  type listOrgInvitesApiV1OrganizationsMeInvitesGetResponse,
  type listOrgMembersApiV1OrganizationsMeMembersGetResponse,
  getGetOrgMemberApiV1OrganizationsMeMembersMemberIdGetQueryKey,
  getListOrgInvitesApiV1OrganizationsMeInvitesGetQueryKey,
  getListOrgMembersApiV1OrganizationsMeMembersGetQueryKey,
  useCreateOrgInviteApiV1OrganizationsMeInvitesPost,
  useGetMyOrgApiV1OrganizationsMeGet,
  useGetMyMembershipApiV1OrganizationsMeMemberGet,
  useGetOrgMemberApiV1OrganizationsMeMembersMemberIdGet,
  useListOrgInvitesApiV1OrganizationsMeInvitesGet,
  useListOrgMembersApiV1OrganizationsMeMembersGet,
  useRevokeOrgInviteApiV1OrganizationsMeInvitesInviteIdDelete,
  useUpdateMemberAccessApiV1OrganizationsMeMembersMemberIdAccessPut,
  useUpdateOrgMemberApiV1OrganizationsMeMembersMemberIdPatch,
} from "@/api/generated/organizations/organizations";
import type {
  BoardRead,
  OrganizationBoardAccessSpec,
  OrganizationInviteRead,
  OrganizationMemberRead,
} from "@/api/generated/model";
import { DashboardSidebar } from "@/components/organisms/DashboardSidebar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { DashboardShell } from "@/components/templates/DashboardShell";
import { cn } from "@/lib/utils";

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

type AccessScope = "all" | "custom";

type BoardAccessState = Record<string, { read: boolean; write: boolean }>;

const buildAccessList = (
  access: BoardAccessState,
): OrganizationBoardAccessSpec[] =>
  Object.entries(access)
    .filter(([, entry]) => entry.read || entry.write)
    .map(([boardId, entry]) => ({
      board_id: boardId,
      can_read: entry.read || entry.write,
      can_write: entry.write,
    }));

const summarizeAccess = (allRead: boolean, allWrite: boolean) => {
  if (allRead || allWrite) {
    if (allRead && allWrite) return "All boards: read + write";
    if (allWrite) return "All boards: write";
    return "All boards: read";
  }
  return "Selected boards";
};

const roleBadgeVariant = (role: string) => {
  if (role === "admin" || role === "owner") return "accent" as const;
  return "outline" as const;
};

const defaultBoardAccess: BoardAccessState = {};

const initialsFrom = (value?: string | null) => {
  if (!value) return "?";
  const parts = value.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return "?";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return `${parts[0][0]}${parts[1][0]}`.toUpperCase();
};

function BoardAccessEditor({
  boards,
  scope,
  onScopeChange,
  allRead,
  allWrite,
  onAllReadChange,
  onAllWriteChange,
  access,
  onAccessChange,
  disabled,
  emptyMessage,
}: {
  boards: BoardRead[];
  scope: AccessScope;
  onScopeChange: (scope: AccessScope) => void;
  allRead: boolean;
  allWrite: boolean;
  onAllReadChange: (next: boolean) => void;
  onAllWriteChange: (next: boolean) => void;
  access: BoardAccessState;
  onAccessChange: (next: BoardAccessState) => void;
  disabled?: boolean;
  emptyMessage?: string;
}) {
  const handleAllReadToggle = () => {
    if (disabled) return;
    const next = !allRead;
    onAllReadChange(next);
    if (!next && allWrite) {
      onAllWriteChange(false);
    }
  };

  const handleAllWriteToggle = () => {
    if (disabled) return;
    const next = !allWrite;
    onAllWriteChange(next);
    if (next && !allRead) {
      onAllReadChange(true);
    }
  };

  const updateBoardAccess = (
    boardId: string,
    next: { read: boolean; write: boolean },
  ) => {
    onAccessChange({
      ...access,
      [boardId]: {
        read: next.read || next.write,
        write: next.write,
      },
    });
  };

  const handleBoardReadToggle = (boardId: string) => {
    if (disabled) return;
    const current = access[boardId] ?? { read: false, write: false };
    const nextRead = !current.read;
    const nextWrite = nextRead ? current.write : false;
    updateBoardAccess(boardId, { read: nextRead, write: nextWrite });
  };

  const handleBoardWriteToggle = (boardId: string) => {
    if (disabled) return;
    const current = access[boardId] ?? { read: false, write: false };
    const nextWrite = !current.write;
    const nextRead = nextWrite ? true : current.read;
    updateBoardAccess(boardId, { read: nextRead, write: nextWrite });
  };

  return (
    <div className="space-y-3">
      <div>
        <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
          Board access
        </p>
        <div className="mt-3 inline-flex rounded-xl border border-slate-200 bg-slate-100 p-1">
          <button
            type="button"
            className={cn(
              "rounded-md px-3 py-1.5 text-xs font-semibold transition",
              scope === "all"
                ? "bg-white text-slate-900 shadow-sm"
                : "text-slate-500 hover:text-slate-700",
            )}
            onClick={() => onScopeChange("all")}
            disabled={disabled}
          >
            All boards
          </button>
          <button
            type="button"
            className={cn(
              "rounded-md px-3 py-1.5 text-xs font-semibold transition",
              scope === "custom"
                ? "bg-white text-slate-900 shadow-sm"
                : "text-slate-500 hover:text-slate-700",
            )}
            onClick={() => onScopeChange("custom")}
            disabled={disabled}
          >
            Selected boards
          </button>
        </div>
      </div>

      {scope === "all" ? (
        <div className="flex flex-wrap items-center gap-6 rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-sm">
          <label className="flex items-center gap-2 text-slate-600">
            <input
              type="checkbox"
              className="h-4 w-4"
              checked={allRead}
              onChange={handleAllReadToggle}
              disabled={disabled}
            />
            Read
          </label>
          <label className="flex items-center gap-2 text-slate-600">
            <input
              type="checkbox"
              className="h-4 w-4"
              checked={allWrite}
              onChange={handleAllWriteToggle}
              disabled={disabled}
            />
            Write
          </label>
          <span className="text-xs text-slate-500">
            Write access implies read permissions.
          </span>
        </div>
      ) : (
        <div>
          {boards.length === 0 ? (
            <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-500">
              {emptyMessage ?? "No boards available yet."}
            </div>
          ) : (
            <div className="overflow-hidden rounded-xl border border-slate-200">
              <table className="min-w-full text-sm">
                <thead className="bg-slate-50 text-[11px] uppercase tracking-wide text-slate-500">
                  <tr>
                    <th className="px-4 py-2 text-left font-medium">Board</th>
                    <th className="px-4 py-2 text-center font-medium">Read</th>
                    <th className="px-4 py-2 text-center font-medium">Write</th>
                  </tr>
                </thead>
                <tbody>
                  {boards.map((board) => {
                    const entry = access[board.id] ?? {
                      read: false,
                      write: false,
                    };
                    return (
                      <tr
                        key={board.id}
                        className="border-t border-slate-200 hover:bg-slate-50"
                      >
                        <td className="px-4 py-3">
                          <div className="text-sm font-semibold text-slate-900">
                            {board.name}
                          </div>
                          <div className="text-xs text-slate-500">
                            {board.slug}
                          </div>
                        </td>
                        <td className="px-4 py-3 text-center">
                          <input
                            type="checkbox"
                            className="h-4 w-4"
                            checked={entry.read}
                            onChange={() => handleBoardReadToggle(board.id)}
                            disabled={disabled}
                          />
                        </td>
                        <td className="px-4 py-3 text-center">
                          <input
                            type="checkbox"
                            className="h-4 w-4"
                            checked={entry.write}
                            onChange={() => handleBoardWriteToggle(board.id)}
                            disabled={disabled}
                          />
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function OrganizationPage() {
  const { isSignedIn } = useAuth();
  const queryClient = useQueryClient();

  const [inviteDialogOpen, setInviteDialogOpen] = useState(false);
  const [inviteEmail, setInviteEmail] = useState("");
  const [inviteRole, setInviteRole] = useState("member");
  const [inviteScope, setInviteScope] = useState<AccessScope>("all");
  const [inviteAllRead, setInviteAllRead] = useState(true);
  const [inviteAllWrite, setInviteAllWrite] = useState(false);
  const [inviteAccess, setInviteAccess] =
    useState<BoardAccessState>(defaultBoardAccess);
  const [inviteError, setInviteError] = useState<string | null>(null);
  const [copiedInviteId, setCopiedInviteId] = useState<string | null>(null);

  const [accessDialogOpen, setAccessDialogOpen] = useState(false);
  const [activeMemberId, setActiveMemberId] = useState<string | null>(null);
  const [accessScope, setAccessScope] = useState<AccessScope>("all");
  const [accessAllRead, setAccessAllRead] = useState(false);
  const [accessAllWrite, setAccessAllWrite] = useState(false);
  const [accessRole, setAccessRole] = useState("member");
  const [accessMap, setAccessMap] =
    useState<BoardAccessState>(defaultBoardAccess);
  const [accessError, setAccessError] = useState<string | null>(null);

  const orgQuery = useGetMyOrgApiV1OrganizationsMeGet<
    getMyOrgApiV1OrganizationsMeGetResponse,
    ApiError
  >({
    query: {
      enabled: Boolean(isSignedIn),
      refetchOnMount: "always",
    },
  });

  const membersQuery = useListOrgMembersApiV1OrganizationsMeMembersGet<
    listOrgMembersApiV1OrganizationsMeMembersGetResponse,
    ApiError
  >(
    { limit: 200 },
    {
      query: {
        enabled: Boolean(isSignedIn),
        refetchOnMount: "always",
      },
    },
  );

  const boardsQuery = useListBoardsApiV1BoardsGet<
    listBoardsApiV1BoardsGetResponse,
    ApiError
  >(
    { limit: 200 },
    {
      query: {
        enabled: Boolean(isSignedIn),
        refetchOnMount: "always",
      },
    },
  );

  const membershipQuery = useGetMyMembershipApiV1OrganizationsMeMemberGet<
    getMyMembershipApiV1OrganizationsMeMemberGetResponse,
    ApiError
  >({
    query: {
      enabled: Boolean(isSignedIn),
      refetchOnMount: "always",
    },
  });

  const isAdmin =
    membershipQuery.data?.status === 200 &&
    (membershipQuery.data.data.role === "admin" ||
      membershipQuery.data.data.role === "owner");

  const invitesQuery = useListOrgInvitesApiV1OrganizationsMeInvitesGet<
    listOrgInvitesApiV1OrganizationsMeInvitesGetResponse,
    ApiError
  >(
    { limit: 200 },
    {
      query: {
        enabled: Boolean(isSignedIn && isAdmin),
        refetchOnMount: "always",
        retry: false,
      },
    },
  );

  const members = useMemo(() => {
    if (membersQuery.data?.status !== 200) return [];
    return membersQuery.data.data.items ?? [];
  }, [membersQuery.data]);

  const invites = useMemo<OrganizationInviteRead[]>(() => {
    if (invitesQuery.data?.status !== 200) return [];
    return invitesQuery.data.data.items ?? [];
  }, [invitesQuery.data]);

  const boards = useMemo<BoardRead[]>(() => {
    if (boardsQuery.data?.status !== 200) return [];
    return boardsQuery.data.data.items ?? [];
  }, [boardsQuery.data]);

  const memberDetailsQuery =
    useGetOrgMemberApiV1OrganizationsMeMembersMemberIdGet<
      getOrgMemberApiV1OrganizationsMeMembersMemberIdGetResponse,
      ApiError
    >(activeMemberId ?? "", {
      query: {
        enabled: Boolean(activeMemberId && accessDialogOpen),
      },
    });

  const createInviteMutation =
    useCreateOrgInviteApiV1OrganizationsMeInvitesPost<ApiError>({
      mutation: {
        onSuccess: (result) => {
          if (result.status === 200) {
            setInviteEmail("");
            setInviteRole("member");
            setInviteScope("all");
            setInviteAllRead(true);
            setInviteAllWrite(false);
            setInviteAccess(defaultBoardAccess);
            setInviteError(null);
            queryClient.invalidateQueries({
              queryKey: getListOrgInvitesApiV1OrganizationsMeInvitesGetQueryKey(
                {
                  limit: 200,
                },
              ),
            });
            setInviteDialogOpen(false);
          }
        },
        onError: (err) => {
          setInviteError(err.message || "Unable to create invite.");
        },
      },
    });

  const revokeInviteMutation =
    useRevokeOrgInviteApiV1OrganizationsMeInvitesInviteIdDelete<ApiError>({
      mutation: {
        onSuccess: () => {
          queryClient.invalidateQueries({
            queryKey: getListOrgInvitesApiV1OrganizationsMeInvitesGetQueryKey({
              limit: 200,
            }),
          });
        },
      },
    });

  const updateMemberAccessMutation =
    useUpdateMemberAccessApiV1OrganizationsMeMembersMemberIdAccessPut<ApiError>(
      {
        mutation: {
          onSuccess: () => {
            queryClient.invalidateQueries({
              queryKey: getListOrgMembersApiV1OrganizationsMeMembersGetQueryKey(
                {
                  limit: 200,
                },
              ),
            });
            if (activeMemberId) {
              queryClient.invalidateQueries({
                queryKey:
                  getGetOrgMemberApiV1OrganizationsMeMembersMemberIdGetQueryKey(
                    activeMemberId,
                  ),
              });
            }
          },
        },
      },
    );

  const updateMemberRoleMutation =
    useUpdateOrgMemberApiV1OrganizationsMeMembersMemberIdPatch<ApiError>({
      mutation: {
        onSuccess: () => {
          queryClient.invalidateQueries({
            queryKey: getListOrgMembersApiV1OrganizationsMeMembersGetQueryKey({
              limit: 200,
            }),
          });
        },
      },
    });

  useEffect(() => {
    if (memberDetailsQuery.data?.status !== 200) return;
    const member = memberDetailsQuery.data.data;
    setAccessRole(member.role);
    const isAll = member.all_boards_read || member.all_boards_write;
    setAccessScope(isAll ? "all" : "custom");
    setAccessAllRead(member.all_boards_read);
    setAccessAllWrite(member.all_boards_write);
    const nextAccess: BoardAccessState = {};
    for (const entry of member.board_access ?? []) {
      nextAccess[entry.board_id] = {
        read: entry.can_read || entry.can_write,
        write: entry.can_write,
      };
    }
    setAccessMap(nextAccess);
    setAccessError(null);
  }, [memberDetailsQuery.data]);

  useEffect(() => {
    if (!accessDialogOpen) {
      setActiveMemberId(null);
      setAccessError(null);
    }
  }, [accessDialogOpen]);

  useEffect(() => {
    if (!inviteDialogOpen) {
      setInviteError(null);
    }
  }, [inviteDialogOpen]);

  const orgName =
    orgQuery.data?.status === 200 ? orgQuery.data.data.name : "Organization";

  const handleInviteSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!isSignedIn || !isAdmin) return;

    const trimmedEmail = inviteEmail.trim().toLowerCase();
    if (!trimmedEmail || !trimmedEmail.includes("@")) {
      setInviteError("Enter a valid email address.");
      return;
    }

    const hasAllAccess =
      inviteScope === "all" && (inviteAllRead || inviteAllWrite);
    const inviteAccessList = buildAccessList(inviteAccess);
    const hasCustomAccess =
      inviteScope === "custom" && inviteAccessList.length > 0;

    if (!hasAllAccess && !hasCustomAccess) {
      setInviteError("Select read or write access for at least one board.");
      return;
    }

    setInviteError(null);
    createInviteMutation.mutate({
      data: {
        invited_email: trimmedEmail,
        role: inviteRole,
        all_boards_read: inviteScope === "all" ? inviteAllRead : false,
        all_boards_write: inviteScope === "all" ? inviteAllWrite : false,
        board_access: inviteScope === "custom" ? inviteAccessList : [],
      },
    });
  };

  const handleCopyInvite = async (invite: OrganizationInviteRead) => {
    try {
      const baseUrl =
        typeof window !== "undefined" ? window.location.origin : "";
      const inviteUrl = baseUrl
        ? `${baseUrl}/invite?token=${invite.token}`
        : invite.token;
      let copied = false;

      if (typeof navigator !== "undefined" && navigator.clipboard) {
        try {
          await navigator.clipboard.writeText(inviteUrl);
          copied = true;
        } catch {
          copied = false;
        }
      }

      if (!copied && typeof document !== "undefined") {
        const textarea = document.createElement("textarea");
        textarea.value = inviteUrl;
        textarea.setAttribute("readonly", "true");
        textarea.style.position = "absolute";
        textarea.style.left = "-9999px";
        document.body.appendChild(textarea);
        textarea.select();
        copied = document.execCommand("copy");
        document.body.removeChild(textarea);
      }

      if (copied) {
        setCopiedInviteId(invite.id);
        setTimeout(() => setCopiedInviteId(null), 2000);
        return;
      }

      if (typeof window !== "undefined") {
        window.prompt("Copy invite link:", inviteUrl);
      }
    } catch {
      setCopiedInviteId(null);
    }
  };

  const openAccessDialog = (memberId: string) => {
    setActiveMemberId(memberId);
    setAccessDialogOpen(true);
  };

  const handleSaveAccess = async () => {
    if (!activeMemberId || !isAdmin) return;

    const hasAllAccess =
      accessScope === "all" && (accessAllRead || accessAllWrite);
    const accessList = buildAccessList(accessMap);
    const hasCustomAccess = accessScope === "custom" && accessList.length > 0;

    if (!hasAllAccess && !hasCustomAccess) {
      setAccessError("Select read or write access for at least one board.");
      return;
    }

    setAccessError(null);

    try {
      if (memberDetailsQuery.data?.status === 200) {
        const member = memberDetailsQuery.data.data;
        if (member.role !== accessRole) {
          await updateMemberRoleMutation.mutateAsync({
            memberId: member.id,
            data: { role: accessRole },
          });
        }
      }

      await updateMemberAccessMutation.mutateAsync({
        memberId: activeMemberId,
        data: {
          all_boards_read: accessScope === "all" ? accessAllRead : false,
          all_boards_write: accessScope === "all" ? accessAllWrite : false,
          board_access: accessScope === "custom" ? accessList : [],
        },
      });

      setAccessDialogOpen(false);
    } catch (err) {
      setAccessError(
        err instanceof Error ? err.message : "Unable to update member access.",
      );
    }
  };

  const memberAccessSummary = (member: OrganizationMemberRead) =>
    summarizeAccess(member.all_boards_read, member.all_boards_write);

  const memberDisplay = (member: OrganizationMemberRead) => {
    const primary =
      member.user?.name ||
      member.user?.preferred_name ||
      member.user?.email ||
      member.user_id;
    const secondary = member.user?.email ?? "No email on file";
    return {
      primary,
      secondary,
      initials: initialsFrom(primary),
    };
  };

  return (
    <DashboardShell>
      <SignedOut>
        <div className="col-span-2 flex min-h-[calc(100vh-64px)] items-center justify-center bg-slate-50 p-10 text-center">
          <div className="rounded-xl border border-slate-200 bg-white px-8 py-6 shadow-sm">
            <p className="text-sm text-slate-600">
              Sign in to manage your organization.
            </p>
            <SignInButton
              mode="modal"
              forceRedirectUrl="/organization"
              signUpForceRedirectUrl="/organization"
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
              <div className="flex flex-wrap items-center justify-between gap-6">
                <div>
                  <div className="flex flex-wrap items-center gap-3">
                    <h1 className="text-2xl font-semibold tracking-tight text-slate-900">
                      Organization
                    </h1>
                    <Badge
                      variant="outline"
                      className="flex items-center gap-2"
                    >
                      <Building2 className="h-3.5 w-3.5" />
                      {orgName}
                    </Badge>
                  </div>
                  <p className="mt-1 text-sm text-slate-500">
                    Manage members and board access across your workspace.
                  </p>
                  <div className="mt-3 flex flex-wrap items-center gap-4 text-xs text-slate-500">
                    <span>
                      <strong className="text-slate-900">
                        {members.length}
                      </strong>{" "}
                      members
                    </span>
                    <span>
                      <strong className="text-slate-900">
                        {boards.length}
                      </strong>{" "}
                      boards
                    </span>
                    <span>
                      <strong className="text-slate-900">
                        {invites.length}
                      </strong>{" "}
                      pending
                    </span>
                  </div>
                </div>
                <Button
                  type="button"
                  onClick={() => setInviteDialogOpen(true)}
                  disabled={!isAdmin}
                  title={
                    isAdmin ? undefined : "Only organization admins can invite"
                  }
                >
                  <UserPlus className="h-4 w-4" />
                  Invite member
                </Button>
              </div>
            </div>
          </div>

          <div className="px-8 py-8">
            <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white">
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 px-5 py-4">
                <div>
                  <h2 className="text-sm font-semibold text-slate-900">
                    Members & invites
                  </h2>
                  <p className="text-xs text-slate-500">
                    Invite teammates and tune their board permissions.
                  </p>
                </div>
                <div className="flex items-center gap-2 text-xs text-slate-500">
                  <Users className="h-4 w-4" />
                  {members.length + invites.length} total
                </div>
              </div>
              <div className="overflow-x-auto">
                <table className="min-w-full text-sm">
                  <thead className="bg-slate-50 text-[11px] uppercase tracking-wide text-slate-500">
                    <tr>
                      <th className="px-5 py-3 text-left font-medium">
                        Member
                      </th>
                      <th className="px-5 py-3 text-left font-medium">
                        Status
                      </th>
                      <th className="px-5 py-3 text-left font-medium">
                        Access
                      </th>
                      <th className="px-5 py-3 text-right font-medium">
                        Actions
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {membersQuery.isLoading ? (
                      <tr>
                        <td
                          colSpan={4}
                          className="px-5 py-6 text-center text-sm text-slate-500"
                        >
                          Loading members...
                        </td>
                      </tr>
                    ) : null}

                    {members.map((member) => {
                      const display = memberDisplay(member);
                      return (
                        <tr
                          key={member.id}
                          className="border-t border-slate-200 hover:bg-slate-50"
                        >
                          <td className="px-5 py-4">
                            <div className="flex items-center gap-3">
                              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-blue-500 to-indigo-500 text-xs font-semibold text-white">
                                {display.initials}
                              </div>
                              <div>
                                <div className="text-sm font-semibold text-slate-900">
                                  {display.primary}
                                </div>
                                <div className="text-xs text-slate-500">
                                  {display.secondary}
                                </div>
                              </div>
                            </div>
                          </td>
                          <td className="px-5 py-4">
                            <Badge variant={roleBadgeVariant(member.role)}>
                              {member.role}
                            </Badge>
                          </td>
                          <td className="px-5 py-4 text-slate-600">
                            {memberAccessSummary(member)}
                          </td>
                          <td className="px-5 py-4 text-right">
                            {isAdmin ? (
                              <Button
                                type="button"
                                variant="outline"
                                size="sm"
                                onClick={() => openAccessDialog(member.id)}
                              >
                                Manage access
                              </Button>
                            ) : (
                              <span className="text-xs text-slate-400">
                                Admin only
                              </span>
                            )}
                          </td>
                        </tr>
                      );
                    })}

                    {isAdmin && invitesQuery.isLoading ? (
                      <tr>
                        <td
                          colSpan={4}
                          className="px-5 py-6 text-center text-sm text-slate-500"
                        >
                          Loading invites...
                        </td>
                      </tr>
                    ) : null}

                    {isAdmin
                      ? invites.map((invite) => (
                          <tr
                            key={invite.id}
                            className="border-t border-slate-200 bg-slate-50/60"
                          >
                            <td className="px-5 py-4">
                              <div className="flex items-center gap-3">
                                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-slate-200 text-xs font-semibold text-slate-600">
                                  {initialsFrom(invite.invited_email)}
                                </div>
                                <div>
                                  <div className="text-sm font-semibold text-slate-900">
                                    {invite.invited_email}
                                  </div>
                                  <div className="text-xs text-slate-500">
                                    Invited {formatTimestamp(invite.created_at)}
                                  </div>
                                </div>
                              </div>
                            </td>
                            <td className="px-5 py-4">
                              <div className="flex flex-wrap items-center gap-2">
                                <Badge variant="warning">Pending</Badge>
                                <Badge variant={roleBadgeVariant(invite.role)}>
                                  {invite.role}
                                </Badge>
                              </div>
                            </td>
                            <td className="px-5 py-4 text-slate-600">
                              {summarizeAccess(
                                invite.all_boards_read,
                                invite.all_boards_write,
                              )}
                            </td>
                            <td className="px-5 py-4 text-right">
                              <div className="flex flex-wrap items-center justify-end gap-2">
                                <Button
                                  type="button"
                                  variant="outline"
                                  size="sm"
                                  onClick={() => handleCopyInvite(invite)}
                                >
                                  <Copy className="h-4 w-4" />
                                  {copiedInviteId === invite.id
                                    ? "Copied"
                                    : "Copy link"}
                                </Button>
                                <Button
                                  type="button"
                                  variant="ghost"
                                  size="sm"
                                  onClick={() =>
                                    revokeInviteMutation.mutate({
                                      inviteId: invite.id,
                                    })
                                  }
                                  disabled={revokeInviteMutation.isPending}
                                >
                                  Revoke
                                </Button>
                              </div>
                            </td>
                          </tr>
                        ))
                      : null}

                    {!membersQuery.isLoading &&
                    (!isAdmin || !invitesQuery.isLoading) &&
                    members.length === 0 &&
                    (!isAdmin || invites.length === 0) ? (
                      <tr>
                        <td
                          colSpan={4}
                          className="px-5 py-6 text-center text-sm text-slate-500"
                        >
                          No members or invites yet.
                        </td>
                      </tr>
                    ) : null}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </main>
      </SignedIn>

      <Dialog open={inviteDialogOpen} onOpenChange={setInviteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Invite a member</DialogTitle>
            <DialogDescription>
              Grant access to all boards or select specific workspaces.
            </DialogDescription>
          </DialogHeader>

          {isAdmin ? (
            <form className="space-y-5" onSubmit={handleInviteSubmit}>
              <div className="grid gap-4 sm:grid-cols-[1fr_200px]">
                <div className="space-y-2">
                  <label className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                    Email address
                  </label>
                  <Input
                    value={inviteEmail}
                    onChange={(event) => setInviteEmail(event.target.value)}
                    placeholder="name@company.com"
                    type="email"
                    required
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                    Role
                  </label>
                  <Select value={inviteRole} onValueChange={setInviteRole}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select role" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="member">Member</SelectItem>
                      <SelectItem value="admin">Admin</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <BoardAccessEditor
                boards={boards}
                scope={inviteScope}
                onScopeChange={setInviteScope}
                allRead={inviteAllRead}
                allWrite={inviteAllWrite}
                onAllReadChange={setInviteAllRead}
                onAllWriteChange={setInviteAllWrite}
                access={inviteAccess}
                onAccessChange={setInviteAccess}
                emptyMessage={
                  boardsQuery.isLoading
                    ? "Loading boards..."
                    : "Create a board to start assigning access."
                }
              />

              {inviteError ? (
                <p className="text-sm text-rose-500">{inviteError}</p>
              ) : null}

              <DialogFooter>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setInviteDialogOpen(false)}
                >
                  Cancel
                </Button>
                <Button type="submit" disabled={createInviteMutation.isPending}>
                  {createInviteMutation.isPending
                    ? "Sending invite..."
                    : "Send invite"}
                </Button>
              </DialogFooter>
            </form>
          ) : (
            <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-500">
              Only organization admins can invite new members.
            </div>
          )}
        </DialogContent>
      </Dialog>

      <Dialog open={accessDialogOpen} onOpenChange={setAccessDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Manage member access</DialogTitle>
            <DialogDescription>
              Adjust board permissions and role for this teammate.
            </DialogDescription>
          </DialogHeader>

          {memberDetailsQuery.isLoading ? (
            <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-500">
              Loading member access...
            </div>
          ) : memberDetailsQuery.data?.status === 200 ? (
            <div className="space-y-6">
              <div className="rounded-xl border border-slate-200 bg-white px-5 py-4">
                <p className="text-sm font-semibold text-slate-900">
                  {memberDetailsQuery.data.data.user?.name ||
                    memberDetailsQuery.data.data.user?.preferred_name ||
                    memberDetailsQuery.data.data.user?.email ||
                    memberDetailsQuery.data.data.user_id}
                </p>
                <p className="mt-1 text-xs text-slate-500">
                  {memberDetailsQuery.data.data.user?.email ??
                    "No email on file"}
                </p>
              </div>

              <div className="space-y-3">
                <label className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Role
                </label>
                <Select value={accessRole} onValueChange={setAccessRole}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select role" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="owner">Owner</SelectItem>
                    <SelectItem value="member">Member</SelectItem>
                    <SelectItem value="admin">Admin</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <BoardAccessEditor
                boards={boards}
                scope={accessScope}
                onScopeChange={setAccessScope}
                allRead={accessAllRead}
                allWrite={accessAllWrite}
                onAllReadChange={setAccessAllRead}
                onAllWriteChange={setAccessAllWrite}
                access={accessMap}
                onAccessChange={setAccessMap}
                emptyMessage={
                  boardsQuery.isLoading ? "Loading boards..." : undefined
                }
              />

              {accessError ? (
                <p className="text-sm text-rose-500">{accessError}</p>
              ) : null}
            </div>
          ) : (
            <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-500">
              Unable to load member access.
            </div>
          )}

          <DialogFooter className="pt-2">
            <Button
              type="button"
              variant="outline"
              onClick={() => setAccessDialogOpen(false)}
            >
              Cancel
            </Button>
            <Button
              type="button"
              onClick={handleSaveAccess}
              disabled={
                updateMemberAccessMutation.isPending ||
                updateMemberRoleMutation.isPending
              }
            >
              {updateMemberAccessMutation.isPending ||
              updateMemberRoleMutation.isPending
                ? "Saving..."
                : "Save changes"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </DashboardShell>
  );
}
