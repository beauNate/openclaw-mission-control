"use client";

import { memo, type HTMLAttributes } from "react";

import ReactMarkdown, { type Components } from "react-markdown";
import remarkBreaks from "remark-breaks";
import remarkGfm from "remark-gfm";

import { cn } from "@/lib/utils";

type MarkdownCodeProps = HTMLAttributes<HTMLElement> & {
  node?: unknown;
  inline?: boolean;
};

const MARKDOWN_CODE_COMPONENTS: Components = {
  pre: ({ node: _node, className, ...props }) => (
    <pre
      className={cn(
        "my-3 overflow-x-auto rounded-lg bg-slate-950 p-3 text-xs leading-relaxed text-slate-100",
        className,
      )}
      {...props}
    />
  ),
  code: (rawProps) => {
    // react-markdown passes `inline`, but the public `Components` typing doesn't
    // currently include it, so we pluck it safely here without leaking it to DOM.
    const {
      node: _node,
      inline,
      className,
      children,
      ...props
    } = rawProps as MarkdownCodeProps;
    const codeText = Array.isArray(children)
      ? children.join("")
      : String(children ?? "");
    const isInline =
      typeof inline === "boolean" ? inline : !codeText.includes("\n");

    if (isInline) {
      return (
        <code
          className={cn(
            "rounded bg-slate-100 px-1 py-0.5 font-mono text-[0.85em] text-slate-900",
            className,
          )}
          {...props}
        >
          {children}
        </code>
      );
    }

    // For fenced blocks, the parent <pre> handles the box styling.
    return (
      <code className={cn("font-mono", className)} {...props}>
        {children}
      </code>
    );
  },
};

const MARKDOWN_TABLE_COMPONENTS: Components = {
  table: ({ node: _node, className, ...props }) => (
    <div className="my-3 overflow-x-auto">
      <table className={cn("w-full border-collapse", className)} {...props} />
    </div>
  ),
  thead: ({ node: _node, className, ...props }) => (
    <thead className={cn("bg-slate-50", className)} {...props} />
  ),
  tbody: ({ node: _node, className, ...props }) => (
    <tbody className={cn("divide-y divide-slate-100", className)} {...props} />
  ),
  tr: ({ node: _node, className, ...props }) => (
    <tr className={cn("align-top", className)} {...props} />
  ),
  th: ({ node: _node, className, ...props }) => (
    <th
      className={cn(
        "border border-slate-200 px-3 py-2 text-left text-xs font-semibold",
        className,
      )}
      {...props}
    />
  ),
  td: ({ node: _node, className, ...props }) => (
    <td
      className={cn("border border-slate-200 px-3 py-2 align-top", className)}
      {...props}
    />
  ),
};

const MARKDOWN_COMPONENTS_BASIC: Components = {
  ...MARKDOWN_TABLE_COMPONENTS,
  ...MARKDOWN_CODE_COMPONENTS,
  p: ({ node: _node, className, ...props }) => (
    <p className={cn("mb-2 last:mb-0", className)} {...props} />
  ),
  ul: ({ node: _node, className, ...props }) => (
    <ul className={cn("mb-2 list-disc pl-5", className)} {...props} />
  ),
  ol: ({ node: _node, className, ...props }) => (
    <ol className={cn("mb-2 list-decimal pl-5", className)} {...props} />
  ),
  li: ({ node: _node, className, ...props }) => (
    <li className={cn("mb-1", className)} {...props} />
  ),
  strong: ({ node: _node, className, ...props }) => (
    <strong className={cn("font-semibold", className)} {...props} />
  ),
};

const MARKDOWN_COMPONENTS_DESCRIPTION: Components = {
  ...MARKDOWN_COMPONENTS_BASIC,
  p: ({ node: _node, className, ...props }) => (
    <p className={cn("mb-3 last:mb-0", className)} {...props} />
  ),
  h1: ({ node: _node, className, ...props }) => (
    <h1 className={cn("mb-2 text-base font-semibold", className)} {...props} />
  ),
  h2: ({ node: _node, className, ...props }) => (
    <h2 className={cn("mb-2 text-sm font-semibold", className)} {...props} />
  ),
  h3: ({ node: _node, className, ...props }) => (
    <h3 className={cn("mb-2 text-sm font-semibold", className)} {...props} />
  ),
};

const MARKDOWN_REMARK_PLUGINS_BASIC = [remarkGfm];
const MARKDOWN_REMARK_PLUGINS_WITH_BREAKS = [remarkGfm, remarkBreaks];

export type MarkdownVariant = "basic" | "comment" | "description";

export const Markdown = memo(function Markdown({
  content,
  variant,
}: {
  content: string;
  variant: MarkdownVariant;
}) {
  const trimmed = content.trim();
  const remarkPlugins =
    variant === "comment"
      ? MARKDOWN_REMARK_PLUGINS_WITH_BREAKS
      : MARKDOWN_REMARK_PLUGINS_BASIC;
  const components =
    variant === "description"
      ? MARKDOWN_COMPONENTS_DESCRIPTION
      : MARKDOWN_COMPONENTS_BASIC;
  return (
    <ReactMarkdown remarkPlugins={remarkPlugins} components={components}>
      {trimmed}
    </ReactMarkdown>
  );
});

Markdown.displayName = "Markdown";
