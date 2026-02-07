"use client";

import { ClerkProvider } from "@clerk/nextjs";
import type { ReactNode } from "react";

function isLikelyValidClerkPublishableKey(
  key: string | undefined,
): key is string {
  if (!key) return false;
  // Clerk publishable keys look like: pk_test_... or pk_live_...
  // In CI we want builds to stay secretless; if the key isn't present/valid,
  // we skip Clerk entirely so `next build` can prerender.
  //
  // Note: Clerk appears to validate key *contents*, not just shape. We therefore
  // use a conservative heuristic to avoid treating obvious placeholders as valid.
  const m = /^pk_(test|live)_([A-Za-z0-9]+)$/.exec(key);
  if (!m) return false;
  const body = m[2];
  if (body.length < 16) return false;
  if (/^0+$/.test(body)) return false;
  return true;
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const publishableKey = process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY;

  if (!isLikelyValidClerkPublishableKey(publishableKey)) {
    return <>{children}</>;
  }

  return (
    <ClerkProvider publishableKey={publishableKey}>{children}</ClerkProvider>
  );
}
