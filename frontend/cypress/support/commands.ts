/// <reference types="cypress" />

type ClerkOtpLoginOptions = {
  clerkOrigin: string;
  email: string;
  otp: string;
};

function requireEnv(name: string): string {
  const value = Cypress.env(name) as string | undefined;
  if (!value) {
    throw new Error(
      `Missing Cypress env var ${name}. ` +
        `Set it via CYPRESS_${name}=... in CI/local before running Clerk login tests.`,
    );
  }
  return value;
}

function normalizeOrigin(value: string): string {
  try {
    const url = new URL(value);
    return url.origin;
  } catch {
    // allow providing just an origin-like string
    return value.replace(/\/$/, "");
  }
}

Cypress.Commands.add("loginWithClerkOtp", () => {
  const clerkOrigin = normalizeOrigin(requireEnv("CLERK_ORIGIN"));
  const email = requireEnv("CLERK_TEST_EMAIL");
  const otp = requireEnv("CLERK_TEST_OTP");

  const opts: ClerkOtpLoginOptions = { clerkOrigin, email, otp };

  // Navigate to a dedicated sign-in route that performs a top-level redirect
  // to Clerk hosted sign-in (avoids modal/iframe limitations in Cypress).
  cy.visit("/sign-in");

  // The Clerk UI is typically hosted on a different origin (clerk.accounts.dev / clerk.com).
  // Use cy.origin to drive the UI in Chrome.
  cy.origin(
    opts.clerkOrigin,
    { args: { email: opts.email, otp: opts.otp } },
    ({ email, otp }) => {
      // Email / identifier input
      cy.get('input[type="email"], input[name="identifier"], input[autocomplete="email"]', {
        timeout: 20_000,
      })
        .first()
        .clear()
        .type(email, { delay: 10 });

      // Submit / continue
      cy.get('button[type="submit"], button')
        .contains(/continue|sign in|send|next/i)
        .click({ force: true });

      // OTP input - Clerk commonly uses autocomplete=one-time-code
      cy.get('input[autocomplete="one-time-code"], input[name*="code"], input[inputmode="numeric"]', {
        timeout: 20_000,
      })
        .first()
        .clear()
        .type(otp, { delay: 10 });

      // Final submit (some flows auto-submit)
      cy.get("body").then(($body) => {
        const hasSubmit = $body
          .find('button[type="submit"], button')
          .toArray()
          .some((el) => /verify|continue|sign in|confirm/i.test(el.textContent || ""));
        if (hasSubmit) {
          cy.get('button[type="submit"], button')
            .contains(/verify|continue|sign in|confirm/i)
            .click({ force: true });
        }
      });
    },
  );
});

declare global {
  // eslint-disable-next-line @typescript-eslint/no-namespace
  namespace Cypress {
    interface Chainable {
      /**
       * Logs in via the real Clerk modal using deterministic OTP credentials.
       *
       * Requires env vars:
       * - CYPRESS_CLERK_ORIGIN (e.g. https://<subdomain>.clerk.accounts.dev)
       * - CYPRESS_CLERK_TEST_EMAIL
       * - CYPRESS_CLERK_TEST_OTP
       */
      loginWithClerkOtp(): Chainable<void>;
    }
  }
}

export {};
