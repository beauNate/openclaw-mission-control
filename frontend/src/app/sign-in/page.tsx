import { redirect } from "next/navigation";
import { auth } from "@clerk/nextjs/server";

export default function SignInPage() {
  const { userId, redirectToSignIn } = auth();

  if (userId) {
    redirect("/activity");
  }

  // Top-level redirect to Clerk hosted sign-in.
  // Cypress E2E cannot reliably drive Clerk modal/iframe login.
  return redirectToSignIn({ returnBackUrl: "/activity" });
}
