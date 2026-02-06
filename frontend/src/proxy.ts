import { NextResponse } from "next/server";
import { clerkMiddleware } from "@clerk/nextjs/server";

const isClerkEnabled = () => Boolean(process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY);

export default isClerkEnabled() ? clerkMiddleware() : () => NextResponse.next();

export const config = {
  matcher: [
    "/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)",
    "/(api|trpc)(.*)",
  ],
};
