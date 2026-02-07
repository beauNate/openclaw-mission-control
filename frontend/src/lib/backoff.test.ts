import { describe, expect, it, vi } from "vitest";

import { createExponentialBackoff } from "./backoff";

describe("createExponentialBackoff", () => {
  it("increments attempt and clamps delay", () => {
    vi.spyOn(Math, "random").mockReturnValue(0);

    const backoff = createExponentialBackoff({
      baseMs: 100,
      factor: 2,
      maxMs: 250,
      jitter: 0,
    });

    expect(backoff.attempt()).toBe(0);
    expect(backoff.nextDelayMs()).toBe(100);
    expect(backoff.attempt()).toBe(1);
    expect(backoff.nextDelayMs()).toBe(200);
    expect(backoff.nextDelayMs()).toBe(250); // capped
  });

  it("reset brings attempt back to zero", () => {
    vi.spyOn(Math, "random").mockReturnValue(0);

    const backoff = createExponentialBackoff({ baseMs: 100, jitter: 0 });
    backoff.nextDelayMs();
    expect(backoff.attempt()).toBe(1);

    backoff.reset();
    expect(backoff.attempt()).toBe(0);
  });
});
