"use client";

import { memo, useCallback, useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

type BoardChatComposerProps = {
  placeholder?: string;
  isSending?: boolean;
  onSend: (content: string) => Promise<boolean>;
};

function BoardChatComposerImpl({
  placeholder = "Message the board lead. Tag agents with @name.",
  isSending = false,
  onSend,
}: BoardChatComposerProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const shouldFocusAfterSendRef = useRef(false);

  useEffect(() => {
    if (isSending) return;
    if (!shouldFocusAfterSendRef.current) return;
    shouldFocusAfterSendRef.current = false;
    textareaRef.current?.focus();
  }, [isSending]);

  const send = useCallback(async () => {
    if (isSending) return;
    const trimmed = value.trim();
    if (!trimmed) return;
    const ok = await onSend(trimmed);
    shouldFocusAfterSendRef.current = true;
    if (ok) {
      setValue("");
    }
  }, [isSending, onSend, value]);

  return (
    <div className="mt-4 space-y-2">
      <Textarea
        ref={textareaRef}
        value={value}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={(event) => {
          if (event.key !== "Enter") return;
          if (event.nativeEvent.isComposing) return;
          if (event.shiftKey) return;
          event.preventDefault();
          void send();
        }}
        placeholder={placeholder}
        className="min-h-[120px]"
        disabled={isSending}
      />
      <div className="flex justify-end">
        <Button
          onClick={() => void send()}
          disabled={isSending || !value.trim()}
        >
          {isSending ? "Sending…" : "Send"}
        </Button>
      </div>
    </div>
  );
}

export const BoardChatComposer = memo(BoardChatComposerImpl);
BoardChatComposer.displayName = "BoardChatComposer";
