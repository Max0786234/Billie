import { describe, expect, it } from "vitest";
import { SSEParser, readEventStream } from "../api/sse";

describe("SSEParser", () => {
  it("parses complete events", () => {
    const parser = new SSEParser();
    const events = parser.push('event: token\ndata: {"text":"Hi"}\n\nevent: status\ndata: {"stage":"generating"}\n\n');
    expect(events).toEqual([
      { event: "token", data: { text: "Hi" } },
      { event: "status", data: { stage: "generating" } },
    ]);
  });

  it("buffers events split across chunks at arbitrary positions", () => {
    const raw = 'event: token\ndata: {"text":"नमस्ते"}\n\nevent: done\ndata: {"answer":"x"}\n\n';
    for (let cut = 1; cut < raw.length; cut += 7) {
      const parser = new SSEParser();
      const events = [...parser.push(raw.slice(0, cut)), ...parser.push(raw.slice(cut))];
      expect(events.map((e) => e.event)).toEqual(["token", "done"]);
    }
  });

  it("handles CRLF line endings and ignores comments and malformed data", () => {
    const parser = new SSEParser();
    const events = parser.push(': keep-alive\r\n\r\nevent: token\r\ndata: {"text":"a"}\r\n\r\nevent: token\ndata: {bad\n\n');
    expect(events).toEqual([{ event: "token", data: { text: "a" } }]);
  });
});

describe("readEventStream", () => {
  it("decodes multi-byte characters split between network chunks", async () => {
    const bytes = new TextEncoder().encode('event: token\ndata: {"text":"ডায়াবেটিস"}\n\n');
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(bytes.slice(0, 30));
        controller.enqueue(bytes.slice(30));
        controller.close();
      },
    });
    const events = [];
    for await (const event of readEventStream(body)) events.push(event);
    expect(events).toEqual([{ event: "token", data: { text: "ডায়াবেটিস" } }]);
  });
});
