import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useChat } from "../hooks/useChat";

function sseResponse(events: [string, unknown][]): Response {
  const body = events.map(([e, d]) => `event: ${e}\ndata: ${JSON.stringify(d)}\n\n`).join("");
  return new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream" } });
}

const source = { n: 1, id: "kb:1", title: "Security", source: "security.md", url: "", text: "…", score: 0.03, components: { bm25: 1 }, metadata: {} };

afterEach(() => vi.restoreAllMocks());

describe("useChat", () => {
  it("streams a turn into a user + assistant message pair", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      sseResponse([
        ["status", { stage: "retrieving" }],
        ["meta", { conversation_id: "c".repeat(32), sources: [source], analysis: {}, retrieval: {}, enrichment: {} }],
        ["status", { stage: "generating" }],
        ["token", { text: "Report it " }],
        ["token", { text: "to security [1]." }],
        ["done", { conversation_id: "c".repeat(32), answer: "Report it to security [1].", citations: [1], grounded: true }],
      ]),
    );
    const onChange = vi.fn();
    const { result } = renderHook(() => useChat(onChange));

    await act(() => result.current.send("Where do I report phishing?", "general"));

    const [user, assistant] = result.current.messages;
    expect(user).toMatchObject({ role: "user", content: "Where do I report phishing?" });
    expect(assistant).toMatchObject({ role: "assistant", content: "Report it to security [1].", streaming: false });
    expect(assistant?.sources).toHaveLength(1);
    expect(assistant?.done?.grounded).toBe(true);
    expect(result.current.conversationId).toBe("c".repeat(32));
    expect(onChange).toHaveBeenCalledWith("c".repeat(32));

    const body = JSON.parse(fetchMock.mock.calls[0]![1]!.body as string);
    expect(body).toMatchObject({ question: "Where do I report phishing?", mode: "general", conversation_id: null });
  });

  it("sends the conversation id on follow-up turns", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
      sseResponse([
        ["meta", { conversation_id: "a".repeat(32), sources: [] }],
        ["done", { conversation_id: "a".repeat(32), answer: "ok", citations: [] }],
      ]),
    );
    const { result } = renderHook(() => useChat(() => undefined));
    await act(() => result.current.send("first", "medical"));
    await act(() => result.current.send("second", "medical"));
    const second = JSON.parse(fetchMock.mock.calls[1]![1]!.body as string);
    expect(second.conversation_id).toBe("a".repeat(32));
  });

  it("shows API errors on the assistant message", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: "Invalid or missing API key." }), { status: 401 }),
    );
    const { result } = renderHook(() => useChat(() => undefined));
    await act(() => result.current.send("hi", "general"));
    await waitFor(() => expect(result.current.messages[1]?.error).toBe("Invalid or missing API key."));
    expect(result.current.busy).toBe(false);
  });

  it("regenerate re-asks the last question and replaces the previous answer", async () => {
    let call = 0;
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
      call += 1;
      return sseResponse([
        ["meta", { conversation_id: "b".repeat(32), sources: [] }],
        ["done", { conversation_id: "b".repeat(32), message_id: call, answer: `answer ${call}`, citations: [] }],
      ]);
    });
    const { result } = renderHook(() => useChat(() => undefined));
    await act(() => result.current.send("question", "general"));
    await act(async () => result.current.regenerate());
    await waitFor(() => expect(result.current.messages[1]?.content).toBe("answer 2"));
    expect(result.current.messages).toHaveLength(2);
    expect(JSON.parse(fetchMock.mock.calls[1]![1]!.body as string)).toMatchObject({ question: "question", regenerate: true });
  });

  it("rating toggles feedback and clicking twice clears it", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (url) =>
      String(url).includes("/feedback")
        ? new Response("{}", { status: 200 })
        : sseResponse([
            ["meta", { conversation_id: "d".repeat(32), sources: [] }],
            ["done", { conversation_id: "d".repeat(32), message_id: 7, answer: "x", citations: [] }],
          ]),
    );
    const { result } = renderHook(() => useChat(() => undefined));
    await act(() => result.current.send("q", "general"));
    const key = result.current.messages[1]!.key;
    await act(() => result.current.rate(key, 7, 1));
    expect(result.current.messages[1]?.feedback).toBe(1);
    await act(() => result.current.rate(key, 7, 1));
    expect(result.current.messages[1]?.feedback).toBeNull();
    const bodies = fetchMock.mock.calls.filter(([u]) => String(u).includes("/feedback")).map(([, i]) => JSON.parse(i!.body as string));
    expect(bodies).toEqual([{ value: 1 }, { value: 0 }]);
  });
});
