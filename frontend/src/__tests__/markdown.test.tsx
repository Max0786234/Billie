import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { Markdown, linkCitations } from "../components/Markdown";

describe("linkCitations", () => {
  it("links bare citations but leaves real Markdown links alone", () => {
    expect(linkCitations("Fact [1][2]. See [docs](https://x.y) and [3](https://z)."))
      .toBe("Fact [1](#cite-1)[2](#cite-2). See [docs](https://x.y) and [3](https://z).");
  });
});

describe("Markdown", () => {
  it("renders citation chips that report the clicked source", () => {
    const onCite = vi.fn();
    render(<Markdown text="Glaucoma damages the optic nerve [2]." available={3} onCite={onCite} />);
    fireEvent.click(screen.getByRole("button", { name: "2" }));
    expect(onCite).toHaveBeenCalledWith(2);
  });

  it("marks citations that do not match a source as invalid", () => {
    const onCite = vi.fn();
    render(<Markdown text="Made up [9]." available={2} onCite={onCite} />);
    const chip = screen.getByRole("button", { name: "9" });
    expect(chip).toHaveAttribute("title", "This citation does not match any source");
    fireEvent.click(chip);
    expect(onCite).not.toHaveBeenCalled();
  });

  it("renders lists and bold text from the model", () => {
    render(<Markdown text={"- **BM25** keyword search [1]\n- dense search [2]"} available={2} />);
    expect(screen.getByText("BM25").tagName).toBe("STRONG");
    expect(screen.getAllByRole("listitem")).toHaveLength(2);
  });
});
