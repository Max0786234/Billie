import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { ModeOverview } from "../api/types";
import { ModeGuide } from "../components/ModeGuide";

const medical: ModeOverview = {
  corpus: "medquad",
  passages: 27582,
  vectors: 16407,
  questions: 16407,
  topics: 5126,
  sources: [{ name: "MedlinePlus", count: 981 }],
  question_types: [
    { name: "symptoms", count: 2748 },
    { name: "treatment", count: 2442 },
  ],
  good_for: ["Symptoms, causes and risk factors"],
  not_covered: ["Drug doses and medication-specific advice"],
  examples: ["Is Huntington disease inherited?"],
};

describe("ModeGuide", () => {
  it("shows what the mode contains, what it can't answer, and working examples", () => {
    const onPick = vi.fn();
    render(<ModeGuide mode="medical" overview={medical} onPick={onPick} />);
    expect(screen.getByText(/16,407 expert answers on 5,126 health topics/)).toBeInTheDocument();
    expect(screen.getByText("Symptoms")).toBeInTheDocument(); // question types as readable chips
    expect(screen.getByText("Drug doses and medication-specific advice")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Is Huntington disease inherited?"));
    expect(onPick).toHaveBeenCalledWith("Is Huntington disease inherited?");
  });

  it("falls back to built-in examples when the overview is not loaded", () => {
    render(<ModeGuide mode="research" onPick={() => undefined} />);
    expect(screen.getAllByRole("button").length).toBeGreaterThan(0);
  });
});
