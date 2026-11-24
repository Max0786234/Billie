import { BookOpen, GraduationCap, HeartPulse, type LucideIcon } from "lucide-react";
import type { Mode, Stage } from "../api/types";

export interface ModeInfo {
  id: Mode;
  label: string;
  icon: LucideIcon;
  tagline: string;
  corpus: string;
  accent: { text: string; bg: string; ring: string; soft: string };
  examples: string[];
}

export const MODES: Record<Mode, ModeInfo> = {
  general: {
    id: "general",
    label: "General",
    icon: BookOpen,
    tagline: "Ask about Billie and your own documents.",
    corpus: "knowledge base",
    accent: {
      text: "text-indigo-600 dark:text-indigo-400",
      bg: "bg-indigo-600",
      ring: "ring-indigo-500/30",
      soft: "bg-indigo-50 text-indigo-700 dark:bg-indigo-950/60 dark:text-indigo-300",
    },
    examples: [
      "How does Billie decide which documents to use?",
      "What happens to my data if I use a cloud model?",
      "How do I add my own documents?",
    ],
  },
  medical: {
    id: "medical",
    label: "Medical",
    icon: HeartPulse,
    tagline: "Health information from 16,000+ NIH question-answer pairs (MedQuAD).",
    corpus: "MedQuAD",
    accent: {
      text: "text-teal-600 dark:text-teal-400",
      bg: "bg-teal-600",
      ring: "ring-teal-500/30",
      soft: "bg-teal-50 text-teal-700 dark:bg-teal-950/60 dark:text-teal-300",
    },
    examples: [
      "What are the early symptoms of glaucoma?",
      "How is type 2 diabetes treated?",
      "मधुमेह के लक्षण क्या हैं?",
    ],
  },
  research: {
    id: "research",
    label: "Research",
    icon: GraduationCap,
    tagline: "Explore ~980,000 computer-science papers from arXiv.",
    corpus: "arXiv",
    accent: {
      text: "text-amber-600 dark:text-amber-400",
      bg: "bg-amber-600",
      ring: "ring-amber-500/30",
      soft: "bg-amber-50 text-amber-800 dark:bg-amber-950/60 dark:text-amber-300",
    },
    examples: [
      "What are recent approaches to reduce hallucination in RAG?",
      "Compare graph neural networks for molecule property prediction",
      "Summarise work on efficient attention for long documents",
    ],
  },
};

export const MODE_LIST = Object.values(MODES);

export function stageLabel(stage: Stage, mode: Mode): string {
  switch (stage) {
    case "reading_image":
      return "Reading your image…";
    case "translating":
      return "Translating your question for search…";
    case "retrieving":
      return `Searching ${MODES[mode].corpus}…`;
    case "generating":
      return "Writing the answer…";
  }
}
