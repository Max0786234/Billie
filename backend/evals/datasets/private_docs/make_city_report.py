"""Generate riverton_annual_report_2025.pdf: a long, realistic test document with known facts.

Facts sit in prose and in tables, spread across ~25 pages of plausible filler that shares
vocabulary with them (so search has real distractors). Every question in questions.json has
its answer here. Run with any Python that has reportlab:  python make_city_report.py
"""

from __future__ import annotations

import random
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).with_name("riverton_annual_report_2025.pdf")
ss = getSampleStyleSheet()
rng = random.Random(7)

TOPICS = {
    "finance": [
        "the annual budget",
        "capital works",
        "revenue collection",
        "grants from the state",
        "audited accounts",
    ],
    "water": ["pipelines", "metering", "pumping stations", "water quality testing", "storage reservoirs"],
    "transport": ["bus depots", "road safety", "signal timing", "footpaths", "parking management"],
    "health": [
        "primary care",
        "disease surveillance",
        "community health workers",
        "maternal care",
        "school health checks",
    ],
    "education": ["teacher training", "classroom repairs", "midday meals", "digital classrooms", "school enrolment"],
    "sanitation": ["door-to-door collection", "composting", "street sweeping", "public toilets", "drain cleaning"],
    "environment": ["urban forests", "lake restoration", "air monitoring", "heat action plans", "tree census"],
    "housing": ["slum upgrading", "rental housing", "building permits", "land records", "redevelopment"],
    "digital": ["online services", "the citizen app", "data centres", "cyber security", "service delivery"],
}
VERBS = ["improved", "expanded", "reviewed", "strengthened", "modernised", "monitored", "planned", "upgraded"]
OPENERS = [
    "During the year,",
    "In 2025,",
    "Across the city,",
    "With support from residents,",
    "As part of the council's plan,",
]


def filler(topic: str, n: int) -> list[str]:
    """Plausible, number-free paragraphs that share vocabulary with the facts (search distractors)."""
    words = TOPICS[topic]
    out = []
    for _ in range(n):
        sentences = []
        for _ in range(rng.randint(4, 6)):
            a, b = rng.sample(words, 2)
            sentences.append(
                f"{rng.choice(OPENERS)} the council {rng.choice(VERBS)} {a} and coordinated with ward offices on {b}, "
                f"so that residents saw steady progress and clearer communication about timelines."
            )
        out.append(" ".join(sentences))
    return out


def table(rows: list[list[str]]) -> Table:
    t = Table(rows, hAlign="LEFT")
    t.setStyle(
        TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)])
    )
    return t


story: list = [Paragraph("Riverton City Council: Annual Report 2025", ss["Title"])]


def section(title: str, facts: list[str], topic: str, filler_paras: int, extra=None) -> None:
    story.append(Paragraph(title, ss["Heading1"]))
    for para in facts:
        story.append(Paragraph(para, ss["BodyText"]))
        story.append(Spacer(1, 6))
    if extra is not None:
        story.append(extra)
        story.append(Spacer(1, 8))
    for para in filler(topic, filler_paras):
        story.append(Paragraph(para, ss["BodyText"]))
        story.append(Spacer(1, 6))
    story.append(PageBreak())


section(
    "Mayor's foreword",
    [
        "This report is presented by Mayor Anjali Deshmukh, who was elected in 2023 and leads a council of 38 elected ward members.",
        "Riverton has a population of 1,284,000 people living across 38 wards and an area of 412 square kilometres.",
    ],
    "finance",
    5,
)
section(
    "Finance",
    [
        "The council's total budget for 2025 was 9,860 crore rupees. The table below shows the allocation by department.",
    ],
    "finance",
    7,
    table(
        [
            ["Department", "Budget 2025 (crore rupees)"],
            ["Water supply", "1,240"],
            ["Roads", "2,115"],
            ["Health", "860"],
            ["Education", "1,430"],
            ["Sanitation", "980"],
            ["Parks", "310"],
            ["Public transport", "1,725"],
            ["Administration", "1,200"],
        ]
    ),
)
section(
    "Water supply",
    [
        "The city supplied an average of 640 million litres of water per day in 2025.",
        "Water lost to leakage fell from 31 percent in 2024 to 24 percent in 2025 after 210 kilometres of old pipes were replaced.",
        "A new water treatment plant at Kharadi, with a capacity of 120 million litres per day, opened on 14 August 2025.",
    ],
    "water",
    7,
)
section(
    "Roads and public transport",
    [
        "The council resurfaced 86 kilometres of roads and added 220 new electric buses to the city fleet.",
        "The Metro Line 2 extension of 9.4 kilometres will have 7 new stations and is expected to open in March 2027.",
        "Average daily bus ridership by quarter is shown below.",
    ],
    "transport",
    7,
    table(
        [
            ["Quarter", "Average daily riders"],
            ["Q1", "312,000"],
            ["Q2", "298,000"],
            ["Q3", "341,000"],
            ["Q4", "365,000"],
        ]
    ),
)
section(
    "Public health",
    [
        "The city runs 14 urban health centres. Reported dengue cases fell from 4,820 in 2024 to 2,910 in 2025 after a "
        "city-wide fogging and larvae-control drive.",
        "Childhood vaccination coverage in 2025 is shown below.",
    ],
    "health",
    7,
    table(
        [
            ["Vaccine", "Coverage (%)"],
            ["Measles", "94"],
            ["Polio", "97"],
            ["DPT", "91"],
            ["Hepatitis B", "89"],
        ]
    ),
)
section(
    "Education",
    [
        "The council runs 412 municipal schools with 186,000 students and a pupil-teacher ratio of 28 to 1.",
        "New STEM laboratories were opened in 60 schools during the year.",
    ],
    "education",
    7,
)
section(
    "Sanitation and waste",
    [
        "The city generates about 2,300 tonnes of solid waste per day, and 71 percent of it was segregated at source in 2025.",
        "Three biogas plants now process wet waste, and the remaining waste goes to the Uruli landfill.",
    ],
    "sanitation",
    7,
)
section(
    "Parks and environment",
    [
        "Riverton has 1,150 hectares of green space. Volunteers and the council planted 48,000 trees in 2025.",
        "The annual average air quality index improved from 104 in 2024 to 92 in 2025.",
    ],
    "environment",
    7,
)
section(
    "Housing",
    [
        "The council completed 12,400 affordable homes under its housing programme in 2025.",
    ],
    "housing",
    6,
)
section(
    "Digital services and complaints",
    [
        "The RivertonOne citizen app passed 2.1 million downloads, and 63 percent of property tax was paid online in 2025.",
        "The council employs 24,600 staff. Complaints received by department are shown below.",
    ],
    "digital",
    6,
    table(
        [
            ["Department", "Complaints received"],
            ["Roads", "22,410"],
            ["Sanitation", "18,240"],
            ["Water supply", "15,930"],
            ["Public transport", "9,880"],
            ["Health", "4,120"],
        ]
    ),
)
section(
    "Outlook for 2026",
    [
        "The council's four priorities for 2026 are: opening the Metro Line 2 extension on schedule, reducing water leakage "
        "below 20 percent, raising waste segregation to 85 percent, and adding 150 more electric buses.",
    ],
    "finance",
    5,
)

SimpleDocTemplate(str(OUT), pagesize=A4, title="Riverton Annual Report 2025").build(story)
print(f"wrote {OUT}")
