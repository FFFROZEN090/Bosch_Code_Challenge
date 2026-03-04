"""UI configuration: page settings, demo questions, and welcome message."""

PAGE_TITLE = "ME Engineering Assistant"
PAGE_ICON = "\u2699\ufe0f"  # gear emoji
LAYOUT = "wide"

DEMO_QUESTIONS = [
    {
        "question": "What is the maximum operating temperature for the ECU-750?",
        "label": "ECU-750 operating temperature",
        "route_hint": "ECU_700",
    },
    {
        "question": "How much RAM does the ECU-850 have?",
        "label": "ECU-850 RAM specs",
        "route_hint": "ECU_800",
    },
    {
        "question": "What are the differences between ECU-850 and ECU-850b?",
        "label": "Compare ECU-850 vs ECU-850b",
        "route_hint": "COMPARE",
    },
    {
        "question": "What are the AI capabilities of the ECU-850b?",
        "label": "ECU-850b AI capabilities",
        "route_hint": "ECU_800",
    },
    {
        "question": "Which ECU models support Over-the-Air (OTA) updates?",
        "label": "OTA update support",
        "route_hint": "COMPARE",
    },
]

WELCOME_MESSAGE = (
    "Welcome! I'm the **ME Engineering Assistant**. "
    "I can answer technical questions about ECU product specifications.\n\n"
    "I have access to documentation for:\n"
    "- **ECU-750** (700 Series)\n"
    "- **ECU-850** (800 Series, Base)\n"
    "- **ECU-850b** (800 Series, AI Enhanced)\n\n"
    "Ask me anything, or try an example question from the sidebar."
)
