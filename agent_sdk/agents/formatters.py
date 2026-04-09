from typing import Dict

# Standard response formats used across the agent hub
RESPONSE_FORMAT_INSTRUCTIONS = {
    "summary": (
        "\n\nRESPONSE FORMAT OVERRIDE: The user wants a QUICK SUMMARY. "
        "Keep your response concise — 5-7 bullet points maximum covering the key metrics, "
        "one-line interpretation of each, and a 2-sentence bottom line. "
        "Skip detailed section headers, long explanations, and tables."
    ),
    "flash_cards": (
        "\n\nRESPONSE FORMAT OVERRIDE: The user wants INSIGHT CARDS. "
        "Format your response as a series of insight cards using this EXACT format for each card:\n\n"
        "### [Topic Label]\n"
        "**Key Insight:** [The main metric, data point, or finding — keep it short and prominent]\n"
        "[1-2 sentence explanation of this means and why it matters]\n\n"
        "STRICT FORMATTING RULES:\n"
        "- Use exactly ### (three hashes) for each card topic — NOT ## or ####\n"
        "- Do NOT wrap topic names in **bold** — just plain text after ###\n"
        "- Do NOT use bullet points (- or *) for the Key Insight line — start it directly with **Key Insight:**\n"
        "- Every card MUST have a **Key Insight:** line\n"
        "- Start directly with the first ### card — no title header, preamble, or introductory text before the cards\n\n"
        "Generate 8-12 cards covering: key financial metrics, valuation, strengths, risks, "
        "growth prospects, and a bottom line view."
    ),
    "detailed": "",  # default — uses the full system prompt format as-is
}
