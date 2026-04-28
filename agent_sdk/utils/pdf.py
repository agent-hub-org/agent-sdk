"""Shared Markdown-to-PDF renderer used by all agent report tools."""
import re


_UNICODE_TO_ASCII = str.maketrans({
    # Greek letters (lowercase)
    "α": "alpha", "β": "beta",  "γ": "gamma", "δ": "delta",
    "ε": "epsilon","ζ": "zeta", "η": "eta",   "θ": "theta",
    "λ": "lambda", "μ": "mu",   "ξ": "xi",   "π": "pi",
    "σ": "sigma",  "τ": "tau",  "φ": "phi",  "χ": "chi",
    "ψ": "psi",    "ω": "omega",
    # Greek letters (uppercase)
    "Γ": "Gamma", "Δ": "Delta", "Θ": "Theta", "Λ": "Lambda",
    "Σ": "Sigma",  "Φ": "Phi",  "Ψ": "Psi",  "Ω": "Omega",
    # Math operators / symbols
    "∑": "sum",   "∏": "prod",  "∫": "integral",
    "∈": "in",    "∉": "not in","⊂": "subset","⊆": "subset=",
    "∪": "union", "∩": "intersect",
    "≤": "<=",    "≥": ">=",   "≠": "!=",
    "→": "->",    "←": "<-",   "↔": "<->",
    "∞": "inf",   "∂": "d",    "∇": "nabla",
    "⊤": "^T",    "⊥": "perp",
    "·": ".",     "•": "*",
    # Currency / typography
    "₹": "Rs.", "€": "EUR", "£": "GBP",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "—": "--", "–": "-",
})


def slugify(text: str, max_len: int = 60) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text[:max_len]


def sanitize_for_pdf(text: str) -> str:
    """Transliterate Unicode and strip LaTeX math delimiters for Helvetica."""
    text = re.sub(r'\$\$(.*?)\$\$', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\$(.*?)\$', r'\1', text)
    return text.translate(_UNICODE_TO_ASCII)


class MarkdownPDFRenderer:
    """Convert a markdown string to a PDF bytes object using fpdf2.

    Supports: # / ## / ### / #### headings, bullet lists (- / *),
    blockquotes (> ), bold (**text**), **Day N** patterns, table rows (|…|),
    and blank-line paragraph spacing.

    Usage::

        renderer = MarkdownPDFRenderer()
        pdf_bytes = renderer.render("## Hello\\n- item 1", title="My Doc")
    """

    def render(self, markdown_content: str, title: str) -> bytes:
        from fpdf import FPDF

        title = sanitize_for_pdf(title)
        markdown_content = sanitize_for_pdf(markdown_content)

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(8)

        pdf.set_font("Helvetica", size=11)
        for line in markdown_content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("# "):
                pdf.set_font("Helvetica", "B", 16)
                pdf.ln(4)
                pdf.multi_cell(0, 10, stripped[2:])
                pdf.set_font("Helvetica", size=11)
            elif stripped.startswith("## "):
                pdf.set_font("Helvetica", "B", 14)
                pdf.ln(3)
                pdf.multi_cell(0, 9, stripped[3:])
                pdf.set_font("Helvetica", size=11)
            elif stripped.startswith("### "):
                pdf.set_font("Helvetica", "B", 12)
                pdf.ln(2)
                pdf.multi_cell(0, 8, stripped[4:])
                pdf.set_font("Helvetica", size=11)
            elif stripped.startswith("#### "):
                pdf.set_font("Helvetica", "BI", 11)
                pdf.ln(2)
                pdf.multi_cell(0, 7, stripped[5:])
                pdf.set_font("Helvetica", size=11)
            elif stripped.startswith("**Day ") or stripped.startswith("- Day "):
                pdf.set_font("Helvetica", "B", 11)
                plain = re.sub(r"\*\*(.*?)\*\*", r"\1", stripped.lstrip("- "))
                pdf.multi_cell(0, 7, plain)
                pdf.set_font("Helvetica", size=11)
            elif stripped.startswith("> "):
                quote = re.sub(r"\*\*(.*?)\*\*", r"\1", stripped[2:])
                quote = re.sub(r"\*(.*?)\*", r"\1", quote)
                pdf.set_fill_color(230, 240, 255)
                pdf.set_font("Helvetica", "I", 10)
                pdf.set_x(14)
                pdf.multi_cell(180, 6, quote, fill=True)
                pdf.set_font("Helvetica", size=11)
                pdf.ln(1)
            elif stripped.startswith("- ") or stripped.startswith("* "):
                bullet = re.sub(r"\*\*(.*?)\*\*", r"\1", stripped[2:])
                bullet = re.sub(r"\*(.*?)\*", r"\1", bullet)
                pdf.set_x(14)
                pdf.multi_cell(0, 6, f"• {bullet}")
            elif stripped.startswith("|"):
                plain = re.sub(r"\*\*(.*?)\*\*", r"\1", stripped)
                plain = re.sub(r"\*(.*?)\*", r"\1", plain)
                pdf.set_font("Courier", size=9)
                pdf.multi_cell(0, 5, plain)
                pdf.set_font("Helvetica", size=11)
            elif stripped:
                plain = re.sub(r"\*\*(.*?)\*\*", r"\1", stripped)
                plain = re.sub(r"\*(.*?)\*", r"\1", plain)
                pdf.multi_cell(0, 6, plain)
            else:
                pdf.ln(3)

        return pdf.output()
