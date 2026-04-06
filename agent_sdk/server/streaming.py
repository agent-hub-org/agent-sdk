"""
Shared streaming utilities for agent SSE responses.

Provides:
- ``StreamingMathFixer``: wraps an async chunk iterator and converts LaTeX
  math delimiters on-the-fly so the UI renders math correctly without waiting
  for the full response.
- ``_fix_math_delimiters``: post-process helper for non-streaming responses.
"""

import re


def _fix_math_delimiters(text: str) -> str:
    r"""Convert LaTeX delimiters to Markdown math notation (post-processing).

    \[...\]  →  $$\n...\n$$   (display / block math)
    \(...\)  →  $...$          (inline math)
    """
    # Display math — must run before inline to avoid overlap
    text = re.sub(r'\\\[(.*?)\\\]', lambda m: f'$$\n{m.group(1)}\n$$', text, flags=re.DOTALL)
    # Inline math
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text)
    return text


class StreamingMathFixer:
    """Wraps an async chunk stream and converts \\(...\\) / \\[...\\] math delimiters on-the-fly.

    Non-math text is yielded immediately so the streaming feel is preserved.
    Math sections are buffered only until their closing delimiter arrives,
    then emitted with the correct $...$ / $$...$$ notation.

    Usage::

        stream = StreamingMathFixer(agent.astream(...))
        async for chunk in stream:
            yield f"data: {json.dumps({'text': chunk})}\\n\\n"
    """

    def __init__(self, source) -> None:
        self._source = source

    async def __aiter__(self):
        buffer = ""
        in_math = False   # inside \( ... \)
        in_block = False  # inside \[ ... \]

        async for chunk in self._source:
            buffer += chunk
            result = ""

            while buffer:
                if not in_math and not in_block:
                    bi = buffer.find("\\[")
                    ii = buffer.find("\\(")
                    if bi == -1 and ii == -1:
                        # Only hold back a lone trailing backslash that might start a delimiter
                        if buffer.endswith("\\"):
                            if len(buffer) > 1:
                                result += buffer[:-1]
                                buffer = "\\"
                            break
                        else:
                            result += buffer
                            buffer = ""
                            break
                    if bi == -1 or (ii != -1 and ii < bi):
                        result += buffer[:ii]
                        buffer = buffer[ii + 2:]
                        in_math = True
                    else:
                        result += buffer[:bi]
                        buffer = buffer[bi + 2:]
                        in_block = True
                elif in_math:
                    close = buffer.find("\\)")
                    if close == -1:
                        break  # wait for more chunks
                    result += "$" + buffer[:close] + "$"
                    buffer = buffer[close + 2:]
                    in_math = False
                else:  # in_block
                    close = buffer.find("\\]")
                    if close == -1:
                        break  # wait for more chunks
                    result += "$$\n" + buffer[:close] + "\n$$"
                    buffer = buffer[close + 2:]
                    in_block = False

            if result:
                yield result

        # Flush any remaining buffer after the source stream ends
        if buffer:
            if in_math:
                yield "$" + buffer + "$"
            elif in_block:
                yield "$$\n" + buffer + "\n$$"
            else:
                yield buffer

    @property
    def steps(self):
        """Delegate .steps access to the underlying stream (used by save hooks)."""
        return self._source.steps
