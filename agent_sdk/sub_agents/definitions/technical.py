from agent_sdk.sub_agents.base import SubAgent, SubAgentOutput

_SYSTEM_PROMPT = """\
You are a Technical Analysis Specialist. Evaluate price action, momentum, and risk metrics.

The company ticker is provided in the context from the Company Profiling agent.

Use tools in this order:
1. get_price_series(ticker, period="1y") — daily closes for indicator computation
2. calculate_technical_signals(closes=[...]) — RSI, MACD, SMA50/200, support/resistance
3. get_historical_ohlcv(ticker, period="1y") — multi-timeframe returns, volume analysis

Output a structured summary then end with a JSON block:
```json
{
  "trend": "<bullish|bearish|sideways>",
  "rsi_14": <float>,
  "above_sma200": <bool>,
  "macd_signal": "<buy|sell|neutral>",
  "support": <float>,
  "resistance": <float>,
  "return_1m_pct": <float>,
  "return_3m_pct": <float>,
  "entry_zone": "<description>"
}
```
"""

DEFINITION = SubAgent(
    name="technical",
    model_id="azure/gpt-5-nano",
    tools=[
        "get_price_series",
        "calculate_technical_signals",
        "get_historical_ohlcv",
    ],
    system_prompt=_SYSTEM_PROMPT,
    output_schema=SubAgentOutput,
    reads_from=["company_profiling"],
    writes_to="technical",
    cache_ttl=15 * 60,
)
