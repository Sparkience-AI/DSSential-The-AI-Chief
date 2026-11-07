# Privacy and data flow

CSV bytes are parsed in memory and bounded to 10 MiB, 250,000 rows, and 250 columns. The source
snapshot is fingerprinted and every analysis receives a deep copy. DSSential does not include
analytics, tracking, accounts, or application-managed storage.

Deterministic forecasting and scenario calculations are local. Optional narration sends only the
exact JSON summary displayed in the interface to the selected Gemini, OpenAI, or Anthropic API
after explicit consent. Raw rows and the API key are excluded from that summary. Keys live only in
Streamlit session state and are never written to environment variables or files by DSSential.
