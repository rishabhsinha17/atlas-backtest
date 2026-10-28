"""Network adapters / local loaders. Each provider normalizes to the canonical
frames of docs/ARCHITECTURE.md §3 and writes into the ParquetStore.

Key-gated providers (alpaca, polygon, tiingo, databento, alphavantage) are
inactive unless their API key environment variables are set; `available()`
returns False and nothing is fetched.
"""
