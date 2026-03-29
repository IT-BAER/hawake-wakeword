#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
exec python -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0
