#!/bin/bash
echo "Launching app.py..."
export GEMINI_API_KEY="AIzaSyCkAKzzGo9741SJkxtuCcF8ofQ9PEVrREc"
streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.maxUploadSize=100 --browser.gatherUsageStats=false