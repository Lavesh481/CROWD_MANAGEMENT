#!/bin/bash

# Crowd Management Deployment Script

echo "🚀 Crowd Management Deployment Helper"
echo "======================================"

if [ "$1" = "local" ]; then
    echo "📱 Switching to LOCAL development (full AI features)..."
    cp requirements-full.txt requirements.txt 2>/dev/null || {
        echo "Creating full requirements for local development..."
        cat > requirements.txt << 'EOF'
# Full requirements for local development
streamlit
opencv-python-headless
numpy
requests
python-dotenv
# AI dependencies for crowd detection
torch
torchvision
ultralytics
mediapipe
EOF
    }
    echo "✅ Local requirements set! Run: streamlit run app.py"
    echo "🎯 Features: Real crowd detection, webcam, AI models"
    
elif [ "$1" = "cloud" ]; then
    echo "☁️ Switching to CLOUD deployment (minimal dependencies)..."
    cp requirements-cloud.txt requirements.txt
    echo "✅ Cloud requirements set! Ready for Streamlit Cloud deployment"
    echo "⚠️ Features: UI works, but shows 0 people (dummy detection)"
    
else
    echo "Usage: $0 [local|cloud]"
    echo ""
    echo "  local  - Full AI features for local development"
    echo "  cloud  - Minimal dependencies for Streamlit Cloud"
    echo ""
    echo "Examples:"
    echo "  $0 local   # Switch to full features"
    echo "  $0 cloud   # Switch to cloud deployment"
fi
