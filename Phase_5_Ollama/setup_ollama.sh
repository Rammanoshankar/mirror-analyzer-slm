#!/bin/bash

# Mirror Analyzer - Phase 5: Setup Ollama + Phi-3
# This script installs Ollama and downloads Phi-3 Mini model

echo "======================================================================"
echo "Mirror Analyzer - Phase 5: Ollama Setup"
echo "======================================================================"
echo ""

# Check if Ollama is installed
if command -v ollama &> /dev/null; then
    echo "✅ Ollama is already installed"
else
    echo "📥 Installing Ollama..."
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        echo "Installing for macOS..."
        curl -fsSL https://ollama.ai/install.sh | sh
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        echo "Installing for Linux..."
        curl -fsSL https://ollama.ai/install.sh | sh
    elif [[ "$OSTYPE" == "msys" ]]; then
        # Windows
        echo "Please download Ollama from: https://ollama.ai/download"
        exit 1
    fi
fi

echo ""
echo "======================================================================"
echo "Downloading Phi-3 Mini Model (2.3GB)"
echo "This may take 10-30 minutes depending on internet speed"
echo "======================================================================"
echo ""

# Download Phi-3
ollama pull phi3

echo ""
echo "======================================================================"
echo "Creating Custom Modelfile"
echo "======================================================================"
echo ""

# Create Modelfile for Mirror Analyzer
cat > Modelfile << 'EOF'
FROM phi3

SYSTEM """You are an expert health analyzer AI trained specifically for the Mirror Analyzer project.

Your role is to:
1. Analyze facial features and skin conditions
2. Interpret sensor readings (temperature, heart rate, blood oxygen)
3. Generate clear, concise health insights
4. Provide actionable recommendations
5. Flag urgent health concerns

Always:
- Use simple, non-technical language
- Be accurate and evidence-based
- Avoid medical diagnosis; instead suggest consulting a doctor
- Format output clearly with sections
- Keep recommendations practical and achievable

Do NOT provide medical diagnosis. Instead, describe observations and suggest lifestyle changes.
"""

PARAMETER temperature 0.7
PARAMETER top_k 40
PARAMETER top_p 0.9
EOF

echo "✅ Modelfile created"

echo ""
echo "======================================================================"
echo "Building Custom Model"
echo "======================================================================"
echo ""

# Build custom model
ollama create mirror-analyzer-slm -f Modelfile

echo ""
echo "======================================================================"
echo "Testing Ollama + Phi-3"
echo "======================================================================"
echo ""

# Start Ollama service (if not running)
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama service..."
    ollama serve &
    sleep 5
fi

echo "Testing Phi-3 inference..."
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mirror-analyzer-slm",
    "prompt": "A person has clear skin, no visible pimples, good hydration level, normal body temperature of 98.6F, and heart rate of 72 BPM. What would be your health assessment?",
    "stream": false
  }'

echo ""
echo ""
echo "======================================================================"
echo "✅ OLLAMA SETUP COMPLETE"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "1. Ollama is running at http://localhost:11434"
echo "2. Model loaded: mirror-analyzer-slm (Phi-3 Mini)"
echo "3. Run Phase 6: python slm_inference.py"
echo ""
echo "To keep Ollama running in background:"
echo "  ollama serve &"
echo ""
echo "======================================================================"
