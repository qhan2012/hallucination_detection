# Hallucination Detection

A system for detecting potential hallucinations in language model outputs using multi-domain verification.

## Features
- Multi-domain statement verification
- Support for various LLM providers (Cerebras, OpenAI, Anthropic)
- Domain-specific fact checking
- Risk level assessment with visual indicators

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hallucination-detection.git
cd hallucination-detection

# Install dependencies
pip install -r requirements.txt
```

## Environment Setup
```bash
export CEREBRAS_API_KEY="your-key-here"
```

## Usage

1. Add your text to be analyzed in `hallucination_detection/samples/sample1.txt`:
```text
The French Revolution started in 1789.
AI can think and feel emotions.
The Earth orbits around the Sun.
```

2. Run the analysis:
```bash
# Set up environment variable
export CEREBRAS_API_KEY="your-key-here"

# Run the detector
python -m hallucination_detection.main
```

3. The output will show:
- Partition of input text
- Statement analysis with risk levels:
  - ✅ GOOD (High reliability)
  - ⚠️ LOW_RISK (Need verification)
  - ❌ HIGH_RISK (Potential hallucination)
- Domain classification for each statement

Example output:
```text
[Statement P1-S1] [HISTORY]
  Content: The French Revolution started in 1789.
  Final Score: 0.95
  Risk Level: ✅ GOOD
  Explanation: Highly reliable statement
------------------------------------------------------------
```

## License
MIT License