# hallucination_detection/main.py

import os
import sys
from pathlib import Path

from hallucination_detection.check_aggregator import CheckAggregator
from hallucination_detection.llm import LLMContainer
from hallucination_detection.debug_logger import set_debug_level, DEBUG_INFO, DEBUG_ERROR, debug_print
from hallucination_detection.statement_parser import StatementParser

def main():
    # Set desired debug level (e.g., DEBUG_INFO for more detail)
    set_debug_level(DEBUG_INFO)

    # 1. Initialize aggregator
    aggregator = CheckAggregator()

    # 2. LLM container with different LLMs
    llm_container = LLMContainer()
    # llm_container.register_llm("OpenAI", "gpt-3.5")
    # llm_container.register_llm("Anthropic", "claude-v1")
    llm_container.register_llm("cerebras", "llama3.1-8b")
    llm_container.register_llm("cerebras", "llama3.3-70b")

    # 3. Read sample text from file using relative path
    current_dir = Path(__file__).parent
    sample_file = os.path.join(current_dir, "samples", "sample1.txt")
    try:
        with open(sample_file, 'r') as f:
            sample_text = f.read()
        debug_print(DEBUG_INFO, f"Successfully read sample text from {sample_file}")
    except FileNotFoundError:
        debug_print(DEBUG_ERROR, f"Sample file not found: {sample_file}")
        return 1
    except Exception as e:
        debug_print(DEBUG_ERROR, f"Error reading sample file: {str(e)}")
        return 1

    # Create a StatementParser that partitions by paragraph first, limiting each partition to 10 words
    parser = StatementParser(max_words=200, split_by_paragraph=True)
    partitions = parser.partition_text(sample_text)

    print("\nPartitions:")
    for i, p in enumerate(partitions):
        print(f"Partition {i+1}: {p}")

    # Extract statements and check each
    print("\nChecking statements:")
    all_statements = []  # Store all analyzed statements
    for i, partition in enumerate(partitions):
        statements = parser.extract_statements(partition)
        for j, statement in enumerate(statements):
            score, domain = aggregator.check_statement(statement)
            all_statements.append({
                'statement': statement,
                'score': score,
                'domain': domain,
                'partition': i+1,
                'statement_num': j+1
            })
            print(f"  P{i+1}-S{j+1} => '{statement}' => score: {score} ({domain})")

    # Generate summary report
    print("\n" + "="*80)
    print("STATEMENT ANALYSIS SUMMARY")
    print("="*80)

    for result in all_statements:
        stmt = result['statement']
        score = result['score']
        domain = result['domain'].upper()
        
        # Define risk levels with emojis
        if score > 0.9:
            risk_class = "✅ GOOD"
            explanation = "Highly reliable statement"
        elif score > 0.6:
            risk_class = "⚠️ LOW_RISK"
            explanation = "Generally reliable but verify"
        else:
            risk_class = "❌ HIGH_RISK"
            explanation = "Potential hallucination detected"
        
        print(f"\n[Statement P{result['partition']}-S{result['statement_num']}] [{domain}]")
        print(f"  Content: {stmt}")
        print(f"  Domain: {domain}")
        print(f"  Final Score: {score:.2f}")
        print(f"  Risk Level: {risk_class}")
        print(f"  Explanation: {explanation}")
        # debug_print(DEBUG_INFO, f"Statement {result['partition']}-{result['statement_num']} Classification: {risk_class} Domain: {domain}")
        print("-" * 60)

    # 4. If we wanted fewer debug prints, set_debug_level(DEBUG_ERROR)
    # set_debug_level(DEBUG_ERROR)

if __name__ == "__main__":
    sys.exit(main())
