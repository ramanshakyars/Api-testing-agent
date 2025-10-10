from crewai import LLM

# ===========================
# LLM Setup with Ollama
# ===========================
ollama_llm = LLM(
    model="ollama/qwen2.5-coder:3b",
    base_url="http://localhost:11434",
    timeout=7200
)
