from crewai import LLM

# ===========================
# LLM Setup with Ollama
# ===========================
ollama_llm = LLM(
    model="ollama/codellama:7b",
    base_url="http://localhost:11434",
    timeout=7200
)
