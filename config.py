import os
import dspy


def configure_lm() -> None:
    """Configure the global DSPy LM from environment variables.

    Uses proxy settings if available, otherwise falls back to direct OpenRouter.
    Controlled by:
      - OPENROUTER_MODEL (default: meta-llama/llama-3-8b-instruct:free)
      - LLM_PROXY_HOST (base URL for both proxy and OpenRouter models)
      - LLM_PROXY_API_KEY (API key for both proxy and OpenRouter models)
      - OPENROUTER_API_KEY (fallback if LLM_PROXY_API_KEY not set)
      - LM_TEMPERATURE (default: 0.1)
      - LM_MAX_TOKENS (default: 8000)
    """
    temperature = float(os.getenv("LM_TEMPERATURE", "0.1"))
    max_tokens = int(os.getenv("LM_MAX_TOKENS", "8000"))

    # Get proxy settings
    proxy_host = os.getenv("LLM_PROXY_HOST")
    proxy_api_key = os.getenv("LLM_PROXY_API_KEY")
    
    # Check if we have a specific proxy model (non-OpenRouter)
    proxy_model = os.getenv("LLM_PROXY_MODEL")
    
    if proxy_model and proxy_api_key and proxy_host:
        # Use specific proxy model (like Claude, etc.)
        dspy.configure(
            lm=dspy.LM(
                proxy_model,
                api_key=proxy_api_key,
                api_base=proxy_host.rstrip("/"),
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
        print(f"Configured with proxy model: {proxy_model} via {proxy_host}")
        return

    # Use OpenRouter model with proxy settings if available
    openrouter_model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct:free")
    
    # Use proxy API key if available, otherwise fall back to OpenRouter API key
    api_key = proxy_api_key or os.getenv("OPENROUTER_API_KEY")
    
    # Use proxy host if available, otherwise use default OpenRouter URL
    api_base = proxy_host.rstrip("/") if proxy_host else "https://openrouter.ai/api/v1"
    
    # Model name handling: 
    # - If using proxy, keep the model name as-is (proxy will handle routing)
    # - If using direct OpenRouter, ensure openrouter/ prefix for litellm
    if proxy_host:
        # Using proxy - keep full model name (including openrouter/ prefix) so proxy can route correctly
        # But still use openai/ prefix for litellm to use OpenAI adapter
        model_name = f"openai/{openrouter_model}"
        connection_type = "proxy"
    else:
        # Direct OpenRouter - normalize for litellm OpenRouter provider
        model_name = openrouter_model if openrouter_model.startswith("openrouter/") else f"openrouter/{openrouter_model}"
        connection_type = "direct"

    dspy.settings.configure(
        lm=dspy.LM(
            model_name,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    )
    
    print(f"Configured OpenRouter model: {model_name} via {api_base} with {connection_type} connection") 