# Poe.com Provider Implementation Guide for LiteLLM

## Overview

This guide provides step-by-step instructions for adding Poe.com as an official provider in LiteLLM. Poe offers access to 100+ AI models through an OpenAI-compatible API at competitive pricing (10-30% cost savings).

**Key Details:**
- **API Base**: `https://api.poe.com/v1`
- **Rate Limits**: 500 requests/minute per API key
- **Architecture**: OpenAI-compatible, leverages LiteLLM's OpenAIGPTConfig pattern
- **Implementation Effort**: ~4-6 hours (minimal code required)

## Architecture Pattern

Poe will be implemented as an OpenAI-compatible provider using LiteLLM's modern config-based architecture. This means:

1. Creating a `PoeChatConfig` class that extends `OpenAIGPTConfig`
2. Automatically inheriting all OpenAI features (streaming, async, function calling)
3. Registering the provider in central configuration files
4. Adding model pricing information

**No need for**:
- Custom completion handlers
- Manual streaming implementation
- Separate async logic
- Extensive routing code

## Implementation Steps

### Step 1: Create Provider Directory Structure

Create the following directory structure:

```bash
mkdir -p litellm/llms/poe/chat
touch litellm/llms/poe/chat/transformation.py
touch litellm/llms/poe/cost_calculator.py  # Optional, only if custom cost logic needed
```

**Result:**
```
litellm/llms/poe/
├── chat/
│   └── transformation.py
└── cost_calculator.py (optional)
```

### Step 2: Implement PoeChatConfig Class

**File:** `litellm/llms/poe/chat/transformation.py`

```python
"""
Poe Chat Transformation Config

Implements Poe.com provider support using OpenAI-compatible API pattern.
"""
from typing import Optional, Tuple, List, Any
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.secret_managers.main import get_secret_str
import litellm


class PoeChatConfig(OpenAIGPTConfig):
    """
    Configuration for Poe.com chat completions.

    Poe provides access to 100+ AI models through an OpenAI-compatible API.
    Supports streaming, async operations, and basic OpenAI parameters.

    API Documentation: https://creator.poe.com/docs/external-applications/openai-compatible-api
    """

    @property
    def custom_llm_provider(self) -> Optional[str]:
        """Return the provider identifier."""
        return "poe"

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Get API base URL and API key for Poe.

        Priority order:
        1. Explicit parameters passed to the function
        2. Environment variables (POE_API_BASE, POE_API_KEY)
        3. Default values

        Args:
            api_base: Optional explicit API base URL
            api_key: Optional explicit API key

        Returns:
            Tuple of (api_base, api_key)
        """
        api_base = (
            api_base
            or get_secret_str("POE_API_BASE")
            or "https://api.poe.com/v1"
        )
        dynamic_api_key = (
            api_key
            or get_secret_str("POE_API_KEY")
        )
        return api_base, dynamic_api_key

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get list of OpenAI parameters supported by Poe.

        Poe supports most standard OpenAI chat completion parameters but NOT:
        - logprobs
        - store
        - metadata
        - response_format (JSON mode not supported)
        - prediction
        - presence_penalty
        - frequency_penalty
        - seed

        Args:
            model: The model name being used

        Returns:
            List of supported parameter names
        """
        base_params = [
            "max_tokens",
            "max_completion_tokens",
            "stream",
            "temperature",
            "top_p",
            "stop",
            "n",
            "extra_headers",
            "max_retries",
        ]

        # Add reasoning effort parameter for reasoning models
        try:
            if litellm.supports_reasoning(
                model=model,
                custom_llm_provider=self.custom_llm_provider
            ):
                base_params.append("reasoning_effort")
        except Exception:
            pass

        # Add web search parameter if supported
        try:
            if litellm.supports_web_search(
                model=model,
                custom_llm_provider=self.custom_llm_provider
            ):
                base_params.append("web_search_options")
        except Exception:
            pass

        return base_params

    def transform_request(
        self,
        model: str,
        messages: List[Any],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform request before sending to Poe API.

        Handles Poe-specific parameter transformations:
        - auto_manage_context: Enable prompt caching
        - thinking_budget: For reasoning models
        - reasoning_effort: For reasoning models

        These parameters are passed via extra_body in LiteLLM and need
        to be included in the request body for Poe.
        """
        # Call parent transformation first
        request_data = super().transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Extract Poe-specific parameters from extra_body if present
        extra_body = optional_params.get("extra_body", {})

        # Handle auto_manage_context for prompt caching
        if "auto_manage_context" in extra_body:
            request_data["auto_manage_context"] = extra_body["auto_manage_context"]

        # Handle reasoning model parameters
        if "thinking_budget" in extra_body:
            request_data["thinking_budget"] = extra_body["thinking_budget"]

        if "reasoning_effort" in extra_body:
            request_data["reasoning_effort"] = extra_body["reasoning_effort"]

        return request_data
```

**Optional:** If Poe has specific cost calculation logic beyond standard token pricing, create:

**File:** `litellm/llms/poe/cost_calculator.py`

```python
"""
Poe Cost Calculator

Optional: Only implement if Poe has special cost calculation requirements
beyond standard token-based pricing.
"""
from typing import Optional, Tuple
from litellm.types.utils import ModelResponse, Usage


def cost_per_token(
    model: str,
    usage: Usage,
    custom_llm_provider: str = "poe"
) -> Tuple[float, float]:
    """
    Calculate cost for Poe models.

    If Poe uses standard token-based pricing, this can be omitted
    and LiteLLM will use the pricing from model_prices_and_context_window.json

    Args:
        model: Model name (with or without poe/ prefix)
        usage: Usage object with token counts
        custom_llm_provider: Should be "poe"

    Returns:
        Tuple of (prompt_cost, completion_cost)
    """
    # This is a placeholder - only implement if custom logic is needed
    # Otherwise, delete this file and rely on standard pricing
    pass
```

### Step 3: Register Provider in Type System

**File:** `litellm/types/utils.py`

Add to the `LlmProviders` enum (around line 2595):

```python
class LlmProviders(str, Enum):
    # ... existing providers ...
    OVHCLOUD = "ovhcloud"
    LEMONADE = "lemonade"
    POE = "poe"  # Add this line
```

### Step 4: Register OpenAI-Compatible Endpoints

**File:** `litellm/constants.py`

**Part A:** Add to `openai_compatible_endpoints` list (around line 513):

```python
openai_compatible_endpoints: List = [
    # ... existing endpoints ...
    "https://api.inference.wandb.ai/v1",
    "https://api.clarifai.com/v2/ext/openai/v1",
    "api.poe.com",  # Add this line
]
```

**Part B:** Add to `openai_compatible_providers` list (around line 550):

```python
openai_compatible_providers: List = [
    # ... existing providers ...
    "wandb",
    "ovhcloud",
    "poe",  # Add this line
]
```

### Step 5: Add Provider Detection Logic

**File:** `litellm/litellm_core_utils/get_llm_provider_logic.py`

Add Poe endpoint detection in the `get_llm_provider()` function (around line 280, after the wandb entry):

```python
                    elif endpoint == "https://api.inference.wandb.ai/v1":
                        custom_llm_provider = "wandb"
                        dynamic_api_key = get_secret_str("WANDB_API_KEY")
                    elif endpoint == "api.poe.com":  # Add this block
                        custom_llm_provider = "poe"
                        dynamic_api_key = get_secret_str("POE_API_KEY")

                    if api_base is not None and not isinstance(api_base, str):
```

### Step 6: Add Model Pricing

**File:** `model_prices_and_context_window.json` (root level)

Add pricing entries for Poe models. Here are the major models with approximate pricing:

```json
{
  "poe/Claude-Sonnet-4": {
    "max_input_tokens": 200000,
    "max_output_tokens": 8192,
    "max_tokens": 200000,
    "input_cost_per_token": 3e-06,
    "output_cost_per_token": 1.5e-05,
    "litellm_provider": "poe",
    "mode": "chat",
    "supports_function_calling": true,
    "supports_prompt_caching": true,
    "supports_vision": true
  },
  "poe/GPT-5-Pro": {
    "max_input_tokens": 128000,
    "max_output_tokens": 16384,
    "max_tokens": 128000,
    "input_cost_per_token": 1.5e-05,
    "output_cost_per_token": 6e-05,
    "litellm_provider": "poe",
    "mode": "chat",
    "supports_function_calling": true,
    "supports_vision": true
  },
  "poe/Gemini-2.5-Pro": {
    "max_input_tokens": 2097152,
    "max_output_tokens": 8192,
    "max_tokens": 2097152,
    "input_cost_per_token": 1e-06,
    "output_cost_per_token": 5e-06,
    "litellm_provider": "poe",
    "mode": "chat",
    "supports_function_calling": true,
    "supports_vision": true
  },
  "poe/o3-mini": {
    "max_input_tokens": 128000,
    "max_output_tokens": 65536,
    "max_tokens": 128000,
    "input_cost_per_token": 1.1e-06,
    "output_cost_per_token": 4.4e-06,
    "output_cost_per_reasoning_token": 4.4e-06,
    "litellm_provider": "poe",
    "mode": "chat",
    "supports_reasoning": true
  },
  "poe/Grok-4": {
    "max_input_tokens": 131072,
    "max_output_tokens": 131072,
    "max_tokens": 131072,
    "input_cost_per_token": 2e-06,
    "output_cost_per_token": 1e-05,
    "litellm_provider": "poe",
    "mode": "chat",
    "supports_function_calling": true,
    "supports_vision": true
  },
  "poe/DeepSeek-R1": {
    "max_input_tokens": 64000,
    "max_output_tokens": 8192,
    "max_tokens": 64000,
    "input_cost_per_token": 5.5e-07,
    "output_cost_per_token": 2.19e-06,
    "output_cost_per_reasoning_token": 2.19e-06,
    "litellm_provider": "poe",
    "mode": "chat",
    "supports_reasoning": true
  }
}
```

**Note:** Add more models as needed. Get current pricing from Poe's documentation.

### Step 7: Create Documentation

**File:** `docs/my-website/docs/providers/poe.md`

```markdown
# Poe

LiteLLM supports all models available through [Poe's API](https://poe.com/api_key).

Poe provides unified access to 100+ AI models from major providers:
- **OpenAI**: GPT-5-Pro, GPT-4.1, o3, o4-mini
- **Anthropic**: Claude-Sonnet-4, Claude-Opus-4, Claude-Haiku-4
- **Google**: Gemini-2.5-Pro, Gemini-2.5-Flash
- **xAI**: Grok-4, Grok-3-mini
- **Meta**: Llama-3.3-70B, Llama-3.2
- **Others**: DeepSeek-R1, Mistral, Qwen, and more

**Key Benefits:**
- 10-30% cost savings compared to direct provider APIs
- Single API for multiple model providers
- 500 requests/minute rate limit
- Prompt caching support

## Quick Start

### Installation

```bash
pip install litellm
```

### Basic Usage

```python
import litellm

response = litellm.completion(
    model="poe/Claude-Sonnet-4",
    messages=[{"role": "user", "content": "Hello!"}],
    api_key="your-poe-api-key"
)
print(response.choices[0].message.content)
```

## Configuration

### Environment Variables

Set your Poe API key as an environment variable:

```bash
export POE_API_KEY="your-poe-api-key"
```

Get your API key from: https://poe.com/api_key

### Proxy Server Configuration

**Basic Configuration:**

```yaml
model_list:
  - model_name: claude-sonnet
    litellm_params:
      model: poe/Claude-Sonnet-4
      api_key: os.environ/POE_API_KEY
      rpm: 450  # Rate limit: 500/min, use 450 for safety

  - model_name: gpt-5
    litellm_params:
      model: poe/GPT-5-Pro
      api_key: os.environ/POE_API_KEY
      rpm: 450
```

**Multi-Provider with Fallback:**

```yaml
model_list:
  # Primary: Use Poe for cost savings
  - model_name: claude-sonnet
    litellm_params:
      model: poe/Claude-Sonnet-4
      api_key: os.environ/POE_API_KEY
      rpm: 450

  # Fallback: Direct Anthropic if Poe is down
  - model_name: claude-sonnet
    litellm_params:
      model: claude-sonnet-4-20250514
      api_key: os.environ/ANTHROPIC_API_KEY

litellm_settings:
  fallbacks:
    - claude-sonnet: [claude-sonnet]
```

## Features

### Streaming

```python
response = litellm.completion(
    model="poe/GPT-5-Pro",
    messages=[{"role": "user", "content": "Count to 5"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Async Support

```python
import asyncio
import litellm

async def main():
    response = await litellm.acompletion(
        model="poe/Claude-Sonnet-4",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)

asyncio.run(main())
```

### Prompt Caching

Enable Poe's automatic context management for prompt caching:

```python
response = litellm.completion(
    model="poe/Claude-Sonnet-4",
    messages=messages,
    extra_body={"auto_manage_context": True}
)
```

### Reasoning Models

For models that support reasoning (like o3-mini, DeepSeek-R1):

```python
response = litellm.completion(
    model="poe/o3-mini",
    messages=[{"role": "user", "content": "Solve this complex problem..."}],
    reasoning_effort="high",  # or "low", "medium"
    extra_body={"thinking_budget": 5000}  # Token budget for reasoning
)
```

### Function Calling

Poe supports function calling for compatible models:

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }
}]

response = litellm.completion(
    model="poe/GPT-5-Pro",
    messages=[{"role": "user", "content": "What's the weather in SF?"}],
    tools=tools,
    tool_choice="auto"
)
```

## Supported Models

Access current model list via Poe API:

```bash
curl https://api.poe.com/v1/models \
  -H "Authorization: Bearer $POE_API_KEY"
```

### Popular Models

| Model | Context | Best For |
|-------|---------|----------|
| Claude-Sonnet-4 | 200K | General purpose, vision |
| GPT-5-Pro | 128K | Advanced reasoning |
| Gemini-2.5-Pro | 2M | Long context |
| Grok-4 | 131K | Real-time info |
| o3-mini | 128K | Complex reasoning |
| DeepSeek-R1 | 64K | Reasoning, coding |

## Pricing

Poe uses a point-based pricing system. Approximate costs per million tokens:

| Model | Input ($/1M) | Output ($/1M) |
|-------|-------------|---------------|
| Claude-Sonnet-4 | $3.00 | $15.00 |
| GPT-5-Pro | $15.00 | $60.00 |
| Gemini-2.5-Pro | $1.00 | $5.00 |
| o3-mini | $1.10 | $4.40 |
| DeepSeek-R1 | $0.55 | $2.19 |

**Subscription Plans:**
- Starter: $4.99/month + usage
- Pro: $19.99/month + reduced rates
- Enterprise: Custom pricing

Get current pricing at: https://poe.com/pricing

## Rate Limits

- **Default**: 500 requests/minute per API key
- **Recommendation**: Configure `rpm: 450` in proxy to prevent hitting limits

```yaml
model_list:
  - model_name: claude
    litellm_params:
      model: poe/Claude-Sonnet-4
      rpm: 450  # Safe limit
```

## Unsupported Parameters

Poe does not support these OpenAI parameters:
- `logprobs`
- `store`
- `metadata`
- `response_format` (JSON mode)
- `prediction`
- `presence_penalty`
- `frequency_penalty`
- `seed`

LiteLLM automatically drops these parameters when using Poe.

## Troubleshooting

### Authentication Errors

```
Error: 401 Unauthorized
```

**Solution:** Verify your API key at https://poe.com/api_key

```bash
# Test your API key
curl https://api.poe.com/v1/models \
  -H "Authorization: Bearer $POE_API_KEY"
```

### Rate Limit Errors

```
Error: 429 Too Many Requests
```

**Solution:** Reduce request rate or configure rate limiting:

```yaml
litellm_params:
  rpm: 450  # Requests per minute
  tpm: 100000  # Tokens per minute (adjust based on plan)
```

### Model Not Found

```
Error: 404 Model not found
```

**Solution:** Check available models:

```python
import litellm
models = litellm.get_model_list(custom_llm_provider="poe")
```

### Cost Tracking

Enable detailed cost logging:

```python
litellm.success_callback = ["langfuse"]  # Or your callback
litellm.set_verbose = True
```

## Advanced Usage

### Load Balancing

```yaml
model_list:
  - model_name: claude
    litellm_params:
      model: poe/Claude-Sonnet-4
      api_key: os.environ/POE_API_KEY_1

  - model_name: claude
    litellm_params:
      model: poe/Claude-Sonnet-4
      api_key: os.environ/POE_API_KEY_2

router_settings:
  routing_strategy: least-busy
  num_retries: 3
```

### Custom API Base

```python
response = litellm.completion(
    model="poe/Claude-Sonnet-4",
    messages=messages,
    api_base="https://api.poe.com/v1",  # Custom endpoint if needed
    api_key="your-key"
)
```

## Support

- **Poe API Docs**: https://creator.poe.com/docs/external-applications/openai-compatible-api
- **Get API Key**: https://poe.com/api_key
- **LiteLLM Docs**: https://docs.litellm.ai/
- **Issues**: https://github.com/BerriAI/litellm/issues
```

### Step 8: Create Comprehensive Tests

**File:** `tests/llm_translation/test_poe.py`

```python
"""
Tests for Poe provider implementation

Run with: pytest tests/llm_translation/test_poe.py -v
"""
import pytest
import litellm
from litellm import completion, acompletion
import os


class TestPoeProviderConfiguration:
    """Test Poe provider is properly configured in LiteLLM"""

    def test_poe_in_provider_lists(self):
        """Test that Poe is registered in all required provider lists"""
        from litellm.types.utils import LlmProviders
        from litellm.constants import (
            openai_compatible_endpoints,
            openai_compatible_providers,
        )

        # Check Poe is in LlmProviders enum
        assert hasattr(LlmProviders, "POE"), "POE not found in LlmProviders enum"
        assert LlmProviders.POE.value == "poe"

        # Check Poe is in openai_compatible_providers
        assert "poe" in openai_compatible_providers, \
            "poe not in openai_compatible_providers list"

        # Check Poe endpoint is in openai_compatible_endpoints
        assert "api.poe.com" in openai_compatible_endpoints, \
            "api.poe.com not in openai_compatible_endpoints list"

        # Check Poe is in provider_list
        assert "poe" in litellm.provider_list or LlmProviders.POE in litellm.provider_list, \
            "poe not in litellm.provider_list"

    def test_poe_model_detection(self):
        """Test that poe/ prefix is correctly detected"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        # Test with poe/ prefix
        model, provider, api_key, api_base = get_llm_provider(
            model="poe/Claude-Sonnet-4",
            custom_llm_provider=None
        )
        assert provider == "poe", f"Expected provider 'poe', got '{provider}'"
        assert model == "Claude-Sonnet-4", f"Expected model 'Claude-Sonnet-4', got '{model}'"

        # Test with explicit custom_llm_provider
        model, provider, api_key, api_base = get_llm_provider(
            model="Claude-Sonnet-4",
            custom_llm_provider="poe"
        )
        assert provider == "poe", f"Expected provider 'poe', got '{provider}'"

    def test_poe_api_base_detection(self):
        """Test that api_base with poe.com is detected"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="Claude-Sonnet-4",
            api_base="https://api.poe.com/v1"
        )
        assert provider == "poe", f"Expected provider 'poe' when api_base contains poe.com, got '{provider}'"


class TestPoeCompletion:
    """Test Poe completion functionality (requires POE_API_KEY)"""

    @pytest.mark.skipif(
        os.getenv("POE_API_KEY") is None,
        reason="POE_API_KEY not set"
    )
    def test_poe_completion_basic(self):
        """Test basic Poe completion"""
        response = completion(
            model="poe/Claude-Sonnet-4",
            messages=[{"role": "user", "content": "Say 'test successful' and nothing else"}],
            api_key=os.getenv("POE_API_KEY"),
            max_tokens=10
        )

        assert response is not None
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0
        assert response.model == "Claude-Sonnet-4" or "claude" in response.model.lower()

    @pytest.mark.skipif(
        os.getenv("POE_API_KEY") is None,
        reason="POE_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_poe_acompletion(self):
        """Test async Poe completion"""
        response = await acompletion(
            model="poe/Claude-Sonnet-4",
            messages=[{"role": "user", "content": "Hello"}],
            api_key=os.getenv("POE_API_KEY"),
            max_tokens=10
        )

        assert response is not None
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None

    @pytest.mark.skipif(
        os.getenv("POE_API_KEY") is None,
        reason="POE_API_KEY not set"
    )
    def test_poe_streaming(self):
        """Test Poe streaming"""
        response = completion(
            model="poe/Claude-Sonnet-4",
            messages=[{"role": "user", "content": "Count to 3"}],
            api_key=os.getenv("POE_API_KEY"),
            stream=True,
            max_tokens=20
        )

        chunks = []
        for chunk in response:
            if chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)

        assert len(chunks) > 0, "No streaming chunks received"
        full_response = "".join(chunks)
        assert len(full_response) > 0

    @pytest.mark.skipif(
        os.getenv("POE_API_KEY") is None,
        reason="POE_API_KEY not set"
    )
    def test_poe_model_name_formats(self):
        """Test different model name formats"""
        # Test with poe/ prefix
        response1 = completion(
            model="poe/Claude-Sonnet-4",
            messages=[{"role": "user", "content": "Hi"}],
            api_key=os.getenv("POE_API_KEY"),
            max_tokens=5
        )
        assert response1.choices[0].message.content is not None

        # Test with custom_llm_provider
        response2 = completion(
            model="Claude-Sonnet-4",
            messages=[{"role": "user", "content": "Hi"}],
            api_key=os.getenv("POE_API_KEY"),
            custom_llm_provider="poe",
            max_tokens=5
        )
        assert response2.choices[0].message.content is not None

    @pytest.mark.skipif(
        os.getenv("POE_API_KEY") is None,
        reason="POE_API_KEY not set"
    )
    def test_poe_usage_tracking(self):
        """Test that usage tokens are tracked correctly"""
        response = completion(
            model="poe/Claude-Sonnet-4",
            messages=[{"role": "user", "content": "Hi"}],
            api_key=os.getenv("POE_API_KEY"),
            max_tokens=10
        )

        assert hasattr(response, "usage"), "Response missing usage attribute"
        assert response.usage.total_tokens > 0, "Total tokens should be > 0"
        assert response.usage.prompt_tokens > 0, "Prompt tokens should be > 0"
        assert response.usage.completion_tokens > 0, "Completion tokens should be > 0"

    def test_poe_invalid_api_key(self):
        """Test error handling for invalid API key"""
        with pytest.raises(Exception) as exc_info:
            completion(
                model="poe/Claude-Sonnet-4",
                messages=[{"role": "user", "content": "Hi"}],
                api_key="invalid-key-12345",
                max_tokens=5
            )

        # Should raise an authentication error
        assert "401" in str(exc_info.value) or "unauthorized" in str(exc_info.value).lower()


class TestPoeAdvancedFeatures:
    """Test Poe-specific features"""

    @pytest.mark.skipif(
        os.getenv("POE_API_KEY") is None,
        reason="POE_API_KEY not set"
    )
    def test_poe_prompt_caching(self):
        """Test Poe prompt caching via auto_manage_context"""
        response = completion(
            model="poe/Claude-Sonnet-4",
            messages=[
                {"role": "user", "content": "Remember this: The secret code is 12345"},
                {"role": "assistant", "content": "I'll remember that."},
                {"role": "user", "content": "What was the secret code?"}
            ],
            api_key=os.getenv("POE_API_KEY"),
            extra_body={"auto_manage_context": True},
            max_tokens=20
        )

        assert response.choices[0].message.content is not None
        # Should contain the code in the response
        assert "12345" in response.choices[0].message.content

    @pytest.mark.skipif(
        os.getenv("POE_API_KEY") is None,
        reason="POE_API_KEY not set"
    )
    def test_poe_temperature_control(self):
        """Test temperature parameter"""
        # Low temperature should give consistent results
        response = completion(
            model="poe/Claude-Sonnet-4",
            messages=[{"role": "user", "content": "Say exactly: Hello World"}],
            api_key=os.getenv("POE_API_KEY"),
            temperature=0.1,
            max_tokens=10
        )

        assert response.choices[0].message.content is not None
        assert "hello" in response.choices[0].message.content.lower()

    @pytest.mark.skipif(
        os.getenv("POE_API_KEY") is None,
        reason="POE_API_KEY not set"
    )
    def test_poe_max_tokens(self):
        """Test max_tokens parameter"""
        response = completion(
            model="poe/Claude-Sonnet-4",
            messages=[{"role": "user", "content": "Write a long story about space"}],
            api_key=os.getenv("POE_API_KEY"),
            max_tokens=10  # Very limited
        )

        assert response.choices[0].message.content is not None
        # Should be short due to token limit
        assert response.usage.completion_tokens <= 15  # Allow some buffer


class TestPoeModelPricing:
    """Test Poe model pricing configuration"""

    def test_poe_models_in_pricing_file(self):
        """Test that major Poe models are in pricing file"""
        import json

        with open("model_prices_and_context_window.json", "r") as f:
            pricing = json.load(f)

        # Check some major models exist
        expected_models = [
            "poe/Claude-Sonnet-4",
            "poe/GPT-5-Pro",
            "poe/Gemini-2.5-Pro",
        ]

        for model in expected_models:
            assert model in pricing, f"Model {model} not found in pricing file"
            assert pricing[model]["litellm_provider"] == "poe"
            assert "input_cost_per_token" in pricing[model]
            assert "output_cost_per_token" in pricing[model]
            assert pricing[model]["mode"] == "chat"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
```

### Step 9: Update Documentation Index

**File:** `docs/my-website/sidebars.js`

Add Poe to the providers section:

```javascript
{
  type: 'category',
  label: 'Providers',
  items: [
    // ... existing providers ...
    'providers/perplexity',
    'providers/poe',  // Add this line
    'providers/openai',
    // ... more providers ...
  ]
}
```

### Step 10: Update Main README

**File:** `README.md`

Add Poe to the supported providers list:

```markdown
## Supported Providers

| Provider | Completion | Streaming | Async | Vision |
|----------|-----------|-----------|-------|--------|
| ... | ✅ | ✅ | ✅ | ✅ |
| Poe | ✅ | ✅ | ✅ | ✅ |
| ... | ✅ | ✅ | ✅ | ✅ |
```

## Testing the Implementation

### Manual Testing

```python
# test_poe_manual.py
import litellm
import os

# Set API key
os.environ["POE_API_KEY"] = "your-api-key"

# Test 1: Basic completion
print("Test 1: Basic completion")
response = litellm.completion(
    model="poe/Claude-Sonnet-4",
    messages=[{"role": "user", "content": "Say 'Hello from Poe!'"}]
)
print(response.choices[0].message.content)
print(f"Tokens used: {response.usage.total_tokens}")

# Test 2: Streaming
print("\nTest 2: Streaming")
for chunk in litellm.completion(
    model="poe/GPT-5-Pro",
    messages=[{"role": "user", "content": "Count to 5"}],
    stream=True
):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
print()

# Test 3: With prompt caching
print("\nTest 3: Prompt caching")
response = litellm.completion(
    model="poe/Claude-Sonnet-4",
    messages=[
        {"role": "user", "content": "Remember: password is SECRET123"},
        {"role": "assistant", "content": "Got it!"},
        {"role": "user", "content": "What's the password?"}
    ],
    extra_body={"auto_manage_context": True}
)
print(response.choices[0].message.content)
```

### Automated Testing

```bash
# Set API key
export POE_API_KEY="your-api-key"

# Run Poe-specific tests
pytest tests/llm_translation/test_poe.py -v

# Run with coverage
pytest tests/llm_translation/test_poe.py --cov=litellm.llms.poe --cov-report=html

# Run all tests to ensure no regressions
pytest tests/ -v
```

### Proxy Testing

Create a test config:

```yaml
# poe_test_config.yaml
model_list:
  - model_name: claude
    litellm_params:
      model: poe/Claude-Sonnet-4
      api_key: os.environ/POE_API_KEY
      rpm: 450
```

Start and test the proxy:

```bash
# Start proxy
litellm --config poe_test_config.yaml --port 4000

# Test via curl
curl -X POST 'http://localhost:4000/chat/completions' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-1234' \
  -d '{
    "model": "claude",
    "messages": [{"role": "user", "content": "Hello from proxy!"}]
  }'
```

## Development Checklist

Use this checklist to track implementation progress:

- [ ] Step 1: Create provider directory structure
- [ ] Step 2: Implement PoeChatConfig class
- [ ] Step 3: Register provider in type system (LlmProviders enum)
- [ ] Step 4: Register OpenAI-compatible endpoints and providers
- [ ] Step 5: Add provider detection logic
- [ ] Step 6: Add model pricing entries (at least 5-10 major models)
- [ ] Step 7: Create provider documentation
- [ ] Step 8: Create comprehensive test suite
- [ ] Step 9: Update documentation index
- [ ] Step 10: Update main README
- [ ] Manual testing completed successfully
- [ ] Automated tests passing
- [ ] Proxy server tested
- [ ] Code formatted with Black
- [ ] Type checking with MyPy passes
- [ ] Linting with Ruff passes
- [ ] Documentation reviewed and accurate
- [ ] Ready for PR submission

## Submitting the Pull Request

### Commit Message

```bash
git add -A
git commit -m "feat: Add Poe.com as official provider

- Implement PoeChatConfig extending OpenAIGPTConfig
- Add sync, async, and streaming support (inherited)
- Register provider in type system and constants
- Add model pricing for major Poe models
- Create comprehensive test suite with >90% coverage
- Add detailed documentation with examples
- Tested with proxy server and various configurations

Fixes #11530"
```

### PR Title

```
feat: Add Poe.com as official provider
```

### PR Description

```markdown
## Description
This PR adds native support for Poe.com as an official LiteLLM provider, enabling users to access 100+ AI models through Poe's OpenAI-compatible API.

## Related Issue
Closes #11530

## Implementation Details
- ✅ Created `litellm/llms/poe/chat/transformation.py` with `PoeChatConfig` class
- ✅ Extends `OpenAIGPTConfig` for automatic OpenAI feature inheritance
- ✅ Registered provider in type system and constants
- ✅ Added model pricing for major Poe models
- ✅ Created comprehensive test suite with >90% coverage
- ✅ Written detailed documentation with examples
- ✅ Tested with proxy server and various configurations

## Key Features
- Support for 100+ models via Poe's OpenAI-compatible API
- Automatic sync, async, and streaming support (inherited from OpenAIGPTConfig)
- Prompt caching via `auto_manage_context` parameter
- Reasoning model support with `reasoning_effort` and `thinking_budget`
- Full integration with LiteLLM's load balancing and fallback features
- Rate limiting configured for 500 rpm

## Testing
- [x] Manual testing with multiple models
- [x] Automated test suite passes (pytest)
- [x] Proxy server tested with sample configurations
- [x] Streaming functionality verified
- [x] Error handling tested (invalid keys, rate limits)
- [x] Cost tracking verified

## Code Quality
- [x] Code formatted with Black
- [x] Type checking with MyPy passes
- [x] Linting with Ruff passes
- [x] No regressions in existing tests

## Documentation
- [x] Provider documentation created (`docs/my-website/docs/providers/poe.md`)
- [x] Configuration examples included
- [x] Troubleshooting guide added
- [x] README updated with Poe in provider list

## Breaking Changes
None - this is a new provider addition with no impact on existing functionality.

## Additional Notes
Poe's OpenAI-compatible API makes this integration straightforward. The implementation leverages existing OpenAI patterns while adding Poe-specific features like context caching and reasoning parameters.

**Cost savings**: Poe often provides 10-30% lower costs compared to direct provider APIs while offering unified access to multiple model families.
```

## Estimated Implementation Time

- **Setup & Code (Steps 1-6)**: 2-3 hours
- **Documentation (Step 7)**: 1-2 hours
- **Testing (Steps 8-10)**: 1-2 hours
- **Review & Polish**: 1 hour

**Total**: 4-6 hours for a complete, production-ready implementation

## Success Criteria

Your implementation is complete when:

1. ✅ All tests pass with >90% coverage
2. ✅ Documentation is comprehensive and accurate
3. ✅ Manual testing with proxy server works
4. ✅ Multiple models tested successfully (at least 3)
5. ✅ Streaming functionality works
6. ✅ Error handling is robust
7. ✅ Rate limiting is properly configured
8. ✅ Cost calculation is accurate
9. ✅ Code quality checks pass (Black, MyPy, Ruff)
10. ✅ PR is ready for review

## Getting Help

- **LiteLLM Contributing Guide**: Check `CONTRIBUTING.md` in the repo
- **Provider Development Docs**: https://docs.litellm.ai/docs/providers
- **Poe API Docs**: https://creator.poe.com/docs/external-applications/openai-compatible-api
- **LiteLLM Discord**: Join for community support
- **GitHub Issues**: https://github.com/BerriAI/litellm/issues

Good luck with your implementation! 🚀
