# LiteLLM Poe.com Provider Implementation Instructions

## Objective
Implement native Poe.com support as an official provider in LiteLLM, enabling users to access Poe's API through LiteLLM's unified interface without workarounds.

## Background Context

**Current Status**: Poe.com is not officially supported in LiteLLM (GitHub issue #11530 closed as "not planned"). However, Poe's OpenAI-compatible API makes integration straightforward.

**Why This Matters**: 
- Poe provides access to 100+ AI models from OpenAI, Anthropic, Google, xAI, Meta, and more
- 10-30% cost savings compared to direct provider APIs
- 500 requests/minute rate limit
- Official API launched July 2025

**Implementation Strategy**: Since Poe's API is OpenAI-compatible, we can leverage LiteLLM's existing OpenAI provider implementation as a foundation while adding Poe-specific configurations and features.

## Repository Structure Overview

Your forked repository should follow this structure:
```
litellm/
├── litellm/
│   ├── llms/               # Provider implementations
│   │   ├── openai.py       # Reference implementation
│   │   ├── anthropic.py
│   │   ├── poe.py          # NEW - To be created
│   ├── proxy/
│   │   ├── proxy_server.py
│   ├── types/
│   │   ├── llms/
│   │   │   ├── openai.py   # Type definitions
│   │   │   ├── poe.py      # NEW - To be created
│   ├── cost_calculator.py  # Cost tracking (to be updated)
│   ├── model_prices_and_context_window.json  # Model configs
│   ├── utils.py
├── docs/
│   ├── my-website/
│   │   ├── docs/
│   │   │   ├── providers/
│   │   │   │   ├── poe.md  # NEW - Documentation
├── tests/
│   ├── test_poe.py         # NEW - Test suite
```

## Implementation Steps

### Step 1: Create Poe Provider Implementation File

**File**: `litellm/llms/poe.py`

**Requirements**:
1. Create a new provider file that handles Poe API requests
2. Since Poe is OpenAI-compatible, use the OpenAI implementation as a base
3. Add Poe-specific configurations and error handling
4. Support both sync and async operations
5. Implement streaming support

**Key Functions to Implement**:
```python
def completion(
    model: str,
    messages: list,
    api_base: str,
    custom_llm_provider: str,
    **kwargs
) -> ModelResponse:
    """
    Handle Poe chat completions
    - Set api_base to https://api.poe.com/v1
    - Transform model names (remove 'poe/' prefix if present)
    - Handle Poe-specific parameters via extra_body
    - Return standardized ModelResponse
    """
    pass

async def acompletion(
    model: str,
    messages: list,
    api_base: str,
    custom_llm_provider: str,
    **kwargs
) -> ModelResponse:
    """Async version of completion"""
    pass

def streaming(
    model: str,
    messages: list,
    api_base: str,
    custom_llm_provider: str,
    **kwargs
):
    """Handle streaming responses"""
    pass
```

**Implementation Notes**:
- API Base URL: `https://api.poe.com/v1`
- Authentication: Bearer token via `api_key` parameter
- Model naming: Support both `poe/Model-Name` and direct `Model-Name` formats
- Rate limits: 500 requests/minute per API key
- Unsupported parameters: `logprobs`, `store`, `metadata`, `response_format`, `prediction`, `presence_penalty`, `frequency_penalty`, `seed`
- Use `drop_params=True` to handle unsupported parameters gracefully

**Reference Files to Study**:
- `litellm/llms/openai.py` - Base implementation pattern
- `litellm/llms/anthropic.py` - Similar provider structure
- `litellm/llms/azure.py` - Custom endpoint handling

### Step 2: Add Provider to Main Router

**File**: `litellm/utils.py`

**Modify the `get_llm_provider()` function**:
Add Poe detection logic to identify when a model should use the Poe provider.

```python
def get_llm_provider(model: str, custom_llm_provider: Optional[str] = None, api_base: Optional[str] = None):
    # ... existing code ...
    
    # Add Poe detection
    if custom_llm_provider == "poe" or model.startswith("poe/"):
        return model, "poe", api_base or "https://api.poe.com/v1", None
    
    # Check if api_base is Poe endpoint
    if api_base and "poe.com" in api_base:
        return model, "poe", api_base, None
    
    # ... rest of existing code ...
```

**Modify the `completion()` function**:
Add Poe provider routing in the main completion function.

```python
def completion(
    model: str,
    messages: list,
    custom_llm_provider: Optional[str] = None,
    **kwargs
):
    # ... existing code ...
    
    # Add Poe routing
    elif custom_llm_provider == "poe" or model_provider == "poe":
        from litellm.llms.poe import completion as poe_completion
        response = poe_completion(
            model=model,
            messages=messages,
            api_base=api_base,
            custom_llm_provider="poe",
            **kwargs
        )
    
    # ... rest of existing code ...
```

### Step 3: Add Model Pricing and Context Windows

**File**: `litellm/model_prices_and_context_window.json`

Add Poe models with their pricing information. Use the point-based pricing that Poe uses.

```json
{
  "poe/Claude-Sonnet-4": {
    "max_tokens": 200000,
    "max_input_tokens": 200000,
    "max_output_tokens": 8192,
    "input_cost_per_token": 0.000003,
    "output_cost_per_token": 0.000015,
    "litellm_provider": "poe",
    "mode": "chat"
  },
  "poe/GPT-5-Pro": {
    "max_tokens": 128000,
    "max_input_tokens": 128000,
    "max_output_tokens": 16384,
    "input_cost_per_token": 0.000015,
    "output_cost_per_token": 0.00006,
    "litellm_provider": "poe",
    "mode": "chat"
  },
  "poe/Gemini-2.5-Pro": {
    "max_tokens": 2097152,
    "max_input_tokens": 2097152,
    "max_output_tokens": 8192,
    "input_cost_per_token": 0.000001,
    "output_cost_per_token": 0.000005,
    "litellm_provider": "poe",
    "mode": "chat"
  }
}
```

**Note**: Add entries for major Poe models. Pricing should reflect the approximate cost through Poe's point system.

### Step 4: Update Cost Calculator

**File**: `litellm/cost_calculator.py`

Ensure the cost calculator properly handles Poe models using the pricing from the JSON file.

**Verify**:
- Cost calculation works for Poe models
- Token counting is accurate
- Caching discounts can be applied (Poe supports prompt caching)

### Step 5: Add Type Definitions

**File**: `litellm/types/llms/poe.py` (NEW)

Create type definitions for Poe-specific parameters and responses.

```python
from typing import Optional, List, Dict, Any, Union
from typing_extensions import TypedDict, Required

class PoeConfig(TypedDict, total=False):
    """Configuration for Poe API requests"""
    api_key: Required[str]
    api_base: str  # Default: https://api.poe.com/v1
    timeout: Optional[int]
    max_retries: Optional[int]
    auto_manage_context: Optional[bool]  # Enable prompt caching
    thinking_budget: Optional[int]  # For reasoning models
    reasoning_effort: Optional[str]  # For reasoning models

class PoeExtraBody(TypedDict, total=False):
    """Extra body parameters specific to Poe"""
    auto_manage_context: Optional[bool]
    thinking_budget: Optional[int]
    reasoning_effort: Optional[str]
```

### Step 6: Create Documentation

**File**: `docs/my-website/docs/providers/poe.md`

Create comprehensive documentation for the Poe provider.

**Documentation Structure**:
```markdown
# Poe

LiteLLM supports all models available through [Poe's API](https://poe.com/api_key).

## Quick Start

```python
import litellm
response = litellm.completion(
    model="poe/Claude-Sonnet-4",
    messages=[{"role": "user", "content": "Hello!"}],
    api_key="your-poe-api-key"
)
```

## Supported Models

Poe provides access to 100+ AI models including:
- OpenAI: GPT-5-Pro, GPT-4.1, o3, o4-mini
- Anthropic: Claude-Sonnet-4.5, Claude-Opus-4.1, Claude-Haiku-4.5
- Google: Gemini-2.5-Pro, Gemini-2.5-Flash
- xAI: Grok-4, Grok-3-mini
- Meta: Llama-3.1-405B, Llama-3.2
- Open source models: DeepSeek, Mistral, Qwen

## Configuration

### Environment Variables
```bash
export POE_API_KEY="your-poe-api-key"
```

### Proxy Configuration
```yaml
model_list:
  - model_name: claude-sonnet
    litellm_params:
      model: poe/Claude-Sonnet-4
      api_key: os.environ/POE_API_KEY
```

## Features

### Prompt Caching
Enable automatic context management for caching:
```python
response = litellm.completion(
    model="poe/Claude-Sonnet-4",
    messages=messages,
    extra_body={"auto_manage_context": True}
)
```

### Rate Limits
- 500 requests per minute per API key
- Configure in proxy: `rpm: 450`

### Streaming
```python
response = litellm.completion(
    model="poe/GPT-5-Pro",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="")
```

## Pricing
Poe uses a point-based pricing system:
- Often 10-30% cheaper than direct provider APIs
- Subscription plans: $4.99-$249.99/month
- Add-on points: $30 per 1M points

## Troubleshooting

### Authentication Errors
Verify your API key at https://poe.com/api_key

### Rate Limits
Stay under 500 rpm by configuring `rpm: 450` in your proxy config

### Model Availability
List available models:
```bash
curl https://api.poe.com/v1/models \
  -H "Authorization: Bearer $POE_API_KEY"
```
```

### Step 7: Create Comprehensive Tests

**File**: `tests/test_poe.py`

Create a test suite covering all Poe functionality.

**Test Coverage Required**:
```python
import pytest
import litellm
from litellm import completion, acompletion
import os

# Test basic completion
def test_poe_completion():
    """Test basic Poe completion"""
    response = completion(
        model="poe/Claude-Sonnet-4",
        messages=[{"role": "user", "content": "Say 'test successful'"}],
        api_key=os.getenv("POE_API_KEY")
    )
    assert response.choices[0].message.content is not None

# Test async completion
@pytest.mark.asyncio
async def test_poe_acompletion():
    """Test async Poe completion"""
    response = await acompletion(
        model="poe/GPT-5-Pro",
        messages=[{"role": "user", "content": "Hello"}],
        api_key=os.getenv("POE_API_KEY")
    )
    assert response.choices[0].message.content is not None

# Test streaming
def test_poe_streaming():
    """Test Poe streaming"""
    response = completion(
        model="poe/Claude-Sonnet-4",
        messages=[{"role": "user", "content": "Count to 3"}],
        api_key=os.getenv("POE_API_KEY"),
        stream=True
    )
    chunks = []
    for chunk in response:
        if chunk.choices[0].delta.content:
            chunks.append(chunk.choices[0].delta.content)
    assert len(chunks) > 0

# Test model name variations
def test_poe_model_name_formats():
    """Test different model name formats"""
    # With poe/ prefix
    response1 = completion(
        model="poe/Claude-Sonnet-4",
        messages=[{"role": "user", "content": "Hi"}],
        api_key=os.getenv("POE_API_KEY")
    )
    assert response1.choices[0].message.content is not None
    
    # With custom_llm_provider
    response2 = completion(
        model="Claude-Sonnet-4",
        messages=[{"role": "user", "content": "Hi"}],
        api_key=os.getenv("POE_API_KEY"),
        custom_llm_provider="poe"
    )
    assert response2.choices[0].message.content is not None

# Test error handling
def test_poe_invalid_api_key():
    """Test error handling for invalid API key"""
    with pytest.raises(Exception):
        completion(
            model="poe/Claude-Sonnet-4",
            messages=[{"role": "user", "content": "Hi"}],
            api_key="invalid-key"
        )

# Test rate limiting headers
def test_poe_rate_limit_headers():
    """Test that rate limit headers are captured"""
    response = completion(
        model="poe/Claude-Sonnet-4",
        messages=[{"role": "user", "content": "Hi"}],
        api_key=os.getenv("POE_API_KEY")
    )
    # Check for rate limit info in response headers
    assert hasattr(response, '_hidden_params')

# Test extra_body parameters
def test_poe_extra_body():
    """Test Poe-specific parameters via extra_body"""
    response = completion(
        model="poe/Claude-Sonnet-4",
        messages=[{"role": "user", "content": "Hi"}],
        api_key=os.getenv("POE_API_KEY"),
        extra_body={
            "auto_manage_context": True
        }
    )
    assert response.choices[0].message.content is not None

# Test cost calculation
def test_poe_cost_calculation():
    """Test that costs are calculated correctly"""
    response = completion(
        model="poe/Claude-Sonnet-4",
        messages=[{"role": "user", "content": "Hi"}],
        api_key=os.getenv("POE_API_KEY")
    )
    assert hasattr(response, 'usage')
    assert response.usage.total_tokens > 0
```

**Test Execution**:
```bash
# Set up test environment
export POE_API_KEY="your-test-api-key"

# Run tests
pytest tests/test_poe.py -v

# Run with coverage
pytest tests/test_poe.py --cov=litellm.llms.poe --cov-report=html
```

### Step 8: Update Provider Lists and Documentation Index

**Files to Update**:

1. **`litellm/main.py`** or equivalent - Add Poe to supported providers list
2. **`docs/my-website/sidebars.js`** - Add Poe to documentation navigation
3. **`README.md`** - Add Poe to list of supported providers
4. **`litellm/__init__.py`** - Ensure Poe provider is exported if needed

### Step 9: Add Configuration Examples

**File**: `docs/my-website/docs/proxy/configs/poe_config.yaml` (NEW)

Create example proxy configurations:

```yaml
# Basic Poe configuration
model_list:
  - model_name: claude-sonnet
    litellm_params:
      model: poe/Claude-Sonnet-4
      api_key: os.environ/POE_API_KEY
      rpm: 450

  - model_name: gpt-5
    litellm_params:
      model: poe/GPT-5-Pro
      api_key: os.environ/POE_API_KEY
      rpm: 450

# Multi-provider with fallback
model_list:
  - model_name: claude-sonnet
    litellm_params:
      model: poe/Claude-Sonnet-4
      api_key: os.environ/POE_API_KEY
      rpm: 450
      
  - model_name: claude-sonnet
    litellm_params:
      model: claude-sonnet-4-20250514
      api_key: os.environ/ANTHROPIC_API_KEY

litellm_settings:
  fallbacks:
    - claude-sonnet: [claude-sonnet]  # Fallback to direct Anthropic
```

### Step 10: Integration with Proxy Features

Ensure Poe works with LiteLLM's advanced features:

**Load Balancing**:
```yaml
router_settings:
  routing_strategy: least-busy
  num_retries: 3
```

**Caching**:
- Ensure Redis caching works with Poe responses
- Test semantic caching if supported

**Logging and Callbacks**:
- Verify success/failure callbacks work
- Test integration with logging platforms (Langfuse, etc.)

**Virtual Keys**:
- Ensure virtual key system works with Poe API keys
- Test key rotation and management

## Development Workflow

### 1. Set Up Development Environment

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/litellm.git
cd litellm

# Create feature branch
git checkout -b feature/add-poe-provider

# Install development dependencies
pip install -e ".[dev]"
pip install -e ".[proxy]"

# Set up pre-commit hooks (if available)
pre-commit install
```

### 2. Implementation Order

Follow this sequence for smooth development:

1. ✅ Create `litellm/llms/poe.py` with basic completion support
2. ✅ Add Poe detection to `litellm/utils.py`
3. ✅ Test basic functionality manually
4. ✅ Add async and streaming support to `poe.py`
5. ✅ Create type definitions in `litellm/types/llms/poe.py`
6. ✅ Add model pricing to JSON file
7. ✅ Write comprehensive tests in `tests/test_poe.py`
8. ✅ Create documentation in `docs/my-website/docs/providers/poe.md`
9. ✅ Update all provider lists and indexes
10. ✅ Run full test suite and fix any issues

### 3. Testing Strategy

**Manual Testing**:
```python
# Test in Python console
import litellm
import os

# Set API key
os.environ["POE_API_KEY"] = "your-key"

# Test basic completion
response = litellm.completion(
    model="poe/Claude-Sonnet-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)

# Test streaming
for chunk in litellm.completion(
    model="poe/GPT-5-Pro",
    messages=[{"role": "user", "content": "Count to 5"}],
    stream=True
):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

**Proxy Testing**:
```bash
# Start proxy with Poe config
litellm --config poe_config.yaml --port 4000

# Test via curl
curl -X POST 'http://localhost:4000/chat/completions' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-1234' \
  -d '{
    "model": "claude-sonnet",
    "messages": [{"role": "user", "content": "Hello from proxy!"}]
  }'
```

**Automated Testing**:
```bash
# Run Poe-specific tests
pytest tests/test_poe.py -v

# Run full test suite
pytest tests/ -v

# Check code coverage
pytest tests/test_poe.py --cov=litellm.llms.poe --cov-report=html
```

### 4. Code Quality Checks

Before committing:

```bash
# Format code (if using black)
black litellm/llms/poe.py tests/test_poe.py

# Type checking (if using mypy)
mypy litellm/llms/poe.py

# Linting (if using flake8)
flake8 litellm/llms/poe.py

# Run pre-commit hooks
pre-commit run --all-files
```

### 5. Documentation Review

Ensure documentation is complete:
- [ ] Provider documentation (`poe.md`) is clear and comprehensive
- [ ] Examples are tested and work
- [ ] Configuration examples are included
- [ ] Troubleshooting section covers common issues
- [ ] Links to Poe's official documentation are included
- [ ] Pricing information is accurate

### 6. Commit and Push

```bash
# Stage changes
git add litellm/llms/poe.py
git add litellm/types/llms/poe.py
git add litellm/utils.py
git add litellm/model_prices_and_context_window.json
git add docs/my-website/docs/providers/poe.md
git add tests/test_poe.py
# ... add other modified files

# Commit with descriptive message
git commit -m "feat: Add Poe.com as official provider

- Implement Poe provider in litellm/llms/poe.py
- Add sync, async, and streaming support
- Include Poe-specific type definitions
- Add comprehensive test coverage
- Create detailed documentation
- Update model pricing and provider lists

Fixes #11530"

# Push to your fork
git push origin feature/add-poe-provider
```

### 7. Create Pull Request

**PR Title**: `feat: Add Poe.com as official provider`

**PR Description Template**:
```markdown
## Description
This PR adds native support for Poe.com as an official LiteLLM provider, enabling users to access 100+ AI models through Poe's API without workarounds.

## Related Issue
Closes #11530

## Implementation Details
- ✅ Created `litellm/llms/poe.py` with full provider implementation
- ✅ Added sync, async, and streaming support
- ✅ Implemented Poe-specific type definitions
- ✅ Added model pricing information for major Poe models
- ✅ Created comprehensive test suite with >90% coverage
- ✅ Written detailed documentation with examples
- ✅ Tested with proxy server and various configurations

## Key Features
- Support for 100+ models via Poe's OpenAI-compatible API
- Automatic rate limiting (500 rpm)
- Prompt caching support via `auto_manage_context`
- Reasoning model parameters (thinking_budget, reasoning_effort)
- Full integration with LiteLLM's load balancing and fallback features

## Testing
- [x] Manual testing with multiple models
- [x] Automated test suite passes
- [x] Proxy server tested with sample configurations
- [x] Streaming functionality verified
- [x] Error handling tested (invalid keys, rate limits, etc.)

## Documentation
- [x] Provider documentation created
- [x] Configuration examples included
- [x] Troubleshooting guide added
- [x] README updated with Poe in provider list

## Checklist
- [x] Code follows project style guidelines
- [x] Tests added and passing
- [x] Documentation complete and accurate
- [x] No breaking changes to existing functionality
- [x] Backward compatible

## Additional Notes
Poe's OpenAI-compatible API makes this integration straightforward. The implementation leverages existing OpenAI patterns while adding Poe-specific features like context caching and reasoning parameters.

Cost savings: Poe often provides 10-30% lower costs compared to direct provider APIs while offering unified access to multiple model families.
```

## Technical Implementation Details

### Poe API Specifics

**Base URL**: `https://api.poe.com/v1`

**Authentication**: 
```python
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
```

**Request Format** (OpenAI-compatible):
```json
{
  "model": "Claude-Sonnet-4",
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "max_tokens": 1000,
  "temperature": 0.7,
  "stream": false
}
```

**Response Format** (OpenAI-compatible):
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "Claude-Sonnet-4",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

**Rate Limit Headers**:
- `x-ratelimit-limit-requests`: 500
- `x-ratelimit-remaining-requests`: Remaining in window
- `x-ratelimit-reset-requests`: Seconds until reset

**Poe-Specific Parameters** (via `extra_body`):
```python
extra_body = {
    "auto_manage_context": True,  # Enable prompt caching
    "thinking_budget": 5000,      # For reasoning models
    "reasoning_effort": "high"    # For reasoning models
}
```

### Error Handling

Implement proper error handling for Poe-specific errors:

```python
# Common errors to handle
- 401: Invalid API key
- 402: Insufficient credits
- 404: Model not found
- 429: Rate limit exceeded
- 500: Poe server error
```

### Model Name Transformations

Support multiple model name formats:
```python
# Input formats to support:
"poe/Claude-Sonnet-4"  # Explicit prefix
"Claude-Sonnet-4"       # Direct name with custom_llm_provider="poe"
```

Strip "poe/" prefix before sending to API:
```python
model = model.replace("poe/", "")
```

## Success Criteria

Your implementation is complete when:

- [x] All test cases pass (>90% coverage)
- [x] Documentation is comprehensive and accurate
- [x] Manual testing with proxy server works
- [x] Multiple models tested successfully
- [x] Streaming functionality works
- [x] Error handling is robust
- [x] Rate limiting is properly configured
- [x] Cost calculation is accurate
- [x] Integration with existing LiteLLM features works
- [x] Code follows project conventions
- [x] PR is ready for review

## Reference Materials

**Poe API Documentation**:
- OpenAI-compatible API: https://creator.poe.com/docs/external-applications/openai-compatible-api
- API Reference: https://creator.poe.com/docs/api
- Model List Endpoint: https://api.poe.com/v1/models

**LiteLLM Reference Implementations**:
- `litellm/llms/openai.py` - Similar API structure
- `litellm/llms/azure.py` - Custom endpoint handling
- `litellm/llms/anthropic.py` - Provider pattern example

**LiteLLM Documentation**:
- Contributing Guide: Check repo for CONTRIBUTING.md
- Provider Development: https://docs.litellm.ai/docs/providers
- Proxy Configuration: https://docs.litellm.ai/docs/simple_proxy

## Tips for Success

1. **Start Simple**: Get basic completion working first, then add streaming and async
2. **Follow Patterns**: Study existing providers closely - especially OpenAI
3. **Test Incrementally**: Test each feature as you add it
4. **Handle Errors**: Implement comprehensive error handling from the start
5. **Document Well**: Good documentation increases PR acceptance chances
6. **Be Responsive**: Engage with PR feedback promptly and professionally

## Questions to Address in PR Review

Be prepared to discuss:
1. Why Poe should be officially supported (cost savings, model access, user demand)
2. How this integrates with existing features (load balancing, caching, etc.)
3. Test coverage and quality assurance approach
4. Maintenance commitment (are you willing to maintain this provider?)
5. Performance implications (any overhead from the implementation?)

## Final Notes

This is a substantial contribution that will benefit many LiteLLM users. By providing a complete, well-tested implementation with comprehensive documentation, you significantly increase the chances of acceptance.

Good luck with your implementation! 🚀
