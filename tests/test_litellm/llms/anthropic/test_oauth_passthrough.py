"""
Test OAuth pass-through support for Anthropic.

This test verifies that:
1. When Authorization header is present (OAuth token), x-api-key is NOT added
2. When Authorization header is absent and ANTHROPIC_API_KEY exists, x-api-key IS added
3. When neither Authorization nor API key is present, appropriate error is raised
"""
import pytest
from litellm.llms.anthropic.common_utils import AnthropicModelInfo
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)


class TestAnthropicOAuthPassthrough:
    """Test OAuth pass-through functionality for Anthropic"""

    def test_oauth_token_present_no_api_key_injection(self):
        """When Authorization header is present, x-api-key should NOT be added"""
        model_info = AnthropicModelInfo()

        # Headers with OAuth Authorization token
        headers = {
            "authorization": "Bearer sk-ant-oat01-test-token-12345",
            "content-type": "application/json",
        }

        result = model_info.validate_environment(
            headers=headers,
            model="claude-3-haiku-20240307",
            messages=[{"role": "user", "content": "test"}],
            optional_params={},
            litellm_params={},
            api_key=None,  # No API key provided
            api_base="https://api.anthropic.com",
        )

        # Should NOT have x-api-key header
        assert "x-api-key" not in result
        # Should still have Authorization header
        assert "authorization" in result
        assert result["authorization"] == "Bearer sk-ant-oat01-test-token-12345"
        # Should have other required headers
        assert "anthropic-version" in result
        assert "content-type" in result

    def test_api_key_present_no_oauth_token(self):
        """When API key is present but no Authorization header, x-api-key should be added"""
        model_info = AnthropicModelInfo()

        headers = {
            "content-type": "application/json",
        }

        result = model_info.validate_environment(
            headers=headers,
            model="claude-3-haiku-20240307",
            messages=[{"role": "user", "content": "test"}],
            optional_params={},
            litellm_params={},
            api_key="sk-ant-api-key-12345",
            api_base="https://api.anthropic.com",
        )

        # Should have x-api-key header
        assert "x-api-key" in result
        assert result["x-api-key"] == "sk-ant-api-key-12345"
        # Should NOT have Authorization header
        assert "authorization" not in result
        # Should have other required headers
        assert "anthropic-version" in result
        assert "content-type" in result

    def test_oauth_token_present_with_beta_headers(self):
        """OAuth tokens should work with beta headers (prompt caching, etc)"""
        model_info = AnthropicModelInfo()

        # Headers with OAuth Authorization token
        headers = {
            "authorization": "Bearer sk-ant-oat01-test-token-12345",
        }

        # Messages with cache_control to trigger beta header
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "test",
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            }
        ]

        result = model_info.validate_environment(
            headers=headers,
            model="claude-3-haiku-20240307",
            messages=messages,
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base="https://api.anthropic.com",
        )

        # Should NOT have x-api-key header
        assert "x-api-key" not in result
        # Should have Authorization header
        assert "authorization" in result
        # Should have beta header for prompt caching
        assert "anthropic-beta" in result
        assert "prompt-caching-2024-07-31" in result["anthropic-beta"]

    def test_experimental_passthrough_oauth_support(self):
        """Test OAuth support in experimental pass-through transformation"""
        config = AnthropicMessagesConfig()

        # Headers with OAuth Authorization token
        headers = {
            "authorization": "Bearer sk-ant-oat01-test-token-12345",
        }

        result_headers, result_api_base = config.validate_anthropic_messages_environment(
            headers=headers,
            model="claude-3-haiku-20240307",
            messages=[{"role": "user", "content": "test"}],
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base="https://api.anthropic.com",
        )

        # Should NOT have x-api-key header
        assert "x-api-key" not in result_headers
        # Should have Authorization header
        assert "authorization" in result_headers
        assert result_headers["authorization"] == "Bearer sk-ant-oat01-test-token-12345"

    def test_experimental_passthrough_api_key_support(self):
        """Test API key support still works in experimental pass-through"""
        config = AnthropicMessagesConfig()

        headers = {}

        result_headers, result_api_base = config.validate_anthropic_messages_environment(
            headers=headers,
            model="claude-3-haiku-20240307",
            messages=[{"role": "user", "content": "test"}],
            optional_params={},
            litellm_params={},
            api_key="sk-ant-api-key-12345",
            api_base="https://api.anthropic.com",
        )

        # Should have x-api-key header
        assert "x-api-key" in result_headers
        assert result_headers["x-api-key"] == "sk-ant-api-key-12345"

    def test_no_auth_raises_error(self):
        """When neither OAuth token nor API key is present, should raise error"""
        model_info = AnthropicModelInfo()

        headers = {
            "content-type": "application/json",
        }

        with pytest.raises(Exception) as exc_info:
            model_info.validate_environment(
                headers=headers,
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": "test"}],
                optional_params={},
                litellm_params={},
                api_key=None,
                api_base="https://api.anthropic.com",
            )

        # Should raise AuthenticationError
        assert "Missing Anthropic API Key" in str(exc_info.value) or "authentication" in str(exc_info.value).lower()
