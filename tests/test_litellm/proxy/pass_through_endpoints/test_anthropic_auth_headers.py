"""
Test authentication header handling in anthropic_proxy_route.

This test suite verifies OAuth pass-through support for Anthropic, ensuring:
1. Client Authorization header takes priority (OAuth pass-through)
2. Client x-api-key header takes priority (custom API key)
3. Server API key is used as fallback
4. No authentication injection when client provides auth
"""
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    anthropic_proxy_route,
)


class TestAnthropicAuthHeaders:
    """Test authentication header handling in anthropic_proxy_route."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock request object."""
        request = MagicMock()
        request.method = "POST"
        request.headers = {}
        return request

    @pytest.fixture
    def mock_response(self):
        """Create a mock FastAPI response object."""
        return MagicMock()

    @pytest.fixture
    def mock_user_api_key_dict(self):
        """Create a mock user API key dict."""
        return {"user_id": "test_user"}

    @pytest.mark.asyncio
    @patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route")
    @patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_streaming_request_fn")
    @patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router")
    async def test_client_authorization_header_priority(
        self,
        mock_router,
        mock_streaming,
        mock_create_route,
        mock_request,
        mock_response,
        mock_user_api_key_dict,
    ):
        """Test that client Authorization header takes priority over server key (OAuth pass-through)."""
        # Setup
        mock_request.headers = {"authorization": "Bearer sk-ant-oat01-client-oauth-token"}
        mock_router.get_credentials.return_value = "sk-ant-server-api-key"
        mock_streaming.return_value = False
        mock_endpoint_func = AsyncMock(return_value="test_response")
        mock_create_route.return_value = mock_endpoint_func

        # Act
        await anthropic_proxy_route(
            endpoint="v1/messages",
            request=mock_request,
            fastapi_response=mock_response,
            user_api_key_dict=mock_user_api_key_dict,
        )

        # Assert
        mock_create_route.assert_called_once()
        call_kwargs = mock_create_route.call_args[1]

        # Should NOT inject x-api-key when Authorization header is present
        assert call_kwargs["custom_headers"] == {}
        assert call_kwargs["_forward_headers"] is True

    @pytest.mark.asyncio
    @patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route")
    @patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_streaming_request_fn")
    @patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router")
    async def test_client_x_api_key_header_priority(
        self,
        mock_router,
        mock_streaming,
        mock_create_route,
        mock_request,
        mock_response,
        mock_user_api_key_dict,
    ):
        """Test that client x-api-key header takes priority over server key."""
        # Setup
        mock_request.headers = {"x-api-key": "sk-ant-client-api-key"}
        mock_router.get_credentials.return_value = "sk-ant-server-api-key"
        mock_streaming.return_value = False
        mock_endpoint_func = AsyncMock(return_value="test_response")
        mock_create_route.return_value = mock_endpoint_func

        # Act
        await anthropic_proxy_route(
            endpoint="v1/messages",
            request=mock_request,
            fastapi_response=mock_response,
            user_api_key_dict=mock_user_api_key_dict,
        )

        # Assert
        mock_create_route.assert_called_once()
        call_kwargs = mock_create_route.call_args[1]

        # Should NOT inject x-api-key when x-api-key header is present
        assert call_kwargs["custom_headers"] == {}
        assert call_kwargs["_forward_headers"] is True

    @pytest.mark.asyncio
    @patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route")
    @patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_streaming_request_fn")
    @patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router")
    async def test_server_api_key_fallback(
        self,
        mock_router,
        mock_streaming,
        mock_create_route,
        mock_request,
        mock_response,
        mock_user_api_key_dict,
    ):
        """Test that server API key is used when no client authentication is provided."""
        # Setup
        mock_request.headers = {}  # No authentication headers
        mock_router.get_credentials.return_value = "sk-ant-server-api-key"
        mock_streaming.return_value = False
        mock_endpoint_func = AsyncMock(return_value="test_response")
        mock_create_route.return_value = mock_endpoint_func

        # Act
        await anthropic_proxy_route(
            endpoint="v1/messages",
            request=mock_request,
            fastapi_response=mock_response,
            user_api_key_dict=mock_user_api_key_dict,
        )

        # Assert
        mock_create_route.assert_called_once()
        call_kwargs = mock_create_route.call_args[1]

        # Should inject server x-api-key when no client auth is provided
        assert call_kwargs["custom_headers"] == {"x-api-key": "sk-ant-server-api-key"}
        assert call_kwargs["_forward_headers"] is True

    @pytest.mark.asyncio
    @patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route")
    @patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_streaming_request_fn")
    @patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router")
    async def test_no_authentication_available(
        self,
        mock_router,
        mock_streaming,
        mock_create_route,
        mock_request,
        mock_response,
        mock_user_api_key_dict,
    ):
        """Test that no x-api-key header is added when no authentication is available."""
        # Setup
        mock_request.headers = {}  # No authentication headers
        mock_router.get_credentials.return_value = None  # No server key
        mock_streaming.return_value = False
        mock_endpoint_func = AsyncMock(return_value="test_response")
        mock_create_route.return_value = mock_endpoint_func

        # Act
        await anthropic_proxy_route(
            endpoint="v1/messages",
            request=mock_request,
            fastapi_response=mock_response,
            user_api_key_dict=mock_user_api_key_dict,
        )

        # Assert
        mock_create_route.assert_called_once()
        call_kwargs = mock_create_route.call_args[1]

        # Should NOT inject x-api-key when no auth is available
        assert call_kwargs["custom_headers"] == {}
        assert call_kwargs["_forward_headers"] is True

    @pytest.mark.asyncio
    @patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route")
    @patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_streaming_request_fn")
    @patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router")
    async def test_both_client_headers_present(
        self,
        mock_router,
        mock_streaming,
        mock_create_route,
        mock_request,
        mock_response,
        mock_user_api_key_dict,
    ):
        """Test that no server key is added when client has both auth headers."""
        # Setup
        mock_request.headers = {
            "authorization": "Bearer sk-ant-oat01-oauth-token",
            "x-api-key": "sk-ant-client-api-key"
        }
        mock_router.get_credentials.return_value = "sk-ant-server-api-key"
        mock_streaming.return_value = False
        mock_endpoint_func = AsyncMock(return_value="test_response")
        mock_create_route.return_value = mock_endpoint_func

        # Act
        await anthropic_proxy_route(
            endpoint="v1/messages",
            request=mock_request,
            fastapi_response=mock_response,
            user_api_key_dict=mock_user_api_key_dict,
        )

        # Assert
        mock_create_route.assert_called_once()
        call_kwargs = mock_create_route.call_args[1]

        # Should NOT inject server key when client has auth headers
        assert call_kwargs["custom_headers"] == {}
        assert call_kwargs["_forward_headers"] is True

    @pytest.mark.asyncio
    @patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route")
    @patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_streaming_request_fn")
    @patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router")
    async def test_case_insensitive_authorization_header(
        self,
        mock_router,
        mock_streaming,
        mock_create_route,
        mock_request,
        mock_response,
        mock_user_api_key_dict,
    ):
        """Test that authorization header check is case-insensitive."""
        # Setup - using uppercase Authorization
        mock_request.headers = {"Authorization": "Bearer sk-ant-oat01-oauth-token"}
        mock_router.get_credentials.return_value = "sk-ant-server-api-key"
        mock_streaming.return_value = False
        mock_endpoint_func = AsyncMock(return_value="test_response")
        mock_create_route.return_value = mock_endpoint_func

        # Act
        await anthropic_proxy_route(
            endpoint="v1/messages",
            request=mock_request,
            fastapi_response=mock_response,
            user_api_key_dict=mock_user_api_key_dict,
        )

        # Assert
        mock_create_route.assert_called_once()
        call_kwargs = mock_create_route.call_args[1]

        # Should NOT inject x-api-key (headers are case-insensitive in HTTP)
        # Note: FastAPI's Request.headers is case-insensitive
        assert call_kwargs["custom_headers"] == {}
        assert call_kwargs["_forward_headers"] is True
