# Quick Start Guide - OAuth PR Submission

This is a quick reference for submitting the OAuth pass-through PR.

## Current Status

✅ **Branch**: `feat/anthropic-oauth-proxy`
✅ **Commits**: 2 commits ready
✅ **Tests**: 7 comprehensive test cases
⏳ **PR**: Not yet submitted

## One-Command PR Submission

When ready to submit, run:

```bash
# 1. Push branch
git push -u origin feat/anthropic-oauth-proxy

# 2. Then visit: https://github.com/BerriAI/litellm/compare/main...petter-b:litellm:feat/anthropic-oauth-proxy
```

## PR Checklist

Before submitting:

### Pre-Submit
- [ ] Read `IMPLEMENTATION_SUMMARY.md` in this branch
- [ ] Review the code changes one more time
- [ ] Sign CLA: https://cla-assistant.io/BerriAI/litellm

### PR Form
Copy this for the PR description:

```markdown
## Title
feat(proxy): Add OAuth pass-through support for Anthropic

## Relevant issues
Fixes #13380
Supersedes #14821

## Pre-Submission checklist
- [x] Added testing in tests/test_litellm/ directory (7 test cases)
- [ ] Added screenshot of tests (will add when CI runs)
- [ ] PR passes all unit tests on make test-unit
- [x] PR scope is isolated - solves OAuth pass-through only

## Type
🆕 New Feature

## Changes

Implements OAuth pass-through for Anthropic proxy endpoint, allowing Claude Code subscription users to authenticate via OAuth tokens without requiring an API key.

### What Changed
- Modified `anthropic_proxy_route` to conditionally inject `x-api-key` header
- Only injects when BOTH `Authorization` and `x-api-key` headers are missing
- Maintains backward compatibility with API key authentication
- Enables OAuth tokens (`sk-ant-oat01-*`) to pass through untouched

### Testing
Added 7 comprehensive unit tests:
- ✅ OAuth token priority (Authorization header)
- ✅ Client API key priority (x-api-key header)
- ✅ Server API key fallback
- ✅ No authentication scenarios
- ✅ Multiple auth headers present
- ✅ Case-insensitive header checking

All tests use mocks (no real API calls).

### Why This Approach
- **Minimal scope**: Proxy endpoint only → easier review
- **Aligned with PR #14821**: Uses same logic as previously approved PR
- **Conservative**: Checks both Authorization and x-api-key headers
- **Real user need**: Claude Code subscriptions use OAuth tokens

### How It Works

**Before** (always injects x-api-key):
```python
custom_headers={"x-api-key": anthropic_api_key}
```

**After** (conditional injection):
```python
custom_headers = {}
if "authorization" not in request.headers and "x-api-key" not in request.headers:
    custom_headers["x-api-key"] = anthropic_api_key
```

### Usage Example

Claude Code users can now use OAuth:
```bash
export ANTHROPIC_BASE_URL="http://localhost:4000/anthropic"
claude  # Uses OAuth token, no API key needed!
```

Traditional API keys still work:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# Works as before
```

### Differences from PR #14821
This PR builds on #14821 with:
- Updated for current codebase structure
- Added case-insensitive header test
- Enhanced documentation

Core library OAuth support can follow as separate PR.
```

### After Submit
- [ ] Monitor CI status
- [ ] Respond to reviewer feedback
- [ ] Update PR if requested

## Quick Commands Reference

```bash
# Check current branch
git branch --show-current

# View commits
git log --oneline -3

# View changes
git diff main

# Run tests locally (if environment is set up)
make test-unit
make lint

# Push to GitHub
git push -u origin feat/anthropic-oauth-proxy
```

## Files Modified

```
✏️  litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py (+10 lines)
➕ tests/test_litellm/proxy/pass_through_endpoints/test_anthropic_auth_headers.py (270 lines)
📄 IMPLEMENTATION_SUMMARY.md (this document)
```

## Key Points for Reviewers

Mention these when responding to questions:

1. **Why proxy-only?**
   - Minimal scope for easier review
   - Matches user deployment scenario
   - Can add core library support in follow-up

2. **Why check BOTH headers?**
   - More defensive (prevents edge cases)
   - Aligned with previously approved PR #14821
   - Anthropic's priority behavior is undocumented

3. **Why this matters?**
   - Claude Code subscriptions use OAuth (not API keys)
   - Currently blocked from using LiteLLM proxy
   - Real user need (issue #13380)

4. **Test coverage?**
   - 7 comprehensive test cases
   - All scenarios covered
   - Fully mocked (no API dependencies)

## Success Indicators

After PR is merged:

✅ Claude Code users can proxy through LiteLLM
✅ API key authentication still works
✅ No breaking changes
✅ Issue #13380 resolved

## Contact

If maintainers have questions, suggest:
- Check `IMPLEMENTATION_SUMMARY.md` for detailed rationale
- All design decisions documented
- Happy to explain any choices made

## Next Phase (Optional)

After this merges, can propose:
- Core library OAuth support
- Beta headers with OAuth
- Documentation updates

But keep those separate PRs!

---

**Remember**: This is a focused, minimal PR. Keep scope isolated. Core library changes can come later.
