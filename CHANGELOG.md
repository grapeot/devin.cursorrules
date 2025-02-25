# Changelog

## [Unreleased]

### Added
- Comprehensive documentation for reasoning tokens across the codebase
- Detailed test cases for token tracking with different providers
- Clear docstrings explaining provider-specific token tracking behavior

### Changed
- Updated `query_llm` function to properly handle reasoning tokens for o1 model
- Improved test coverage for token tracking across all providers
- Enhanced documentation in test files to clarify token tracking behavior

### Fixed
- Proper handling of reasoning tokens for non-o1 models (explicitly set to None)
- Token tracking tests to verify correct behavior for all providers 