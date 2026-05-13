# Lazy import to avoid dependency issues when using MobileOVModel
def _get_omni_video_unified_gen():
    """Lazy import OmniVideoX2XUnified to avoid import errors when not needed."""
    from .omni_video_unified_gen import OmniVideoX2XUnified
    return OmniVideoX2XUnified

# Only import when explicitly requested
__all__ = ['_get_omni_video_unified_gen']