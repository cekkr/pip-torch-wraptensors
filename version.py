from typing import Optional

__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip']
__version__ = '2.3.0.dev20240308+rocm6.0'
debug = False
cuda: Optional[str] = None
git_version = '55e93013211e3d9a52e259aee5df4f6198a3cdcc'
hip: Optional[str] = '6.0.32830-d62f6a171'
