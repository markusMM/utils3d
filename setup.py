from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='folding3d',
    ext_modules=[cpp_extension.CppExtension(
        'folding3d', ['folding3d_bindings.cpp', 'folding3d.cpp'])],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    version='1.2.0',
    requires=["torch"]
)

Extension(
    name='folding3d',
    sources=['folding3d_bindings.cpp', 'folding3d.cpp'],
    include_dirs=cpp_extension.include_paths(),
    language='c++'
)
