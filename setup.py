from setuptools import setup, find_packages

setup(
    name='reactive-optical-flow',
    version='0.1.0',
    description='Reactive optical flow UAV navigation',
    packages=find_packages(exclude=['tests', 'flow_logs']),
    install_requires=[
        'opencv-python',
        'numpy',
        'airsim',
        'msgpack-rpc-python',
        'msgpack',
        'pandas',
        'plotly',
        'scipy',
    ],
    python_requires='>=3.8',
)
