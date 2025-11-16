from setuptools import setup, find_packages


setup(
    name="quadloco",
    version="0.1.0",
    author="pzy",
    description="Multi-gait locomotion project with diffgait, legged_gym, and rsl_rl",
    packages=find_packages(include=["legged_gym", "rsl_rl","legged_gym.*", "rsl_rl.*"]),
    python_requires=">=3.8",

)