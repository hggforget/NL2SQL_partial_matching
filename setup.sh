#!/bin/bash
cd NL2SQL_partial_matching/
pip3 install poetry
poetry install

# 获取 Poetry 管理的虚拟环境的路径
VENV_PATH=$(poetry env info -p)

# 激活虚拟环境
source "$VENV_PATH/bin/activate"

pip3 install /opt/tiger/NL2SQL_partial_matching/GMN/dependencies/torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl
pip3 install /opt/tiger/NL2SQL_partial_matching/GMN/dependencies/torch_sparse-0.6.17+pt113cu117-cp39-cp39-linux_x86_64.whl


# 运行你的 Python 脚本
# python3 GMN/train.py