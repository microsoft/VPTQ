#!/bin/bash

# Format Python files using yapf
echo "Running yapf..."
find . -type f -name "*.py" \
    ! -path "./build/*" \
    ! -path "./.git/*" \
    ! -path "*.egg-info/*" \
    -print0 | xargs -0 yapf --in-place

# Format Python imports using isort
echo "Running isort..."
isort .

# Format C++ files using clang-format
echo "Formatting C++ files..."
find csrc/ \( -name '*.h' -o -name '*.cc' -o -name '*.cu' -o -name '*.cuh' \) -print | xargs clang-format -i

echo "Formatting complete!"
