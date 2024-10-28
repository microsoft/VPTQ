#!/bin/bash

# Format Python files using yapf
echo "Running yapf..."
yapf --recursive . --in-place

# Format Python imports using isort
echo "Running isort..."
isort .

# Format C++ files using clang-format
echo "Formatting C++ files..."
find csrc/ \( -name '*.h' -o -name '*.cc' -o -name '*.cu' -o -name '*.cuh' \) -print | xargs clang-format -i

echo "Formatting complete!"