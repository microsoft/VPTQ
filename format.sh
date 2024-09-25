yapf  --recursive . --style='{based_on_style: google, column_limit: 120, indent_width: 4}' -i

isort .

find csrc/ \( -name '*.h' -o -name '*.cc' -o -name '*.cu' -o -name '*.cuh' \) -print | xargs clang-format -i
