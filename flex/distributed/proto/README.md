In order to create the proto files in your computer run the following command in this directory, having previously installed grpc-tools.

```shell
python -m grpc_tools.protoc -I../../proto/flex/proto/ --python_out=. --pyi_out=. --grpc_python_out=. ../../proto/flex/proto/*.proto
```
Run the following command in case that test yield import errors
```shell
grep -Rl "import .*_pb2" . | xargs sed -i -E 's/^import ([a-zA-Z0-9_]+_pb2)( as ([a-zA-Z0-9_]+))?$/from . import \1\2/'
```