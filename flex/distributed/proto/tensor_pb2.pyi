from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Tensor(_message.Message):
    __slots__ = ("shape", "data")
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    shape: _containers.RepeatedScalarFieldContainer[int]
    data: bytes
    def __init__(self, shape: _Optional[_Iterable[int]] = ..., data: _Optional[bytes] = ...) -> None: ...
