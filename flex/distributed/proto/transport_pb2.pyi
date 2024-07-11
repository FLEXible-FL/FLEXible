import tensor_pb2 as _tensor_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Error(_message.Message):
    __slots__ = ("reason",)
    REASON_FIELD_NUMBER: _ClassVar[int]
    reason: str
    def __init__(self, reason: _Optional[str] = ...) -> None: ...

class ClientMessage(_message.Message):
    __slots__ = ("handshake_res", "get_weights_res", "send_weights_res", "train_res", "error", "eval_res", "health_ins")
    class HandshakeRes(_message.Message):
        __slots__ = ("status",)
        STATUS_FIELD_NUMBER: _ClassVar[int]
        status: int
        def __init__(self, status: _Optional[int] = ...) -> None: ...
    class GetWeightsRes(_message.Message):
        __slots__ = ("weights",)
        WEIGHTS_FIELD_NUMBER: _ClassVar[int]
        weights: _containers.RepeatedCompositeFieldContainer[_tensor_pb2.Tensor]
        def __init__(self, weights: _Optional[_Iterable[_Union[_tensor_pb2.Tensor, _Mapping]]] = ...) -> None: ...
    class SendWeightsRes(_message.Message):
        __slots__ = ("status",)
        STATUS_FIELD_NUMBER: _ClassVar[int]
        status: int
        def __init__(self, status: _Optional[int] = ...) -> None: ...
    class TrainRes(_message.Message):
        __slots__ = ("metrics",)
        class MetricsEntry(_message.Message):
            __slots__ = ("key", "value")
            KEY_FIELD_NUMBER: _ClassVar[int]
            VALUE_FIELD_NUMBER: _ClassVar[int]
            key: str
            value: float
            def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
        METRICS_FIELD_NUMBER: _ClassVar[int]
        metrics: _containers.ScalarMap[str, float]
        def __init__(self, metrics: _Optional[_Mapping[str, float]] = ...) -> None: ...
    class EvalRes(_message.Message):
        __slots__ = ("metrics",)
        class MetricsEntry(_message.Message):
            __slots__ = ("key", "value")
            KEY_FIELD_NUMBER: _ClassVar[int]
            VALUE_FIELD_NUMBER: _ClassVar[int]
            key: str
            value: float
            def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
        METRICS_FIELD_NUMBER: _ClassVar[int]
        metrics: _containers.ScalarMap[str, float]
        def __init__(self, metrics: _Optional[_Mapping[str, float]] = ...) -> None: ...
    class HealthPing(_message.Message):
        __slots__ = ("status",)
        STATUS_FIELD_NUMBER: _ClassVar[int]
        status: int
        def __init__(self, status: _Optional[int] = ...) -> None: ...
    HANDSHAKE_RES_FIELD_NUMBER: _ClassVar[int]
    GET_WEIGHTS_RES_FIELD_NUMBER: _ClassVar[int]
    SEND_WEIGHTS_RES_FIELD_NUMBER: _ClassVar[int]
    TRAIN_RES_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    EVAL_RES_FIELD_NUMBER: _ClassVar[int]
    HEALTH_INS_FIELD_NUMBER: _ClassVar[int]
    handshake_res: ClientMessage.HandshakeRes
    get_weights_res: ClientMessage.GetWeightsRes
    send_weights_res: ClientMessage.SendWeightsRes
    train_res: ClientMessage.TrainRes
    error: Error
    eval_res: ClientMessage.EvalRes
    health_ins: ClientMessage.HealthPing
    def __init__(self, handshake_res: _Optional[_Union[ClientMessage.HandshakeRes, _Mapping]] = ..., get_weights_res: _Optional[_Union[ClientMessage.GetWeightsRes, _Mapping]] = ..., send_weights_res: _Optional[_Union[ClientMessage.SendWeightsRes, _Mapping]] = ..., train_res: _Optional[_Union[ClientMessage.TrainRes, _Mapping]] = ..., error: _Optional[_Union[Error, _Mapping]] = ..., eval_res: _Optional[_Union[ClientMessage.EvalRes, _Mapping]] = ..., health_ins: _Optional[_Union[ClientMessage.HealthPing, _Mapping]] = ...) -> None: ...

class ServerMessage(_message.Message):
    __slots__ = ("get_weights_ins", "send_weights_ins", "train_ins", "error", "eval_ins", "health_ins")
    class GetWeightsIns(_message.Message):
        __slots__ = ("status",)
        STATUS_FIELD_NUMBER: _ClassVar[int]
        status: int
        def __init__(self, status: _Optional[int] = ...) -> None: ...
    class SendWeightsIns(_message.Message):
        __slots__ = ("weights",)
        WEIGHTS_FIELD_NUMBER: _ClassVar[int]
        weights: _containers.RepeatedCompositeFieldContainer[_tensor_pb2.Tensor]
        def __init__(self, weights: _Optional[_Iterable[_Union[_tensor_pb2.Tensor, _Mapping]]] = ...) -> None: ...
    class TrainIns(_message.Message):
        __slots__ = ("status",)
        STATUS_FIELD_NUMBER: _ClassVar[int]
        status: int
        def __init__(self, status: _Optional[int] = ...) -> None: ...
    class EvalIns(_message.Message):
        __slots__ = ("status",)
        STATUS_FIELD_NUMBER: _ClassVar[int]
        status: int
        def __init__(self, status: _Optional[int] = ...) -> None: ...
    class HealthPing(_message.Message):
        __slots__ = ("status",)
        STATUS_FIELD_NUMBER: _ClassVar[int]
        status: int
        def __init__(self, status: _Optional[int] = ...) -> None: ...
    GET_WEIGHTS_INS_FIELD_NUMBER: _ClassVar[int]
    SEND_WEIGHTS_INS_FIELD_NUMBER: _ClassVar[int]
    TRAIN_INS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    EVAL_INS_FIELD_NUMBER: _ClassVar[int]
    HEALTH_INS_FIELD_NUMBER: _ClassVar[int]
    get_weights_ins: ServerMessage.GetWeightsIns
    send_weights_ins: ServerMessage.SendWeightsIns
    train_ins: ServerMessage.TrainIns
    error: Error
    eval_ins: ServerMessage.EvalIns
    health_ins: ServerMessage.HealthPing
    def __init__(self, get_weights_ins: _Optional[_Union[ServerMessage.GetWeightsIns, _Mapping]] = ..., send_weights_ins: _Optional[_Union[ServerMessage.SendWeightsIns, _Mapping]] = ..., train_ins: _Optional[_Union[ServerMessage.TrainIns, _Mapping]] = ..., error: _Optional[_Union[Error, _Mapping]] = ..., eval_ins: _Optional[_Union[ServerMessage.EvalIns, _Mapping]] = ..., health_ins: _Optional[_Union[ServerMessage.HealthPing, _Mapping]] = ...) -> None: ...
