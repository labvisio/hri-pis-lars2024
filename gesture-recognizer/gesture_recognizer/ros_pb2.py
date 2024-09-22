# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: ros.proto

import sys

_b = sys.version_info[0] < 3 and (lambda x: x) or (lambda x: x.encode("latin1"))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import struct_pb2 as google_dot_protobuf_dot_struct__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
    name="ros.proto",
    package="is.ros",
    syntax="proto3",
    serialized_options=None,
    serialized_pb=_b(
        '\n\tros.proto\x12\x06is.ros\x1a\x1cgoogle/protobuf/struct.proto"D\n\nROSMessage\x12\x0c\n\x04type\x18\x01 \x01(\t\x12(\n\x07\x63ontent\x18\x02 \x01(\x0b\x32\x17.google.protobuf.Structb\x06proto3'
    ),
    dependencies=[
        google_dot_protobuf_dot_struct__pb2.DESCRIPTOR,
    ],
)


_ROSMESSAGE = _descriptor.Descriptor(
    name="ROSMessage",
    full_name="is.ros.ROSMessage",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name="type",
            full_name="is.ros.ROSMessage.type",
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=_b("").decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="content",
            full_name="is.ros.ROSMessage.content",
            index=1,
            number=2,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=51,
    serialized_end=119,
)

_ROSMESSAGE.fields_by_name["content"].message_type = (
    google_dot_protobuf_dot_struct__pb2._STRUCT
)
DESCRIPTOR.message_types_by_name["ROSMessage"] = _ROSMESSAGE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ROSMessage = _reflection.GeneratedProtocolMessageType(
    "ROSMessage",
    (_message.Message,),
    dict(
        DESCRIPTOR=_ROSMESSAGE,
        __module__="ros_pb2",
        # @@protoc_insertion_point(class_scope:is.ros.ROSMessage)
    ),
)
_sym_db.RegisterMessage(ROSMessage)


# @@protoc_insertion_point(module_scope)
