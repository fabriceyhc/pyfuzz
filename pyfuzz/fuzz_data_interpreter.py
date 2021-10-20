import struct

class FuzzedDataInterpreter(object):
    def __init__(self, data, bytes_idx=0):
        self.data = data
        self.bytes_idx = bytes_idx

    def bytes_remaining(self):
        return len(self.data) - self.bytes_idx

    def claim_int(self, num_bytes=4):
        if self.bytes_remaining() < num_bytes:
            return 0.0

        if num_bytes == 1:
            struct_format = "B"
        elif num_bytes == 2:
            struct_format = "H"
        elif num_bytes == 4:
            struct_format = "I"
        elif num_bytes == 8:
            struct_format = "Q"
        else:
            raise ValueError("Unsupported num_bytes for claim_int. " +
                             "Must be either 1, 2, 4, or 8")
        out_bytes = self.data[self.bytes_idx:self.bytes_idx+num_bytes]
        out = struct.unpack(struct_format, out_bytes)[0]
        self.bytes_idx += num_bytes
        return out


    def claim_float(self):
        num_bytes = 4 # float always has 4 bytes
        struct_format = "f"
        if self.bytes_remaining() < num_bytes:
            return 0.0
        out_bytes = self.data[self.bytes_idx:self.bytes_idx+num_bytes]
        out = struct.unpack(struct_format, out_bytes)[0]
        self.bytes_idx += num_bytes
        return out


    def claim_double(self):
        num_bytes = 8 # double always has 8 bytes
        struct_format = "d"
        if self.bytes_remaining() < num_bytes:
            return 0.0
        out_bytes = self.data[self.bytes_idx:self.bytes_idx+num_bytes]
        out = struct.unpack(struct_format, out_bytes)[0]
        self.bytes_idx += num_bytes
        return out