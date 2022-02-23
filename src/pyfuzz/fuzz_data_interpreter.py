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

    def claim_probability(self):
        num_bytes = 4 
        if self.bytes_remaining() < num_bytes:
            return 0.0
        int_val = self.claim_int(num_bytes=4) # unsigned
        max_val = 2 ** 32 # max unsigned int
        return int_val / max_val

    def claim_float_in_range(self, min_val=0, max_val=100):
        assert max_val > min_val, "max_val must be greater than min_val"
        num_bytes = 4
        delta = max_val - min_val
        return delta * self.claim_probability()

    def claim_int_in_range(self, min_val=0, max_val=100):
        assert max_val > min_val, "max_val must be greater than min_val"
        num_bytes = 4
        return int(self.claim_float_in_range(min_val, max_val))


if __name__ == '__main__':
    
    claims = [
        'claim_int',
        'claim_float', 
        'claim_double',
        'claim_probability',
        'claim_float_in_range',
        'claim_int_in_range'
    ]

    for claim in claims:
        fdi = FuzzedDataInterpreter(b"\x01\x02\x03\x04\x05\x06\x07\x08")
        out = getattr(fdi, claim)()
        print(claim, out)