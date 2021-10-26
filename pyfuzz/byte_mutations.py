import random
import secrets
import struct

class Mutation(object):

    INTERESTING8 = [-128, -1, 0, 1, 16, 32, 64, 100, 127]
    INTERESTING16 = [0, 128, 255, 256, 512, 1000, 1024, 4096, 32767, 65535]
    INTERESTING32 = [0, 1, 32768, 65535, 65536, 100663045, 2147483647, 4294967295]

    def __init__(self):
        self.min_bytes_required = 0

    def mutate(self, input):
        "input : bytearray"
        if len(input) < self.min_bytes_required:
            return input
        # mutation logic here.
        return input

    @staticmethod
    def _rand(n):
        if n < 2:
            return 0
        return secrets.randbelow(n)

    @staticmethod
    def _choose_len(n):
        x = Mutation._rand(100)
        if x < 90:
            return Mutation._rand(min(8, n)) + 1
        elif x < 99:
            return Mutation._rand(min(32, n)) + 1
        else:
            return Mutation._rand(n) + 1

    @staticmethod
    def copy(src, dst, start_source, start_dst, end_source=None, end_dst=None):
        end_source = len(src) if end_source is None else end_source
        end_dst = len(dst) if end_dst is None else end_dst
        byte_to_copy = min(end_source-start_source, end_dst-start_dst)
        src[start_source:start_source+byte_to_copy] = dst[start_dst:start_dst+byte_to_copy]


class RemoveByteRange(Mutation):
    def __init__(self):
        self.min_bytes_required = 2

    def mutate(self, input):
        if len(input) < self.min_bytes_required:
            return input

        res = input[:]
        pos0 = self._rand(len(res))
        pos1 = pos0 + self._choose_len(len(res) - pos0)
        self.copy(res, res, pos1, pos0)
        res = res[:len(res) - (pos1-pos0)]
        return res

class InsertRandomBytes(Mutation):
    def __init__(self):
        self.min_bytes_required = 0

    def mutate(self, input):
        if len(input) < self.min_bytes_required:
            return input

        res = input[:]
        pos = self._rand(len(res) + 1)
        n = self._choose_len(10)
        for k in range(n):
            res.append(0)
        self.copy(res, res, pos, pos+n)
        for k in range(n):
            res[pos+k] = self._rand(256)
        return res

class DuplicateByteRange(Mutation):
    def __init__(self):
        self.min_bytes_required = 1

    def mutate(self, input):
        if len(input) < self.min_bytes_required:
            return input

        res = input[:]
        src = self._rand(len(res))
        dst = self._rand(len(res))
        while (src == dst) & (len(input)>1):
            dst = self._rand(len(res))
        n = self._choose_len(len(res) - src)
        tmp = bytearray(n)
        self.copy(res, tmp, src, 0)
        for k in range(n):
            res.append(0)
        self.copy(res, res, dst, dst+n)
        for k in range(n):
            res[dst+k] = tmp[k]
        return res

class CopyBytes(Mutation):
    def __init__(self):
        self.min_bytes_required = 1

    def mutate(self, input):
        if len(input) < self.min_bytes_required:
            return input

        res = input[:]
        src = self._rand(len(res))
        dst = self._rand(len(res))
        while (src == dst) & (len(input)>1):
            dst = self._rand(len(res))
        n = self._choose_len(len(res) - src)
        self.copy(res, res, src, dst, src+n)
        return res

class BitFlip(Mutation):
    def __init__(self):
        self.min_bytes_required = 1

    def mutate(self, input):
        if len(input) < self.min_bytes_required:
            return input

        res = input[:]
        pos = self._rand(len(res))
        res[pos] ^= 1 << self._rand(8)
        return res

class ByteToRandomValue(Mutation):
    def __init__(self):
        self.min_bytes_required = 1

    def mutate(self, input):
        if len(input) < self.min_bytes_required:
            return input

        res = input[:]
        pos = self._rand(len(res))
        res[pos] ^= self._rand(255) + 1
        return res

class SwapTwoBytes(Mutation):
    def __init__(self):
        self.min_bytes_required = 2

    def mutate(self, input):
        if len(input) < self.min_bytes_required:
            return input

        res = input[:]                
        src = self._rand(len(res))
        dst = self._rand(len(res))
        while src == dst:
            dst = self._rand(len(res))
        res[src], res[dst] = res[dst], res[src]
        return res

class AddOrSubtractFromByte(Mutation):
    def __init__(self):
        self.min_bytes_required = 1

    def mutate(self, input):
        if len(input) < self.min_bytes_required:
            return input

        res = input[:]                
        pos = self._rand(len(res))
        v = self._rand(2 ** 8)
        res[pos] = (res[pos] + v) % 256
        return res     

class AddOrSubtractFromNBytes(Mutation):
    def __init__(self, N=None):
        self.N = N
        self.min_bytes_required = 1

    def mutate(self, input):
        if len(input) < self.min_bytes_required:
            return input

        if not self.N:
            self.N = random.randint(1, len(input))

        self.N = len(input) if len(input) < self.N else self.N

        res = input[:]                
        pos = self._rand(len(res) - self.N)
        for i in range(self.N):
            v = self._rand(2 ** 8)
            res[pos + i] = (res[pos] + v) % 256
        return res   

class AddOrSubtractFromAllBytes(Mutation):
    def __init__(self):
        self.min_bytes_required = 1

    def mutate(self, input):
        if len(input) < self.min_bytes_required:
            return input

        res = input[:]                
        for i in range(len(res)):
            v = self._rand(2 ** 8)
            res[i] = (res[i] + v) % 256
        return res   

class ReplaceByteWithInterestingValue(Mutation):
    def __init__(self):
        self.min_bytes_required = 1

    def mutate(self, input):
        if len(input) < self.min_bytes_required:
            return input

        res = input[:] 
        pos = self._rand(len(res))               
        res[pos] = Mutation.INTERESTING8[self._rand(len(Mutation.INTERESTING8))] % 256
        return res   

class Replace2BytesWithInterestingValue(Mutation):
    def __init__(self):
        self.min_bytes_required = 2

    def mutate(self, input):
        if len(input) < self.min_bytes_required:
            return input

        res = input[:] 
        pos = self._rand(len(res) - 1)
        v = random.choice(Mutation.INTERESTING16)
        if bool(random.getrandbits(1)):
            v = struct.pack('>H', v)
        else:
            v = struct.pack('<H', v)
        for i in range(self.min_bytes_required):
            res[pos + i] = v[i] % 256
        return res   

class Replace4BytesWithInterestingValue(Mutation):
    def __init__(self):
        self.min_bytes_required = 4

    def mutate(self, input):
        if len(input) < self.min_bytes_required:
            return input

        res = input[:] 
        pos = self._rand(len(res) - 3)
        v = random.choice(Mutation.INTERESTING32)
        if bool(random.getrandbits(1)):
            v = struct.pack('>I', v)
        else:
            v = struct.pack('<I', v)
        for i in range(self.min_bytes_required):
            res[pos + i] = v[i] % 256
        return res   

class ReplaceASCIIDigits(Mutation):
    def __init__(self):
        self.min_bytes_required = 1

    def mutate(self, input):
        if len(input) < self.min_bytes_required:
            return input

        res = input[:]                
        digits = []
        for k in range(len(res)):
            if ord('0') <= res[k] <= ord('9'):
                digits.append(k)
        if len(digits) == 0:
            return input
        pos = self._rand(len(digits))
        was = res[digits[pos]]
        now = was
        while was == now:
            now = self._rand(10) + ord('0')
        res[digits[pos]] = now
        return res  

def mutate_bytes(input):
    """Return input (bytes) with a random mutation applied"""
    mutators = [
        RemoveByteRange(),
        InsertRandomBytes(),
        DuplicateByteRange(),
        CopyBytes(),
        BitFlip(),
        ByteToRandomValue(),
        SwapTwoBytes(),
        AddOrSubtractFromByte(),
        AddOrSubtractFromNBytes(),
        AddOrSubtractFromAllBytes(),
        ReplaceByteWithInterestingValue(),
        Replace2BytesWithInterestingValue(),
        Replace4BytesWithInterestingValue(),
        ReplaceASCIIDigits()
    ]
    mutator = random.choice(mutators)
    return mutator.mutate(input)