import ctypes
import os

# Ruta a la biblioteca compartida compilada
lib_path = os.path.abspath("./build/libfunctions.so")
lib = ctypes.CDLL(lib_path)

# Definición de tipos de argumentos y resultado de la función `exponents`
lib.exponents.argtypes = [
    ctypes.c_float,           # M
    ctypes.c_char_p,          # typeM
    ctypes.c_float,           # SNR
    ctypes.c_float,           # R
    ctypes.c_float,           # N
    ctypes.POINTER(ctypes.c_float)  # results (puntero a 3 floats)
]
lib.exponents.restype = ctypes.POINTER(ctypes.c_float)  # También retorna ese puntero

def call_exponents(M: float, typeModulation: str, SNR: float, R: float, N: float) -> list[float]:
    """
    Llama a la función C++ y devuelve [Pe, exponent, rho].
    """
    # Preparamos el buffer de salida
    results = (ctypes.c_float * 3)()

    # Llamamos a la función C++
    lib.exponents(
        ctypes.c_float(M),
        typeModulation.encode('utf-8'),
        ctypes.c_float(SNR),
        ctypes.c_float(R),
        ctypes.c_float(N),
        results
    )

    return [results[i] for i in range(3)]
