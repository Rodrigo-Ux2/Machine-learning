"""
Comprueba que la backpropagation de red_neuronal.py calcula el gradiente correcto.

Compara el gradiente analitico (retropropagar) contra el numerico por
diferencias finitas centradas, y verifica que la red aprende XOR, que un
perceptron de una sola capa no puede resolver.

Uso:  .venv/bin/python src/test_red_neuronal.py
"""

import numpy as np

from red_neuronal import PerceptronMulticapa


def gradiente_numerico(red, X, Y, eps=1e-6):
    numerico = []
    for W in red.W:
        g = np.zeros_like(W)
        for idx in np.ndindex(W.shape):
            original = W[idx]
            W[idx] = original + eps
            arriba = red.perdida(red.propagar(X)[-1], Y)
            W[idx] = original - eps
            abajo = red.perdida(red.propagar(X)[-1], Y)
            W[idx] = original
            g[idx] = (arriba - abajo) / (2 * eps)
        numerico.append(g)
    return numerico


def test_gradientes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(7, 4))
    casos = {'regresion': rng.normal(size=(7, 1)),
             'clasificacion': np.eye(3)[rng.integers(0, 3, 7)]}
    for tarea, Y in casos.items():
        red = PerceptronMulticapa([4, 5, 3, Y.shape[1]], tarea, alfa=0.1, semilla=1)
        analitico, _ = red.retropropagar(red.propagar(X), Y)
        for a, n in zip(analitico, gradiente_numerico(red, X, Y)):
            error = np.abs(a - n).max() / max(np.abs(n).max(), 1e-12)
            assert error < 1e-5, f"{tarea}: gradiente incorrecto (error relativo {error:.2e})"


def test_aprende_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    Y = np.eye(2)[[0, 1, 1, 0]]
    red = PerceptronMulticapa([2, 8, 2], 'clasificacion', tasa=0.1, alfa=0, semilla=3)
    red.entrenar(X, Y, X, Y, epocas=2000, lote=4, paciencia=2000)
    assert (red.predecir(X).argmax(axis=1) == [0, 1, 1, 0]).all(), "no aprendio XOR"


if __name__ == '__main__':
    test_gradientes()
    test_aprende_xor()
    print("OK: backpropagation coincide con el gradiente numerico y la red aprende XOR.")
