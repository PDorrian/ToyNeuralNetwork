import random
import numpy as np


class Matrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.data = [[0] * cols for _ in range(rows)]

    def randomize(self):
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] = random.uniform(-1, 1)

    @staticmethod
    def multiply(a, b):
        # Ensure matrices are the correct dimensions
        if a.cols != b.rows:
            print("Columns of A must match rows of B.")
        else:
            result = Matrix(a.rows, b.cols)
            for i in range(result.rows):
                for j in range(result.cols):
                    total = 0
                    for k in range(a.cols):
                        total += a.data[i][k] * b.data[k][j]
                    result.data[i][j] = total
            return result

    def scale(self, n):
        if type(n) == Matrix:
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] *= n.data[i][j]
        else:
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] *= n

    def add(self, n):
        # Element-wise addition
        if type(n) == Matrix:
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] += n.data[i][j]
        # Scalar addition
        else:
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] += n

    @staticmethod
    def subtract(a, b):
        if a.rows == b.rows and a.cols == b.cols:
            result = Matrix(a.rows, a.cols)
            for i in range(a.rows):
                for j in range(a.cols):
                    result.data[i][j] = a.data[i][j] - b.data[i][j]
            return result
        else:
            print("Matrices must have equal rows and columns for subtraction.")

    @staticmethod
    def transpose(matrix):
        result = Matrix(matrix.cols, matrix.rows)
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                result.data[j][i] = matrix.data[i][j]
        return result

    def print(self):
        for i in range(self.rows):
            for j in range(self.cols):
                print("{:8}".format(self.data[i][j]), end=" ")
            print("")

    def map(self, func):
        for i in range(self.rows):
            for j in range(self.cols):
                val = self.data[i][j]
                self.data[i][j] = func(val)

    @staticmethod
    def map_static(matrix, func):
        result = Matrix(matrix.rows, matrix.cols)
        for i in range(result.rows):
            for j in range(result.cols):
                val = matrix.data[i][j]
                result.data[i][j] = func(val)
        return result

    @staticmethod
    def from_array(arr):
        matrix = Matrix(len(arr), 1)
        for i in range(len(arr)):
            matrix.data[i][0] = arr[i]
        return matrix

    @staticmethod
    def to_array(matrix):
        arr = []
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                arr.append(matrix.data[i][j])
        return arr
