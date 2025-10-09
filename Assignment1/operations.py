from math import sqrt

def check_if_square_matrix(matrix: list) -> bool:
    count_row = len(matrix)
    return all([True if count_row == len(lst) else False for lst in matrix])


def minor(matrix: list[list[float]], row: int, column: int) -> list[list[float]]:
    if not check_if_square_matrix(matrix):
        raise ValueError("The matrix parameter must be square")
    return [ [elem for j, elem in enumerate(r) if j != column] for i, r in enumerate(matrix) if i != row ]


# determinant for n-dimension square matrix
def determinant(matrix: list[list[float]]) -> float:
    if not check_if_square_matrix(matrix):
        raise ValueError("The matrix parameter must be square")
        
    # 2x2 matrix
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # iterate each element of the first line and moake n sub-matrixes 
    # with the size n-1 and calculate the determinant
    # apply LaPlace theorem !!
    return sum(list(map(lambda i: (-1)**i * matrix[0][i] * determinant(minor(matrix,0,i)),range(len(matrix)))))


# sum the elements from the main diagonal
def trace(matrix: list[list[float]]) -> float:
     if not check_if_square_matrix(matrix):
        raise ValueError("The matrix parameter must be square")
     return sum(list(map(lambda i: matrix[i][i],range(len(matrix)))))


# Euclidian Norm
def norm(vector: list[float]) -> float:
    return sqrt(sum(list(map(lambda el: el**2,vector))))


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [ [row[i] for row in matrix] for i in range(len(matrix)) ]


# dot product === inner product
def dot_product(vector1: list[float],vector2: list[float])->float:
    if len(vector1) != len(vector1):
        ValueError("The size of matrix isn't satisfiable for the vector given as parameter")
    return sum([a*b for (a,b) in zip(vector1,vector2)])


def multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    if len(matrix) != len(vector):
        ValueError("The size of matrix isn't satisfiable for the vector given as parameter")
    return [dot_product(row,vector) for row in matrix]


# Suppose that matrix is valid
def replace_column(matrix:list[list[float]],idx:int,vector: list[float])->list[list[float]]:
    if len(matrix)!=len(vector):
        raise ValueError("The number of rows in the matrix doesn't have the same size as size vector parameter")
    if abs(idx) >= len(matrix):
        raise ValueError("The index parameter is invalid for the vector and matrix data")
    
    result = [row[:] for row in matrix]
    for i in range(len(result)):
        result[i][idx] = vector[i]

    return result
    

def solve_cramer(matrix: list[list[float]], vector: list[float]) -> list[float]:
    # replace each columns with B list/vector and calculate the 'x' ,'y' and 'z' values
    return [ determinant(replace_column(matrix,i,vector)) / determinant(matrix) for i in range(3)]
    

def cofactor(matrix: list[list[float]]) -> list[list[float]]:
    #copy for input matrix
    result = [row[:] for row in matrix]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            result[i][j] = (-1)**(i+j) * determinant(minor(matrix,i,j))
    return result


def adjoint(matrix: list[list[float]]) -> list[list[float]]:
    return transpose(matrix)


def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    determinant = determinant(matrix)
    if determinant == 0:
        raise ValueError("The determinant for the imput matrix is zero")
    
    value = 1/determinant
    inverse_A = [ [value * el for el in row] for row in adjoint(cofactor(matrix))]

    return multiply(inverse_A,vector)
