import import_data
from operations import solve_cramer,solve,determinant

def main():
    a, b = import_data.load_system("Assignment1/data.txt")
    print("Matrix A:")
    for row in a:
        print(row)
    print("Matrix B:")
    print(b)

    print(f"Determinant of matrix A is: {determinant(a):3}")
    print(f"Solution with Cramer Rule:   {solve_cramer(a,b)}")
    print(f"Solution with inversion of matrix A: {solve(a,b)}")


    matrix = [
        [ 1, 2, 3, 4 ],
        [ 5, 6, 7, 8 ],
        [ 9, 1, 1, 2 ],
        [ 3, 4, 5, 6 ]
    ]
    print("\n\n\nExample matrix:")
    for row in matrix:
        print(row)
    print(f"The determinant for the example matrix is: {determinant(matrix)}")

# if this file is run directly, execute main()
if __name__ == "__main__":
    main()
