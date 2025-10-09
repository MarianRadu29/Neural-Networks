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
    print(f"Solution with Cramer Rule: \t\t{solve_cramer(a,b)}")
    print(f"Solution with inversion of matrix A: \t{solve(a,b)}")

# if this file is run directly, execute main()
if __name__ == "__main__":
    main()