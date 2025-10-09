import import_data
from operations import solve_cramer,solve,determinant

def main():
    a, b = import_data.load_system("Assignment1/data.txt")
    print("Matrix A:")
    for row in a:
        print(row)
    print("Matrix B:")
    print(b)

    print(f"Determinant of matrix a is: {determinant(a)}")
    print(solve_cramer(a,b))
    print(solve(a,b))

# if this file is run directly, execute main()
if __name__ == "__main__":
    main()