import pathlib

def load_system(path: pathlib.Path) -> tuple[list[list[float]], list[float]]:
    '''
    This funnction read the content of file.txt and parse the expressions to form 3X3 matrix `a` and 3X1 matrix(list) `b`
    3X3 matrix `a` contains the coefficients of x,y and z
    3X1 matrix(list) `b` contains the constants on the right side of equations
    '''
    content = pathlib.Path(path).read_text()

    # initialize 3X3 matrix `a` with zeros
    a = [[0,0,0] for _ in range(3)]

    # getting 3X1 matrix(list) `b` from the right side of equations
    b = [int(y) for x in content.split("\n") for y in x.split("=")[1].strip()]

    for (index,line) in enumerate(content.split("\n")):
        line = line.split("=")[0]
        terms = line.split(" ")
        operators = terms[1::2]
        operands = terms[::2]

        idx,x = next(((i, elem) for i, elem in enumerate(operands) if "x" in elem), (None, None))
        if x is not None:
            if x == "x":
                x = "1x"
            a[index][0] = int(str(x).replace("x",""))

        idx, y = next(((i, elem) for i, elem in enumerate(operands) if "y" in elem), (None, None))
        
        # for `y` and `z` we need to consider the operator before them
        # if `y` is freedom variable then it is `1y`
        # same for `z`
        if y is not None:
            if y == "y":
                y = "1y"
            y = ( "" if idx == 0 or operators[idx-1] == "+" else "-" ) + y
            a[index][1] = int(str(y).replace("y", ""))

        idx, z = next(((i, elem) for i, elem in enumerate(operands) if "z" in elem), (None, None))
        if z is not None:
            if z == "z":
                z = "1z"
            z =  ("" if idx == 0 or operators[idx - 1] == "+" else "-") + z
            a[index][2] = int(str(z).replace("z", ""))
            
    # return 3X3 matrix `a` and 3X1 matrix(list) `b`
    return (a,b)