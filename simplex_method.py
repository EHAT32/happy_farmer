import re
import numpy as np
import sys
import os
import pandas as pd

M = 1e5

def parse_from_xlsx(file_path, variant):
    data_table_1 = pd.read_excel(file_path, header = 1, nrows = 19, usecols = range(1,7)).to_numpy()
    data_table_2 = pd.read_excel(file_path, header = 1, nrows = 19, usecols = range(7,13)).to_numpy()
    data_table_3 = pd.read_excel(file_path, header = 1, nrows = 19, usecols = range(13,19)).to_numpy()

    restr_1 = data_table_1[variant]
    restr_2 = data_table_2[variant]
    opt = data_table_3[variant]

    restrictions = [(list(zip(range(1, len(restr_1) + 1), restr_1)), '<=', 10)]
    for i in range(len(restr_2)):
        restrictions.append(([(i + 1, 1)], '<=', restr_2[i]))
        restrictions.append(([(i + 1, 1)], '>=', 0))

    opt = (list(zip(range(1, len(opt) + 1), opt)), 'max')

    return opt, restrictions

def make_table(opt_func, restrictions):
    opt_func_variables, _ = opt_func
    variables_count = len(opt_func_variables) # count main variables
    extra_variables_count = 0
    for restriction in restrictions: # count extra variables for canonical representation
        _, sign, _ = restriction
        if sign != '=':
            extra_variables_count += 1
    
    table_cols = variables_count + extra_variables_count + 1 # + 1 is for a right value of equation
    table_rows = len(restrictions) + 1 # + 1 is for an optimization function

    table = np.zeros((table_rows, table_cols), dtype=float)
    for var in opt_func_variables:
        index, multiplier = var
        table[-1, index - 1] = -multiplier

    extra_variables_offset = 0
    for restrictionIdx in range(0, len(restrictions)):
        variables, sign, value = restrictions[restrictionIdx]
        for var in variables:
            index, multiplier = var
            table[restrictionIdx, index - 1] = multiplier
        table[restrictionIdx, -1] = value
        
        if sign == '<=':
            table[restrictionIdx, variables_count + extra_variables_offset] = 1
            extra_variables_offset += 1
        elif sign == '>=':
            table[restrictionIdx, variables_count + extra_variables_offset] = -1
            extra_variables_offset += 1
    return table

def get_basis_vars_indices(table):
    basis_vars_indices = []
    basis_var_rows = []
    for varIdx in range(np.size(table, 1) - 1):
        col = table[:-1, varIdx]
        nnz = np.nonzero(col)[0]
        if len(nnz) == 1 and col[nnz[0]] == 1:
            basis_vars_indices.append(varIdx)
            basis_var_rows.append(nnz[0])
    return basis_vars_indices, basis_var_rows

def addMCoeffs(table):
    basis_vars_indices, _ = get_basis_vars_indices(table)

    rowsIdsWithM = []
    for rowIdx in range(np.size(table, 0) - 1):
        need2AddMcoeff = True
        for basisVarIdx in basis_vars_indices:
            if table[rowIdx, basisVarIdx] != 0:
                need2AddMcoeff = False
                break
        if need2AddMcoeff:
            rowsIdsWithM.append(rowIdx)

    rows, cols = table.shape
    tableWithM = np.zeros((rows, cols + len(rowsIdsWithM)))
    tableWithM[:, :cols - 1] = table[:, :cols - 1]
    tableWithM[:rows - 1, -1] = table[:rows - 1, -1]

    offset = 0
    for rowIdx in rowsIdsWithM:
        for coefIdx in range(cols + offset):
            tableWithM[-1, coefIdx] += tableWithM[rowIdx, coefIdx] * -M
        tableWithM[-1, -1] += tableWithM[rowIdx, -1] * -M
        tableWithM[rowIdx, cols - 1 + offset] = 1
        offset += 1

    return tableWithM

def check_optimal_solution(table):
    for var in table[-1, :-1]:
        if var < 0:
            return False
    return True

def check_solution_exist(table):
    rows, _ = table.shape
    indexOfResolvingColumn = np.argmin(table[-1, :-1])
    for row in range(rows - 1):
        if table[row, indexOfResolvingColumn] > 0:
            return True
    return False

def check_infinity_solutions(table):
    rows, cols = table.shape
    basis_vars_indices, _ = get_basis_vars_indices(table)
    for i in range(cols - 1):
        if i in basis_vars_indices:
            continue
        if table[rows - 1, i] == 0:
            return True
    return False

def simplex_iteration(table):
    rows, cols = table.shape
    minColumnIndex = np.argmin(table[-1, :-1])

    minRowIndex = -1
    min = sys.float_info.max

    basis_vars_indices, _ = get_basis_vars_indices(table)
    basis_vars_indices = np.sort(basis_vars_indices)
    for row in range(rows - 1):
        if (table[row, minColumnIndex] <= 0):
            continue
        dividedValue = table[row, -1] / table[row, minColumnIndex]

        if (abs(dividedValue - min) < sys.float_info.epsilon):
            for basis_var_index in basis_vars_indices:
                if table[row, basis_var_index] == 1:
                    minRowIndex = row
                    break

        if (dividedValue < min):
            min = dividedValue
            minRowIndex = row

    resolvingElement = table[minRowIndex, minColumnIndex]
    newTable = np.zeros_like(table)
    newTable[minRowIndex, :] = table[minRowIndex, :] / resolvingElement

    for row in range(rows):
        if row == minRowIndex:
            continue
        factor = table[row, minColumnIndex] / resolvingElement
        newTable[row, :] = table[row, :] - table[minRowIndex, :] * factor

    return newTable

def simplex_method(table, desiredIterations = 100, maximize = True):
    result = np.copy(table)
    if not maximize:
        result[-1, :] *= -1

    iteration = 0
    isSolutionExists = check_solution_exist(result)
    isPlanOptimal = check_optimal_solution(result)
    while (iteration < desiredIterations and not isPlanOptimal and isSolutionExists):
        result = simplex_iteration(result)
        iteration += 1
        isPlanOptimal = check_optimal_solution(result)
        if (not isPlanOptimal):
            isSolutionExists = check_solution_exist(result)

    if not isSolutionExists:
        print("Solution doesn't exist")
        return result

    if check_infinity_solutions(result):
        print("There are infinity count of solutions")
        return result

    print("Optimal plan is found")
    return result

def main():
    np.set_printoptions(suppress=True,
        formatter={'float_kind':'{:.3f}'.format})
    opt_func, restrictions = parse_from_xlsx('FARMER.xlsx', 9)
    table = make_table(opt_func, restrictions)
    table = addMCoeffs(table)

    _, opt_type = opt_func
    maximize = True if opt_type == 'max' else False
    
    result = simplex_method(table, maximize=maximize)
    print(result)

    basis_indices, basis_rows = get_basis_vars_indices(result)
    print('basis_variables:')
    print(basis_indices)
    print('basis rows:')
    print(basis_rows)

    for i in range(len(basis_indices)):
        print('x{} = {}'.format(basis_indices[i] + 1, result[basis_rows[i], -1]))
    print('Profit: {}'.format(result[-1, -1]))
    return 0

if __name__ == '__main__':
    main()