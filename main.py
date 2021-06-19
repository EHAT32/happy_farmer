import numpy as np
import sys
import pandas as pd

M = 1e5

#Parsing from excel table, creating restrictions

def parse_from_xlsx(file_path, variant):
    def formatnumbers(x):
        x = str(x).replace(',', '.')
        return float(x)

    def convert(df : pd.DataFrame):
        for col in df.columns:
            df[col] =  df[col].apply(formatnumbers)
        return df

    data_table_1 = convert(pd.read_excel(file_path, header = 1, nrows = 19, usecols = range(1,7))).to_numpy()
    data_table_2 = convert(pd.read_excel(file_path, header = 1, nrows = 19, usecols = range(7,13))).to_numpy()
    data_table_3 = convert(pd.read_excel(file_path, header = 1, nrows = 19, usecols = range(13,19))).to_numpy()

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
    for restriction_index in range(0, len(restrictions)):
        variables, sign, value = restrictions[restriction_index]
        for var in variables:
            index, multiplier = var
            table[restriction_index, index - 1] = multiplier
        table[restriction_index, -1] = value
        
        if sign == '<=':
            table[restriction_index, variables_count + extra_variables_offset] = 1
            extra_variables_offset += 1
        elif sign == '>=':
            table[restriction_index, variables_count + extra_variables_offset] = -1
            extra_variables_offset += 1
    return table

def get_basis_vars_indices(table):
    basis_vars_indices = []
    basis_var_rows = []
    for var_index in range(np.size(table, 1) - 1):
        col = table[:-1, var_index]
        nnz = np.nonzero(col)[0]
        if len(nnz) == 1 and col[nnz[0]] == 1:
            basis_vars_indices.append(var_index)
            basis_var_rows.append(nnz[0])
    return basis_vars_indices, basis_var_rows

def addMCoeffs(table):
    basis_vars_indices, _ = get_basis_vars_indices(table)

    rows_indices_with_M = []
    for row_index in range(np.size(table, 0) - 1):
        need_2_add_M_coeff = True
        for basis_var_index in basis_vars_indices:
            if table[row_index, basis_var_index] != 0:
                need_2_add_M_coeff = False
                break
        if need_2_add_M_coeff:
            rows_indices_with_M.append(row_index)

    rows, cols = table.shape
    table_with_M = np.zeros((rows, cols + len(rows_indices_with_M)))
    table_with_M[:, :cols - 1] = table[:, :cols - 1]
    table_with_M[:rows - 1, -1] = table[:rows - 1, -1]

    offset = 0
    for row_index in rows_indices_with_M:
        for coef_index in range(cols + offset):
            table_with_M[-1, coef_index] += table_with_M[row_index, coef_index] * -M
        table_with_M[-1, -1] += table_with_M[row_index, -1] * -M
        table_with_M[row_index, cols - 1 + offset] = 1
        offset += 1

    return table_with_M

def check_solution_exist(table):
    rows, _ = table.shape
    index_of_resolving_column = np.argmin(table[-1, :-1])
    for row in range(rows - 1):
        if table[row, index_of_resolving_column] > 0:
            return True
    return False

def check_optimal_solution(table):
    for var in table[-1, :-1]:
        if var < 0:
            return False
    return True

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
    rows, _ = table.shape
    min_column_index = np.argmin(table[-1, :-1])

    min_row_index = -1
    min = sys.float_info.max

    basis_vars_indices, _ = get_basis_vars_indices(table)
    basis_vars_indices = np.sort(basis_vars_indices)
    for row in range(rows - 1):
        if (table[row, min_column_index] <= 0):
            continue
        divided_value = table[row, -1] / table[row, min_column_index]

        if (abs(divided_value - min) < sys.float_info.epsilon):
            for basis_var_index in basis_vars_indices:
                if table[row, basis_var_index] == 1:
                    min_row_index = row
                    break

        if (divided_value < min):
            min = divided_value
            min_row_index = row

    resolving_element = table[min_row_index, min_column_index]
    new_table = np.zeros_like(table)
    new_table[min_row_index, :] = table[min_row_index, :] / resolving_element

    for row in range(rows):
        if row == min_row_index:
            continue
        factor = table[row, min_column_index] / resolving_element
        new_table[row, :] = table[row, :] - table[min_row_index, :] * factor

    return new_table

def simplex_method(table, desired_iterations = 100, maximize = True):
    result = np.copy(table)
    if not maximize:
        result[-1, :] *= -1

    iteration = 0
    does_solution_exist = check_solution_exist(result)
    is_plan_optimal = check_optimal_solution(result)
    while (iteration < desired_iterations and not is_plan_optimal and does_solution_exist):
        result = simplex_iteration(result)
        iteration += 1
        is_plan_optimal = check_optimal_solution(result)
        if (not is_plan_optimal):
            does_solution_exist = check_solution_exist(result)

    if not does_solution_exist:
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