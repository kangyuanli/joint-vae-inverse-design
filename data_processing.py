import numpy as np

def get_element_index(element_list, periodic_element_table):
    element_index_list = []
    for e_list in element_list:
        e_index_list = []
        for e in e_list:
            index = periodic_element_table.index(e)
            e_index_list.append(index)
        element_index_list.append(e_index_list)
    return element_index_list

def create_composition_matrix(element_index_list, Fraction, periodic_element_table):
    composition_matrix = np.zeros((len(element_index_list), len(periodic_element_table)))
    for sample in range(len(element_index_list)):
        index_list = element_index_list[sample]
        number = 0
        for index in index_list:
            composition_matrix[sample][index] = float(Fraction[sample][number])
            number += 1
    return composition_matrix

def get_based_element_data(Based_element):
    with open('.txt', 'r') as f:
        line_feature = f.readlines()

    with open('.txt', 'r') as f1:
        line_target = f1.readlines()

    Composition_list = []
    Fraction_list = []
    lnD_list = []

    for lines in line_feature[1:]:
        s = lines.split()
        composition_number = int(s[1])
        Composition_list.append(s[2:2 + composition_number])
        Fraction_list.append(s[2 + composition_number:2 + 2 * composition_number])

    for lines in line_target[1:]:
        s = lines.split()
        lnD_list.append(float(s[0]))

    sort_composition_list = []
    sort_fraction_list = []

    for i in range(len(Composition_list)):
        fraction_numbers = [float(x) for x in Fraction_list[i]]
        sorted_id = sorted(range(len(fraction_numbers)), key=lambda k: fraction_numbers[k], reverse=True)
        sort_composition_list.append([Composition_list[i][x] for x in sorted_id])
        sort_fraction_list.append([Fraction_list[i][x] for x in sorted_id])

    based_composition_list = []
    based_fraction_list = []
    based_target_list = []

    for i in range(len(Composition_list)):
        if sort_composition_list[i][0] == Based_element:
            based_composition_list.append(sort_composition_list[i])
            based_fraction_list.append(sort_fraction_list[i])
            based_target_list.append(lnD_list[i])

    All_element_list = []
    for i in range(len(based_composition_list)):
        for element in based_composition_list[i]:
            All_element_list.append(element)

    periodic_element_table = list(set(All_element_list))
    periodic_element_table.sort(key=All_element_list.index)

    element_index_list = get_element_index(based_composition_list, periodic_element_table)
    composition_matrix = create_composition_matrix(element_index_list, based_fraction_list, periodic_element_table)

    target_matrix = np.zeros((composition_matrix.shape[0], 1))
    for i in range(composition_matrix.shape[0]):
        target_matrix[i] = based_target_list[i]

    return composition_matrix, target_matrix, periodic_element_table

def get_based_element_feature_data(composition_matrix, periodic_element_table):
    import openpyxl

    wb = openpyxl.load_workbook('.xlsx')
    sheet = wb['Sheet1']

    data = [row for row in sheet.iter_rows(min_row=2, values_only=True)]
    wb.close()

    New_matrix = [x for x in data if x[0] in periodic_element_table]
    New_matrix.sort(key=lambda x: periodic_element_table.index(x[0]))

    def dot_product_sum(list1, list2):
        return sum([a * b for a, b in zip(list1, list2)])

    Feature_number = 6
    Feature_matrix = np.zeros((composition_matrix.shape[0], Feature_number))

    for sample_number in range(composition_matrix.shape[0]):
        for feature_index in range(Feature_number):
            mean_feature = dot_product_sum(list(composition_matrix[sample_number]), [x[feature_index + 1] for x in New_matrix])
            Feature_matrix[sample_number][feature_index] = mean_feature

    return Feature_matrix