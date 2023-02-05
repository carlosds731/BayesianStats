# The file should in the same folder as your code
file_name = 'juliet_is_late_by.txt'
file1 = open(file_name, 'r')
Lines = file1.readlines()
data = list()
for line in Lines:
    data.append(float(line))