
list = [[1,2,4]]
list2 = []

list2.append([1,2,3])
list.extend([[1,2,3], [3,4,5]]) if [1,2,3] not in list else list
print(list)
print(list2)



