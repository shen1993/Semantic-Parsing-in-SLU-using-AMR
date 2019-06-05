with open('annotation2', 'r') as file:
    my_list = []
    full_list = []
    all_list = []
    pivot = False

    for line in file:
        if line.startswith("    :intent"):
            temp = line.replace("    :intent “", "").replace("”\n", "")
            my_list.append("# ::action " + temp)
        elif line.startswith("        :intent"):
            temp = line.replace("        :intent “", "").replace("”\n", "")
            my_list.append("# ::action " + temp)
        else:
            full_list.append(line.replace("\n", ""))

        if pivot:
            if not line.startswith("#"):
                print("!!")

        if line.startswith("# ::parameters"):
            pivot = True
        else:
            pivot = False



with open('annotation1', 'w') as file:
    for count, item in enumerate(full_list):
        for line in item:
            file.write(line)
        file.write('\n')

# with open('annotation2', 'w') as file:
#     for count, item in enumerate(all_list):
#         for line in item:
#             file.write(line)
#         file.write('\n')
