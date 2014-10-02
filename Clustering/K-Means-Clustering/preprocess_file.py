__author__ = 'kesavsundar'
with open("/home/kesavsundar/machinelearning/data/20newsgroup/processed_data/train.txt", 'r') as dat_file_read:
    lines = dat_file_read.readlines()
dat_file_read.close()

len_of_total_data = len(lines)
print "Length of the file is :", len(lines)
ten_percent = len_of_total_data * .01
each_data = ten_percent / 8.0
dict_labels = dict()
with open("/home/kesavsundar/machinelearning/data/20newsgroup/processed_data/train.txt", 'w') as dat_file_write:
    for line in lines:
        label = line.split(' ')[0]
        try:
            dict_labels[label] += 1
            if dict_labels[label] <= each_data:
                dat_file_write.write(line)
        except KeyError:
            dict_labels[label] = 1