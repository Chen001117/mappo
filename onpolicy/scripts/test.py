import os 

path = "./results/two_agent"
file_list = os.listdir(path)

for i in range(25):
    old_name = path + os.sep + "{:04d}.json".format(i)
    new_name = path + os.sep + "{:04d}.json".format(i+150)
    os.rename(old_name, new_name)