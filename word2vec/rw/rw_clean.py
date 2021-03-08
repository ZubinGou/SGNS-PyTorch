clean_f = open('rw_clean.txt', 'w')

with open('rw.txt', 'r') as f:
    for line in f.readlines():
        line = line.split()
        clean_f.write(f"{line[0]}\t{line[1]}\t{line[2]}\n")

clean_f.close()