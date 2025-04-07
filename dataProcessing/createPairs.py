#input file, output file
import sys 
import re

with open(sys.argv[1], 'r') as in_file, open(sys.argv[2], 'w') as out_file:
    out_file.write("syllabified\tnotSyllabified\n" )
    for line in in_file:
        out_file.write(line.strip() + '\t' + re.sub(r';', "", line))

out_file.close()
in_file.close()




