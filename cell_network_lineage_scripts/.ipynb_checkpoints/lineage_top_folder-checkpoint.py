import glob
from sys import argv

assert len(argv) > 1, "(directory containing lineages) (optional: output jobfile name .txt) (rerun)"
directory_name = argv[1]

rerun = False

print(len(argv))
if len(argv) == 2:
    outfile = open("lineage_jobfile.txt","w")
elif len(argv) > 2:
    outfile = open(argv[2],"w")

if len(argv) == 4:
    rerun = True
elif len(argv) > 4:
    print("additional args ignored")

if rerun == False:
    for path_name in glob.glob(directory_name + '/*[!.lineage]'):
        outfile.write(f'python compute_lineage.py {path_name}\n')
else:
    for path_name in glob.glob(directory_name + '/*[!.lineage]'):
        outfile.write(f'python compute_lineage.py {path_name} rerun \n')

outfile.close()
