import glob
from sys import argv

assert len(argv) > 1, "specify directory name at minimum"
directory_name = argv[1]
if len(argv) == 2:
    outfile = open("jobfile.txt","w")
elif len(argv) == 3:
    outfile = open(argv[2],"w")
else:
    print("additional args ignored")

for path_name in sorted(glob.glob(directory_name + '/**/*[!visual].JPG', recursive = True)):
    outfile.write(f'python model_predict_func.py {path_name}\n')

outfile.close() 
