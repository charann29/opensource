# Python script to merge two files line by line

# Define the file names
file1 = 'file1.txt'
file2 = 'file2.txt'
output_file = 'merged.txt'

# Open the files and output file
with open(file1, 'r') as f1, open(file2, 'r') as f2, open(output_file, 'w') as out_file:
    # Read lines from both files
    f1_lines = f1.readlines()
    f2_lines = f2.readlines()
    
    # Determine the maximum number of lines in both files
    max_lines = max(len(f1_lines), len(f2_lines))
    
    # Merge the files line by line
    for i in range(max_lines):
        if i < len(f1_lines):
            out_file.write(f1_lines[i].strip() + '\n')
        if i < len(f2_lines):
            out_file.write(f2_lines[i].strip() + '\n')

print(f"Files {file1} and {file2} have been merged into {output_file}.")
