import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Print directory tree
def print_directory_tree(root, space_depth=0):
    elements = os.listdir(root)
    print(" "*4*space_depth + root + f" ({len(elements)} elements)")
    for el in elements:
        path = os.path.join(root, el)
        if os.path.isdir(path):
            print_directory_tree(path, space_depth+1)

# Print the nth file from the given directory
def print_file_from_dir(dir_path, file_n=0):
    root, dirs, files = next(os.walk(dir_path))
    if len(files) > file_n:
        file_to_print = os.path.join(dir_path, files[file_n])
        print(f"File name: {file_to_print}")
        if file_to_print.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = mpimg.imread(file_to_print)
            imgplot = plt.imshow(img)
            plt.show()
        else:
            with open(file_to_print, 'r') as f:
                print(f.read())
    else:
        print("Error: there are not so many files")
