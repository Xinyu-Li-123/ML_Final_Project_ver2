import os

print("Hello world form Python :)")
for filename in os.listdir("./"):
    # checking if it is a file
    f = filename
    print(f, type(f), os.path.isfile(f), os.path.isdir(f))
    if os.path.isfile(f) and f.split(".")[-1] in ['jpg', 'png', 'jpeg', 'bmp']:
    
        # axes[1].imshow(cv.imread(f)[:,:,::-1])
        # axes[1].set_title(f.split("\\")[-1])
        # plt.show()        
        print(f)