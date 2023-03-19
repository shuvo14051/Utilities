# source folder with gif images
folder_path = "cancer"
# png images will be save in this folder
save_path = "cancer_png"

for filename in os.listdir(folder_path):
    if filename.endswith(".gif"):
        gif_path = os.path.join(folder_path, filename)
        png_path = os.path.join(save_path, filename.replace(".gif", ".png"))
        with Image.open(gif_path) as im:
            im.save(png_path)