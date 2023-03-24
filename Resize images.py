width = 256
height = 256


i =  1
for filename in os.listdir(data_dir+'/NORMAL'):
    image = Image.open(data_dir+'/NORMAL/'+filename)
    resized_image = image.resize((width, height))
    resized_image.save("CX2/NORMAL/normal{}.png".format(i))
    i+=1
