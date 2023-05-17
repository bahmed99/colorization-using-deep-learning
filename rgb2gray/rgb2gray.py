import sys
import os
import cv2 

if(len(sys.argv) != 3) : 
    sys.exit("*** error : Enter the command python ./rgb2gray.py <rgbImgPath> <grayscaleImgPath>")

input_path = str(sys.argv[1])
print ('Input path :', input_path)

output_path = str(sys.argv[2])
print ('Output path :', output_path)

if not os.path.exists(input_path):
    sys.exit("*** error : Input path doesn't exist")

if not os.path.exists(output_path):
   os.makedirs(output_path)
   print (output_path, 'created')

fileList=os.listdir(input_path)
tot = len(fileList)

print("nb images : ",tot)

for i, filename in enumerate(fileList) :
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img=cv2.imread(os.path.join(input_path,filename))
        gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(os.path.join(output_path,filename), gray_image)
    
    sys.stdout.write("\r{:.2f} %".format(((i+1)/tot*100))) 

print()    
