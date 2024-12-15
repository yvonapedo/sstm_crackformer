import os
def do():
    # img_path = r'C:\Users\yvona\Documents\NPU_research\research3\domain adaptation\UCOS-DA-main\data\from_wkst\test_c200\images'
    img_path= r"C:\Users\yvona\Documents\NPU_research\research3\domain adaptation\UCOS-DA-main\data\from_wkst\crackTree200\image"
    for split in os.listdir(img_path):
        file_name = os.path.splitext(split)

        with open('train.txt', 'a') as f:
            # f.write("images_split/"+file_name[0] +'.jpg, masks_split/' + file_name[0] +'.png' +'\n')
            # f.write(r"C:\Users\yvona\Downloads\ct200\image/"+file_name[0] +".jpg "+r"C:\Users\yvona\Downloads\ct200\mask/"+file_name[0] +".png\n")
            f.write(r"C:\Users\yvona\Downloads\ct200\image/"+file_name[0] +".jpg\n")
            # f.write( file_name[0] +'.png\n')


if __name__ == '__main__':
    do()
