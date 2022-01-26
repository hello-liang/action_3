import io
import glob
dir=glob.glob("/media/liang/ssd/lrp_toolbox_heatmap/caffe-master-lrp/demonstrator/someimages/*.jpg")

fo = open("/media/liang/ssd/lrp_toolbox_heatmap/caffe-master-lrp/demonstrator/testfilelist.txt",'w')

print(dir)
# Write sequence of lines at the end of the file.
for item in dir:
	result=item.split('demonstrator')
	fo.write('.'+result[1]+" -2"+'\n')

# Close opend file
fo.close()
