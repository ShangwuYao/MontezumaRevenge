import os
path = './'
files = os.listdir(path)
i = 0

for file in files:
	name = file.split('.')[0]
	if file.endswith('.txt'):
		os.rename(os.path.join(path, file), os.path.join(path, str(i)+name+'.txt'))
	elif file.endswith('.png'):
		os.rename(os.path.join(path, file), os.path.join(path, str(i)+name+'.png'))
