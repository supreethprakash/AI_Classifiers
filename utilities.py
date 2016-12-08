'''
This file contains the utility functions that can be used through out the assignment.
'''


def readFile(fileName):
	file = open(fileName, 'r')
	lines = file.readlines()
	file.close()
	return lines


def convertToRGB(fileName):
	RGBList = dict()
	contents = readFile(fileName)
	for eachLine in contents:
		rgbValues = ()
		imageVector = eachLine.split(' ')
		for i in range(2, len(imageVector), 3):
			rgbValues = rgbValues + (map(int, imageVector[i:i+3]), )
		RGBList[imageVector[0]] = (int(imageVector[1]), rgbValues)
	return RGBList




