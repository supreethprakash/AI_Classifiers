'''
This file contains the utility functions that can be used through out the assignment.
'''
import random
import string

def readFile(fileName):
	file = open(fileName, 'r')
	lines = file.readlines()
	file.close()
	return lines


def convertToRGB(fileName):
	RGBList = dict()
	ctr = 0
	contents = readFile(fileName)
	for eachLine in contents:
		rgbValues = () 
		imageVector = eachLine.split(' ')
		for i in range(2, len(imageVector), 3):
			rgbValues = rgbValues + (findIntensity(map(int,imageVector[i:i+3])), )
		RGBList[ctr] = (int(imageVector[1]), rgbValues)
		ctr += 1
	return RGBList

def findIntensity(pixel):
	return (0.2989 * pixel[0]) + (0.5870 * pixel[1]) + (0.1140 * pixel[2])