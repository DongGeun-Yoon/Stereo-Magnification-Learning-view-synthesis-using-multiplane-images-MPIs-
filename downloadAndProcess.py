import os
import cv2
from grpc import RpcContext

from pytube import YouTube
import glob, os
import pickle

# download 
def downloadVideo(videoPathURL):
	youtubeIDOffset = videoPathURL.find("/watch?v=") + len('/watch?v=')
	youtubeID = videoPathURL[youtubeIDOffset:]
	targetPath = "./downloaded/{}".format(youtubeID)
	if os.path.exists(targetPath) == False:
		yt = YouTube(videoPathURL)
		try: # vaild
			stream = yt.streams
			stream.get_by_itag(22).download('./downloaded', filename=youtubeID)
			return targetPath, youtubeID
		except: # not valid
			return None, None
	return targetPath, youtubeID

def getBestMatchingFrame(frameTimeStamp, case, maxFrameMatchingDistanceInNS=8000):
	for caseIdx, c in enumerate(case):
		distance = abs(c['timeStamp'] - frameTimeStamp)
		if distance < maxFrameMatchingDistanceInNS:
			print(c['timeStamp'], frameTimeStamp)
			print('case index', caseIdx, 'distance',distance)
			return caseIdx, distance
	return None, None

def delete_txt():
	txt = open('not_valid.txt', 'r')
	names = txt.readlines()
	for name in names:
		name = name.strip()
		try:
			os.remove(os.path.join(basePath, 'train', name))
		except:
			os.remove(os.path.join(basePath, 'test', name))
	txt.close()
	
if __name__ == "__main__":
	Trainset = True
	Testset = True
	video_delete = True

	no_valid = open('not_valid.txt', 'w') # pricacy video
	basePath = './real-estate-10k-run/RealEstate10K'
	outputResultPath = './real-estate-10k-run/transcode/'

	for rootPath in os.listdir(basePath): # train, test
		if not(Trainset) and ('train' in rootPath):
			continue
		if not(Testset) and ('test' in rootPath):
			continue

		subRootPath = os.path.join(basePath, rootPath)
		for subPath in os.listdir(subRootPath): 
			dataFilePath = os.path.join(subRootPath, subPath)
			print(dataFilePath)
			case = []
			with open(dataFilePath) as f:
				videoPathURL = f.readline().rstrip()
				# process all the rest of the lines 	
				for l in f.readlines():
					line = l.split(' ')

					timeStamp = int(line[0])
					intrinsics = [float(i) for i in line[1:7]]
					pose = [float(i) for i in line[7:19]]
					case.append({
						'timeStamp': timeStamp, 
						'intrinsics': intrinsics,
						'pose': pose})

			downloadedVideoPath, youtubeID = downloadVideo(videoPathURL)
			
			# No youtube available
			if downloadedVideoPath is None:
				print(subPath, file=no_valid)
				continue

			# build out the specific frames for the case
			video = cv2.VideoCapture(downloadedVideoPath) 
			video.set(cv2.CAP_PROP_POS_MSEC, 0) 

			while video.isOpened(): 
				frameOK, imgFrame = video.read() 
				if frameOK == False:
					print('video processing complete')
					break

				frameTimeStamp = (int)(round(video.get(cv2.CAP_PROP_POS_MSEC)*1000))
				caseOffset, distance = getBestMatchingFrame(frameTimeStamp, case)
				if caseOffset is not None:
					# match was successful, write frame
					imageOutputDir = os.path.join(outputResultPath, youtubeID)
					
					if not os.path.exists(imageOutputDir):
						os.makedirs(imageOutputDir)
					imageOutputPath = os.path.join(imageOutputDir, '{}.jpg'.format(case[caseOffset]['timeStamp']) )
					cv2.imwrite(imageOutputPath, imgFrame)
					case[caseOffset]['imgPath'] = imageOutputPath

			# write the case file to disk
			caseFileOutputPath = os.path.join(imageOutputDir, 'case.pkl')
			with open(caseFileOutputPath, 'wb') as f:
				pickle.dump(case, f)

	no_valid.close()
	if video_delete:
		delete_txt()