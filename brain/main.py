import os
from sys import argv
import time

start_time = time.time()

clip = input("place le clip > ")

os.chdir("../audio")
os.system("cmd /c python checkAudio.py "+clip)

os.chdir("../transcription")
os.system("cmd /c python transcription.py "+clip)
os.system("cmd /c python translate.py")
os.system("cmd /c python evaluatetext.py")

os.chdir("../face")
os.system("cmd /c python face.py "+clip)

os.chdir("../analysis")
os.system("cmd /c python graphAnalysis.py")


print(f"\n\nTemps total : {time.time() - start_time:.2f} secondes\n")

time.sleep(5)