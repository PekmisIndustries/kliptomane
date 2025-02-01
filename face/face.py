import os
import time
import subprocess
from sys import argv

start_time = time.time()

try:
    clip = argv[1]
    print("CLIPPPPPPPP",clip)
except:
    input("ON NE TRAITE PAS TON CLIP, CONTINUER?")
    clip = "../pclip.mp4"
#chemin absolu vers python
pathToPython = os.path.dirname(os.path.abspath(__file__)) + "/venv/Scripts/python.exe"

subprocess.run(["cmd", "/c", pathToPython, "extractFrames.py", clip])
print("1 -------------------------------")
subprocess.run(["cmd", "/c", pathToPython, "detectFaces.py"])
print("2 -------------------------------")
subprocess.run(["cmd", "/c", pathToPython, "analyzeEmotions.py"])
print("3 -------------------------------")
subprocess.run(["cmd", "/c", pathToPython, "aggregateScores.py", clip])
print("4 -------------------------------")

#delete full folder
subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", "temp_image"])
#print("supprimé les fichiers temporaires")


print(f"\n\nTemps total : {time.time() - start_time:.2f} secondes\n")

time.sleep(5)