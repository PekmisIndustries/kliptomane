import os
import time

start_time = time.time()


os.chdir("../audio")
os.system("cmd /c audio.bat")

os.chdir("../transcription")
os.system("cmd /c transcribe.bat")
os.system("cmd /c evaluate.bat")

os.chdir("../analysis")
os.system("cmd /c analyze.bat")



print(f"\n\nTemps total : {time.time() - start_time:.2f} secondes\n")

time.sleep(5)