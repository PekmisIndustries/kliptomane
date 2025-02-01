import matplotlib.pyplot as plt
import os


def CleanLine(line):
    line = line.strip()
    return line



audioResultsPath = "../audio/scored_segments.txt"

with open(audioResultsPath, "r") as audiof:
    content = audiof.readlines()
    audioAxisGraph = []
    audioOrdinateGraph = []
    for i in range(len(content)): #i is the second we're working on
        currentLine = CleanLine(content[i])

        part = 0
        audioAxisGraph.append(i)

        for j in currentLine:
            if part == 0 and j == ",": #we're at the comma
                part = 1
            elif part == 1: #we're just before the score
                part = 2
            elif part == 2: #we're at the score
                audioOrdinateGraph.append(float(j))
                


textResultsPath = "../transcription/scored_intervals.txt"

with open(textResultsPath, "r") as textf:
    content = textf.readlines()
    textAxisGraph = []
    textOrdinateGraph = []
    for i in range(len(content)): #i is the 2 seconds we're working on

        textAxisGraph.append(i*2+1)
        lineScore = ""
        currentLine = content[i]

        for j in currentLine:
            if not j == "|":
                lineScore += j 
            else:
                textOrdinateGraph.append(float(lineScore))
                break
            

faceResultsPath = "../face/scored_intervals.txt"

with open(faceResultsPath, "r") as facef:
    content = facef.readlines()
    faceAxisGraph = []
    faceOrdinateGraph = []
    for i in range(len(content)): #i is the second we're working on

        faceAxisGraph.append(i)
        lineScore = ""
        currentLine = content[i]

        for j in currentLine:
            if not j == ",":
                lineScore += j 
            else:
                faceOrdinateGraph.append(float(lineScore))
                break



averageAxisGraph = []
averageOrdinateGraph = []


# Calculate the average scores
max_length = max(len(audioOrdinateGraph), len(textOrdinateGraph) * 2)
for k in range(max_length):
    if k % 2 == 1:
        averageAxisGraph.append(k)
        audio_score = audioOrdinateGraph[k] if k < len(audioOrdinateGraph) else 0
        text_score = textOrdinateGraph[k // 2] if k // 2 < len(textOrdinateGraph) else 0
        face_score = faceOrdinateGraph[k] if k < len(faceOrdinateGraph) else 0
        averageOrdinateGraph.append((audio_score + 2*text_score + 2.5*face_score) / 5.5) # modifier les coeffs

# Clear out range
textAxisGraph = textAxisGraph[:max(audioAxisGraph) // 2+1]
textOrdinateGraph = textOrdinateGraph[:max(audioAxisGraph) // 2+1]
averageAxisGraph = averageAxisGraph[:max(audioAxisGraph) // 2+1]
averageOrdinateGraph = averageOrdinateGraph[:max(audioAxisGraph) // 2+1]



# Create the graph
thickness = 1.5
marker_size = 3

plt.figure(figsize=(20, 6), facecolor='black', edgecolor='black')
plt.gca().set_facecolor('black')
plt.gca().tick_params(axis='x', colors='white')
plt.gca().tick_params(axis='y', colors='white')
plt.gca().grid(color='dimgray')
plt.xlabel('Temps', color='white')
plt.ylabel('Scores', color='white')
plt.title('Intérêt par temps', color='white')

plt.plot(audioAxisGraph, audioOrdinateGraph, label='Audio', color='mediumorchid', marker='o', linestyle='-', linewidth=thickness, markersize=marker_size)
plt.plot(textAxisGraph, textOrdinateGraph, label='Transcription', color='dodgerblue', marker='o', linestyle='-', linewidth=thickness, markersize=marker_size)
plt.plot(faceAxisGraph, faceOrdinateGraph, label='Visage', color='springgreen', marker='o', linestyle='-', linewidth=thickness, markersize=marker_size)
plt.plot(averageAxisGraph, averageOrdinateGraph, label='Moyenne', color='pink', marker='o', linestyle='-', linewidth=3.5, markersize=marker_size)
plt.xlim(0, min(len(textAxisGraph) * 2, len(audioAxisGraph)))
plt.grid(True)
plt.legend(facecolor='black', edgecolor='black', labelcolor='white', loc='upper right')
plt.tight_layout()
plt.savefig("output.png", dpi=300, bbox_inches='tight')
os.system("output.png")