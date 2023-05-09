collection = {}
count = 0

fhand = open("song.txt")

for line in fhand:
    line = line.rstrip()
    words = line.split()
    for word in words:
        if word in collection:
            collection[word] += 1
        else:
            collection[word] = 1

fhand.close()

for words in collection:
    if(collection[words] == 1):
        print(words)
        count += 1

print(count)