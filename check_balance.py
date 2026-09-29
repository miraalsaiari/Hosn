import csv

stay = 0
handover = 0

with open("hosn_data.csv", "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        if row["decision"] == "STAY":
            stay += 1
        elif row["decision"] == "HANDOVER":
            handover += 1

print("STAY:", stay)
print("HANDOVER:", handover)
print("TOTAL:", stay + handover)