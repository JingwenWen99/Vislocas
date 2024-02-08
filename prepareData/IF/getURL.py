from multiprocessing import Pool, Lock
import csv

dataDir = "data/"

def saveUrl(fileName):
    with open(fileName, "r") as f1:
        reader = csv.reader(f1, delimiter=",")
        next(reader)
        for line in reader:
            urls = line[5].split(';')
            for url in urls:
                basicUrl = '_'.join(url.split('_')[:-3])
                with open(dataDir + "url.csv", "a", encoding="utf-8", newline="") as f2:
                    writer = csv.writer(f2)
                    writer.writerow([line[1], line[2], line[3], basicUrl + "_blue_red_green_yellow.jpg"])
                    writer.writerow([line[1], line[2], line[3], basicUrl + "_blue.jpg"])
                    writer.writerow([line[1], line[2], line[3], basicUrl + "_red.jpg"])
                    writer.writerow([line[1], line[2], line[3], basicUrl + "_green.jpg"])
                    writer.writerow([line[1], line[2], line[3], basicUrl + "_yellow.jpg"])


if __name__ == '__main__':
    saveUrl(dataDir + "location.csv")