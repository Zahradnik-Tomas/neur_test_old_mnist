import math
import random
import numpy
import json
import os

with numpy.load("mnist.npz", allow_pickle=True) as file:
    maticeCisel = file["x_train"]
    maticeReseni = file["y_train"]


class NeuronovaSit():
    def __init__(self, topologie):  # [2,4,1]
        self.koeficientUceni = 0.2
        self.sit = []
        posledni = 0
        for i in range(0, len(topologie) - 1):
            self.sit.append([])
            self.sit[posledni].append([])
            for j in range(topologie[i]):
                self.sit[posledni][0].append(0.0)
            self.sit[posledni][0].append(1.0)
            posledni += 1
            self.sit.append([])
            for j in range(topologie[i] + 1):
                self.sit[posledni].append([])
                for z in range(topologie[i + 1]):
                    self.sit[posledni][j].append(0.5 - random.random())
            posledni += 1
        self.sit.append([])
        self.sit[-1].append([])
        for i in range(topologie[-1]):
            self.sit[-1][0].append(0.0)

    def NasobeniMatic(self, matice1, matice2):
        temp = []
        for i in range(len(matice1)):
            temp.append([])
            for j in range(len(matice2[0])):
                temp[i].append(0.0)
                for z in range(len(matice1[0])):
                    temp[i][j] += matice1[i][z] * matice2[z][j]
        return temp

    def Sigmoid(self, input):
        return (1 / (1 + math.e ** -input))

    def Forward(self, inputVektor):
        self.input = inputVektor
        for i in range(len(self.sit[0][0]) - 1):
            self.sit[0][0][i] = inputVektor[i]
        for i in range(0, len(self.sit) - 2, 2):
            temp = self.NasobeniMatic(self.sit[i], self.sit[i + 1])
            for j in range(len(temp[0])):
                self.sit[i + 2][0][j] = self.Sigmoid(temp[0][j])

    def BackProp(self):
        tempSitChyb = []
        last = False
        for i in range(len(self.sit) - 2, 0, -2):
            tempSitChyb.append([])
            for j in range(len(self.sit[i][0])):
                if len(tempSitChyb[0]) == 0:
                    last = True
                if last:
                    tempSitChyb[-1].append(
                        (self.spravne[j] - self.sit[i + 1][0][j]) * self.sit[i + 1][0][j] * (1 - self.sit[i + 1][0][j]))
                else:
                    suma = 0
                    for error in range(len(tempSitChyb[-2])):
                        suma += self.sit[i + 2][j][error] * tempSitChyb[-2][error]
                    tempSitChyb[-1].append(suma * self.sit[i + 1][0][j] * (1 - self.sit[i + 1][0][j]))
            last = False
        momentalni = 0
        for i in range(len(self.sit) - 2, 0, -2):
            for j in range(len(self.sit[i])):
                for k in range(len(self.sit[i][0])):
                    self.sit[i][j][k] += self.koeficientUceni * tempSitChyb[momentalni][k] * self.sit[i - 1][0][j]
            momentalni += 1

    def ZiskejSpravnouVypovedniMatici(self, spravne):
        temp = []
        for i in range(10):
            if i == spravne:
                temp.append(1)
            else:
                temp.append(0)
        self.spravne = temp

    def Uloz(self):
        with open("vahy.txt", "w") as file:
            json.dump(self.sit, file, indent=1)

    def Load(self):
        if os.path.isfile("vahy.txt"):
            with open("vahy.txt", "r") as file:
                self.sit = json.load(file)


a = NeuronovaSit([784, 96, 10])

a.Load()


def Znormalizuj(vektor):
    temp = []
    for i in vektor:
        temp.append(i / 255)
    return temp


def Extenduj(matice):
    temp = []
    for i in matice:
        temp.extend(i)
    return temp


def train(iterace, sit):
    for i in range(iterace):
        inp = random.randint(0, len(maticeCisel) - 1)
        j = Extenduj(maticeCisel[inp])
        j = Znormalizuj(j)
        sit.Forward(j)
        sit.ZiskejSpravnouVypovedniMatici(maticeReseni[inp])
        sit.BackProp()
        print(i)


#train(5000, a)

#a.Uloz()

inp = random.randint(0, len(maticeCisel) - 1)
j = Extenduj(maticeCisel[inp])
j = Znormalizuj(j)
a.Forward(j)
print(a.sit[-1])
print(maticeReseni[inp])
