import numpy as np

# Skenario Lahan Pertanian
sc1 = np.full((100, 100), 1, np.uint8)
sc1[:15, :15] = sc1[:15, 85:] = sc1[85:, :15] = sc1[85:, 85:] = sc1[30:71, 30:71] = 0

sc2 = np.full((100, 100), 0, np.uint8)
sc2[40:60] = sc2[:, 40:60] = 1

sc3 = np.full((100, 100), 1, np.uint8)
sc3[45:90, 10:55] = sc3[:10, 90:] = sc3[:15, :15] = sc3[10:25, 45:60] = sc3[35:50, 90:] = 0
sc3[60:75, 70:85] = sc3[90:, 90:] = 0

np.random.seed(0)
sc4 = np.full((100, 100), 0, np.uint8)
for i in range(10):
    for j in range(10):
        rand_cls = np.random.randint(0,2)
        for r in range((i*10), (i+1)*10):
            for c in range((j*10), (j+1)*10):
                sc4[r,c] = rand_cls

scenarios = [sc1, sc2, sc3, sc4]
