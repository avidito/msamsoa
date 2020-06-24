import numpy as np
import pandas as pd

# Kombinasi Parameter Dasar
n_list = [3, 6, 9]

# Menghasilkan Kombinasi UAV yang Rusak
def generateBrokenUAVCombination(na):
    up = na//3
    broken_count = np.random.randint(1, up+1)
    broken_id = []
    for y in range(broken_count):
        a_id = np.random.randint(na)
        if(a_id not in broken_id):
            broken_id.append(a_id)

    for i in range(len(broken_id)):
        power = np.random.randint(100, 501)
        broken_id[i] = (broken_id[i], power)
    return broken_id

def simulateAlgorithm(algorithm, scenarios, repeat, broken=False):
    result = []
    for i in range(1, len(scenarios)):
        print("Skenario-{}".format(i))

        scenario = scenarios[i]
        for n in n_list:
            print("Jumlah UAV: {}. Pengujian ke-:".format(n), end=' ')

            for p in range(1, repeat+1):
                print("{}{}".format(p, ',' if p < repeat else '\n'), end='')
                sim = algorithm(scenario, n)
                if(broken):
                    broken_id = generateBrokenUAVCombination(n)
                    for (idx, power) in broken_id:
                        sim.agents[idx].power = power
                sim.execute()
                result.append((i, p, sim))
    return result

# Ekstraksi Informasi DF1
def extractDF1(result):
    df1 = pd.DataFrame({
        'skenario': [res[0] for res in result],
        'algoritma': [res[2].name for res in result],
        'jumlah_uav': [res[2].na for res in result],
        'pengujian': [res[1] for res in result],
        'waktu_surveillance': [res[2].tracker.ts for res in result],
        'waktu_pemupukan': [res[2].tracker.tf for res in result],
    })
    return df1

# Ekstraksi Informasi DF2
def extractDF2(result):
    df2 = pd.DataFrame({
        'skenario': [res[0] for res in result],
        'algoritma': [res[2].name for res in result],
        'jumlah_uav': [res[2].na for res in result],
        'pengujian': [res[1] for res in result],
        'tingkat_surveillance': [(res[2].tracker.s_tracks[-1]/res[2].size)*100 for res in result],
        'tingkat_pemupukan': [(res[2].tracker.f_tracks[-1]/res[2].nt)*100 for res in result],
    })
    return df2
