#simulation vs theory
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import time
from tqdm import tqdm
import os

def L(r, alpha, beta):
    return np.exp(-alpha * np.power(r, beta))

def laplace_transform(s, lambda_, R, P0, alpha, beta, epsilon, djmax, stepdj, steprj):
    def inner_integral1(dj):
        rj_values = np.arange(dj, djmax, steprj)
        L_rj_values = L(rj_values, alpha, beta)
        denominator = (1 + s * P0 * L_rj_values)
        denominator[denominator < 1e-10] = 1e-10
        inner_values = L_rj_values * (1 / denominator) * rj_values
        return np.sum(inner_values) * steprj

    def inner_integral2(dj):
        rj_values = np.arange(dj, djmax, steprj)
        L_rj_values = L(rj_values, alpha, beta)
        L_dj = L(dj, alpha, beta * epsilon)
        L_dj = max(L_dj, 1e-10)
        denominator = (1 + s * P0 * (L_rj_values / L_dj))
        denominator[denominator < 1e-10] = 1e-10
        inner_values = (L_rj_values / L_dj) * (1 / denominator) * rj_values
        return np.sum(inner_values) * steprj

    def outer_integral1():
        dj_values = np.arange(0.0001, R, stepdj)
        outer_values = np.array([inner_integral1(dj) * dj * np.exp(-np.pi * lambda_ * dj**2) for dj in dj_values])
        return np.sum(outer_values) * stepdj

    def outer_integral2():
        dj_values = np.arange(R, djmax, stepdj)
        outer_values = np.array([inner_integral2(dj) * dj * np.exp(-np.pi * lambda_ * dj**2) for dj in dj_values])
        return np.sum(outer_values) * stepdj
    expterm = np.exp(-np.pi * lambda_ * R**2)
    return np.exp(-((2 * np.pi * lambda_)**2) * s * P0 * (  outer_integral1() + outer_integral2()))

def integral(func, a, b, step):
    x = np.arange(a, b, step)
    y = np.array([func(xi) for xi in x])
    return np.sum(y) * step

def P_e(T, lambda_, R, gamma, P0, alpha, beta, epsilon, rmax, stepr, djmax, stepdj, steprj):
    def integrand(r):
        L_r = L(r, alpha, beta ) * np.exp(alpha * np.power(r, beta * epsilon))
        L_r = max(L_r, 1e-10)
        return 2 * np.pi * lambda_ * r * np.exp(-T / (gamma * L_r)) * laplace_transform(T / (P0 * L_r), lambda_, R, P0, alpha, beta, epsilon, djmax, stepdj, steprj) * np.exp(-np.pi * lambda_ * r**2)
    return integral(integrand, R, rmax, stepr)

def P_c(T, lambda_, R, gamma, P0, alpha, beta, stepr, djmax, stepdj, steprj):
    def integrand(r):
        L_r = L(r, alpha, beta)
        L_r = max(L_r, 1e-10)
        return 2 * np.pi * lambda_ * r * np.exp(-T / (gamma * L_r)) * laplace_transform(T / (P0 * L_r), lambda_, R, P0, alpha, beta, 0.001, djmax, stepdj, steprj) * np.exp(-np.pi * lambda_ * r**2)
    return integral(integrand, 0.0001, R, stepr)

def P(T, lambda_, R, gamma, P0, alpha, beta, epsilon, rmax, stepr, djmax, stepdj, steprj):
    expterm = np.exp(-np.pi * lambda_ * R**2)
    return P_c(T, lambda_, R, gamma, P0, alpha, beta, stepr, djmax, stepdj, steprj) + P_e(T, lambda_, R, gamma, P0, alpha, beta, epsilon, rmax, stepr, djmax, stepdj, steprj)

def calculate_P_for_T(args):
    T, lambda_, R, gamma, P0, alpha, beta, epsilon, rmax, stepr, djmax, stepdj, steprj = args
    worker_id = os.getpid()
    print(f"[THEORY] Worker {worker_id} dang xu ly T = {10*np.log10(T):.2f} dB")
    result = P(T, lambda_, R, gamma, P0, alpha, beta, epsilon, rmax, stepr, djmax, stepdj, steprj)
    return result

# Ham mo phong Monte Carlo song song
def simulate_batch(args):
    batch_size, T_vals, lambda_, alpha, beta, gamma, epsilon, diskRadius, R = args
    worker_id = os.getpid()
    print(f"[SIMULATION] Worker {worker_id} dang xu ly batch kich thuoc {batch_size}")

    diskArea = np.pi * diskRadius**2
    countCover = np.zeros(len(T_vals))

    for _ in range(batch_size):
        rUser = np.sqrt(-np.log(1 - np.random.rand()) / (np.pi * lambda_))
        PUser = np.where(rUser > R, np.exp(alpha * rUser ** (epsilon * beta)), 1)
        servingSignal = PUser * np.random.exponential(1) * np.exp(-alpha * np.power(rUser, beta))

        randNumb = np.random.poisson(diskArea * lambda_)
        interference_sum = 0

        if randNumb > 1:
            rInterUsers = np.sqrt(-np.log(1 - np.random.rand(randNumb - 1)) / (np.pi * lambda_))
            PInterUsers = np.where(rInterUsers > R, np.exp(alpha * rInterUsers ** (epsilon * beta)), 1)
            rRand = np.zeros(randNumb - 1)

            for indexd in range(randNumb - 1):
                rRand[indexd] = diskRadius * np.sqrt(np.random.rand())
                while rRand[indexd] < rInterUsers[indexd]:
                    rRand[indexd] = diskRadius * np.sqrt(np.random.rand())

            InterferencePower = PInterUsers * np.random.exponential(1,randNumb - 1) * np.exp(-alpha * rRand ** beta)
            interference_sum = np.sum(InterferencePower)

        for idx, T in enumerate(T_vals):
            SINR = servingSignal / (interference_sum + 1 / gamma)
            if SINR >= T:
                countCover[idx] += 1

    return countCover

def monte_carlo_simulation_parallel(T_vals, lambda_, alpha, beta, gamma, epsilon, simNumb, diskRadius, R):
    num_cores = cpu_count()
    print(f"[SIMULATION] Dang su dung {num_cores} CPU cores cho mo phong Monte Carlo")

    # Determine batch size for each process
    batch_size_per_process = simNumb // num_cores
    remainder = simNumb % num_cores

    # Prepare arguments for each process
    args_list = []
    for i in range(num_cores):
        # Add the remainder to the first process
        current_batch_size = batch_size_per_process + (1 if i < remainder else 0)
        args_list.append((current_batch_size, T_vals, lambda_, alpha, beta, gamma, epsilon, diskRadius, R))

    # Create a process pool and distribute the work
    start_time = time.time()
    with Pool(processes=num_cores) as pool:
        results = list(tqdm(pool.imap(simulate_batch, args_list),
                           total=num_cores,
                           desc="Tien trinh mo phong Monte Carlo"))

    # Combine results from all processes
    total_counts = np.sum(results, axis=0)

    end_time = time.time()
    print(f"[SIMULATION] Mo phong hoan thanh trong {end_time - start_time:.2f} giay")

    return total_counts / simNumb

def calculate_theory_parallel(T_vals, lambda_, R, gamma, P0, alpha, beta, epsilon, rmax, stepr, djmax, stepdj, steprj):
    num_cores = cpu_count()
    print(f"[THEORY] Dang su dung {num_cores} CPU cores cho tinh toan ly thuyet")

    # Prepare arguments for each T value
    args_list = [(T, lambda_, R, gamma, P0, alpha, beta, epsilon, rmax, stepr, djmax, stepdj, steprj)
                for T in T_vals]

    # Create a process pool and distribute the work
    start_time = time.time()
    with Pool(processes=num_cores) as pool:
        P_values = list(tqdm(pool.imap(calculate_P_for_T, args_list),
                             total=len(T_vals),
                             desc="Tien trinh tinh toan ly thuyet"))

    end_time = time.time()
    print(f"[THEORY] Tinh toan ly thuyet hoan thanh trong {end_time - start_time:.2f} giay")

    return np.array(P_values)

def save_results_to_txt(gamma_dB, T_dB, P_values, simPCovkCCU):
    filename = f"simRe_gamma{gamma_dB}.txt"
    with open(filename, 'w') as f:
        f.write(f"gamma_dB = {gamma_dB};\n")
        f.write(f"T_dB = {np.array2string(T_dB, separator=', ')};\n")
        f.write(f"P_values = {np.array2string(P_values, separator=', ')};\n")
        f.write(f"simPCovkCCU = {np.array2string(simPCovkCCU, separator=', ')};\n")


def main():
    print(f"Tong so CPU cores co san: {cpu_count()}")

    # Cac tham so
    lambda_ = 0.01
    R = 3
    P0 = 1
    alpha = 0.5
    beta = 1
    epsilon = 0.1
    rmax = 250
    stepr = 0.25
    djmax = 250
    stepdj = 0.25
    steprj = 0.25
    simNumb = int(2e4)  
    diskRadius = 600

    T_dB = np.arange(-40, 0, 3)
    T_vals = 10**(T_dB / 10)
    gamma_dB_values = [-20, -10, 0, 20]
    colors = ['b', 'g', 'r', 'c', 'm']

    plt.figure()
    for i, gamma_dB in enumerate(gamma_dB_values):
        gamma = 10**(gamma_dB / 10)

        print(f"\n=== Dang xu ly cho gamma_dB = {gamma_dB} ===")

        print(f"Dang tinh toan gia tri ly thuyet cho gamma_dB = {gamma_dB}...")
        P_values = calculate_theory_parallel(T_vals, lambda_, R, gamma, P0, alpha, beta, epsilon, rmax, stepr, djmax, stepdj, steprj)

        print(f"Dang chay mo phong Monte Carlo song song cho gamma_dB = {gamma_dB}...")
        simPCovkCCU = monte_carlo_simulation_parallel(T_vals, lambda_, alpha, beta, gamma, epsilon, simNumb, diskRadius, R)

        save_results_to_txt(gamma_dB, T_dB, P_values, simPCovkCCU)

        plt.plot(T_dB, P_values, linestyle='-', color=colors[i], label=f"Ly thuyet SNR={gamma_dB} dB")
        plt.plot(T_dB, simPCovkCCU, marker='*', linestyle='', markersize=15, color=colors[i], label=f"Mo phong SNR={gamma_dB} dB")

    with open("simRe.txt", 'w') as f:
        for i, gamma_dB in enumerate(gamma_dB_values):
            filename = f"simRe_gamma{gamma_dB}.txt"
            if os.path.exists(filename):
                with open(filename, 'r') as gamma_file:
                    f.write(gamma_file.read())
                f.write("\n")

    print("Tat ca ket qua da duoc tong hop vao file simRe.txt")

    plt.xlabel("T_dB")
    plt.ylabel("Xac suat phu song")
    plt.grid()
    plt.legend()
    plt.savefig('coverage_probability.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()