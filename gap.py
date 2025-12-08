import numpy as np
import scipy.sparse as sparse
import scipy.linalg as la
import scipy.stats
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. PARÁMETROS AJUSTADOS (Para ver correlación)
# ==========================================
L = 11        
# AUMENTAMOS J para que los vecinos se "sientan" más fuerte
J = 1.5       # Antes 1.0. Al subirlo, aumentamos la correlación.
hx = 0.9      # Campo Transverso
hz = 0.4      # Campo Longitudinal
n_bins = 100  # Más bins para mayor resolución en la curva roja

# Nombre del archivo de salida
output_filename = f"ETH_CrossCorrelation_L{L}_J{J}.png"

print(f"Construyendo sistema L={L} con J={J} (Buscando correlaciones fuertes)...")

# ==========================================
# 2. CONSTRUCCIÓN (Igual que antes)
# ==========================================
sx = sparse.csr_matrix([[0, 1], [1, 0]])
sz = sparse.csr_matrix([[1, 0], [0, -1]])
id2 = sparse.identity(2)

def gen_many_body_op(op, site, length):
    ops_list = [id2] * length
    ops_list[site] = op
    full_op = ops_list[0]
    for i in range(1, length):
        full_op = sparse.kron(full_op, ops_list[i])
    return full_op

H = sparse.csr_matrix((2**L, 2**L))
for i in range(L - 1):
    op_zz = gen_many_body_op(sz, i, L) @ gen_many_body_op(sz, i+1, L)
    H -= J * op_zz
for i in range(L):
    H -= hx * gen_many_body_op(sx, i, L)
    H -= hz * gen_many_body_op(sz, i, L)

# ==========================================
# 3. DIAGONALIZACIÓN
# ==========================================
print("Diagonalizando...")
E, V = np.linalg.eigh(H.toarray())

# ==========================================
# 4. ELEMENTOS DE MATRIZ
# ==========================================
print("Calculando operadores...")
site_A = L // 2
site_B = L // 2 + 1 
Op1_matrix = gen_many_body_op(sz, site_A, L).toarray()
Op2_matrix = gen_many_body_op(sz, site_B, L).toarray()

# Rotación a base de energía
Op1_E = V.T @ Op1_matrix @ V
Op2_E = V.T @ Op2_matrix @ V

# ==========================================
# 5. EXTRACCIÓN DE DATOS (GAP)
# ==========================================
center_energy = np.mean(E)
window_width = (E.max() - E.min()) * 0.05
indices = np.where(np.abs(E - center_energy) < window_width)[0]

if len(indices) > 300: # Tomamos un poco más de muestra
    indices = np.random.choice(indices, 300, replace=False)
indices.sort()

omega_list = []
product_list = []

print("Procesando correlaciones cruzadas...")
for i in indices:
    for j in range(len(E)):
        if i == j: continue
        
        omega = E[j] - E[i]
        # Producto cruzado: <n|A|m><m|B|n>
        val = Op1_E[i, j] * Op2_E[j, i] 
        
        omega_list.append(omega)
        product_list.append(np.real(val))

# ==========================================
# 6. VISUALIZACIÓN Y GUARDADO (MODIFICADO ZOOM -1 a 1)
# ==========================================
omega_arr = np.array(omega_list)
prod_arr = np.array(product_list)

# Nombre del archivo para esta versión con zoom
output_filename_zoom = f"ETH_CrossCorrelation_L{L}_J{J}_Zoom.png"

def running_mean(x, y, bins=50):
    # Usamos statistic='mean' para obtener el valor promedio en cada bin
    bin_means, bin_edges, _ = scipy.stats.binned_statistic(x, y, statistic='mean', bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, bin_means

# --- CAMBIO 1: REDUCIR LA VENTANA DE DATOS ---
# Filtramos solo los datos entre -1 y 1 para que el promedio se calcule con
# alta resolución en esta zona específica.
limit = 1.0 
mask = (np.abs(omega_arr) < limit) 

omega_filt = omega_arr[mask]
prod_filt = prod_arr[mask]

# Calculamos el promedio móvil. 
# Al tener una ventana más pequeña (-1 a 1), con 80-100 bins tendremos 
# una resolución excelente para ver la forma de la "joroba".
bin_centers, bin_means = running_mean(omega_filt, prod_filt, bins=100)

plt.figure(figsize=(10, 6))

# Nube de puntos (Hacemos los puntos un poco más grandes para verlos mejor al hacer zoom)
plt.scatter(omega_filt, prod_filt, alpha=0.2, s=5, color='gray', label='Elementos individuales')

# Promedio Móvil (Línea roja)
plt.plot(bin_centers, bin_means, color='red', linewidth=3, label='Noise Kernel $K_{12}(\omega)$')

# Línea de referencia en cero
plt.axhline(0, color='black', linestyle='--', linewidth=1)

# --- CAMBIO 2: FIJAR LÍMITES DE EJES ---
plt.xlim(-limit, limit) 
# Opcional: Si quieres ver mejor la altura, descomenta la siguiente línea para limitar Y
# plt.ylim(-0.01, 0.01) 

plt.xlabel(r'$\omega = E_m - E_n$', fontsize=14)
plt.ylabel(r'$Re[\langle n|\hat{O}_1|m\rangle \langle m|\hat{O}_2|n\rangle]$', fontsize=14)
plt.title(f'Detalle de la Joroba ($L={L}, J={J}$)', fontsize=16)
plt.legend(fontsize=12, loc='upper right')
plt.grid(alpha=0.3)
plt.tight_layout()

# Guardar
plt.savefig(output_filename_zoom, dpi=300)
print(f"Gráfica con zoom guardada exitosamente como: {output_filename_zoom}")

plt.show()