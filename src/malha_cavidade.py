import numpy as np


def gerar_malha_cavidade(
    Nx=11, 
    Ny=11, 
    Lx=np.pi, 
    Ly=np.pi, 
    tipo_interior="alternado", 
    jitter_frac=0.0, 
    seed=42
):
    """
    Gera uma distribuição nodal regular ou quasi-regular para uma cavidade retangular [0, Lx] x [0, Ly]
    com imposição estrita de direcionalidade tangente nas fronteiras PEC.
    
    Regras de Direcionalidade de Fronteira (Paredes PEC):
    - Distribuição defasada (staggered / centrada em subintervalos): nós nos pontos médios (k + 0.5)*dx.
    - Nenhum nó é situado sobre os 4 vértices/cantos da fronteira.
    - Arestas horizontais (y=0 e y=Ly): vetor unitário t = [1, 0]^T (tangente ao eixo x).
    - Arestas verticais (x=0 e x=Lx): vetor unitário t = [0, 1]^T (tangente ao eixo y).
    
    Parâmetros:
    - Nx: Número de divisões nodais ao longo do eixo x (mínimo 3).
    - Ny: Número de divisões nodais ao longo do eixo y (mínimo 3).
    - Lx: Comprimento da cavidade na direção x (padrão: pi).
    - Ly: Comprimento da cavidade na direção y (padrão: pi).
    - tipo_interior: Tipo de orientação para os nós internos:
        * 'alternado': Alterna direções a 45° e 135° (ou eixos x e y).
        * 'diagonal': Direção fixa a 45° ([1/sqrt(2), 1/sqrt(2)]).
        * 'aleatorio': Ângulo theta uniforme em [0, 2*pi).
        * 'cruzado': Alterna entre [1,0], [0,1], [1,1]/sqrt(2), [-1,1]/sqrt(2).
    - jitter_frac: Fração máxima de deslocamento aleatório para nós internos (0.0 = perfeitamente regular).
    - seed: Semente para reprodutibilidade aleatória.
    
    Retorna:
    - coords: Array numpy (N, 2) com as coordenadas cartesianas (x, y) dos nós.
    - vectors: Array numpy (N, 2) com os vetores unitários de direção associados (tx, ty).
    - is_boundary: Array booleano (N,) indicando True para nós situados na fronteira PEC.
    """
    if Nx < 3 or Ny < 3:
        raise ValueError("Nx e Ny devem ser maiores ou iguais a 3.")
        
    np.random.seed(seed)
    
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    
    coords_list = []
    vectors_list = []
    is_boundary_list = []
    
    # 1. Nós interiores
    for j in range(1, Ny - 1):
        for i in range(1, Nx - 1):
            x = i * dx
            y = j * dy
            
            if jitter_frac > 0.0:
                jx = np.random.uniform(-jitter_frac * dx, jitter_frac * dx)
                jy = np.random.uniform(-jitter_frac * dy, jitter_frac * dy)
                pos = np.array([x + jx, y + jy], dtype=float)
            else:
                pos = np.array([x, y], dtype=float)
                
            if tipo_interior == "diagonal":
                vec = np.array([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)], dtype=float)
            elif tipo_interior == "alternado":
                if (i + j) % 2 == 0:
                    vec = np.array([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)], dtype=float)
                else:
                    vec = np.array([-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)], dtype=float)
            elif tipo_interior == "cruzado":
                idx_mod = (i + j) % 4
                if idx_mod == 0:
                    vec = np.array([1.0, 0.0], dtype=float)
                elif idx_mod == 1:
                    vec = np.array([0.0, 1.0], dtype=float)
                elif idx_mod == 2:
                    vec = np.array([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)], dtype=float)
                else:
                    vec = np.array([-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)], dtype=float)
            elif tipo_interior == "aleatorio":
                theta = np.random.uniform(0.0, 2.0 * np.pi)
                vec = np.array([np.cos(theta), np.sin(theta)], dtype=float)
            else:
                raise ValueError(f"tipo_interior desconhecido: '{tipo_interior}'")
                
            coords_list.append(pos)
            vectors_list.append(vec / np.linalg.norm(vec))
            is_boundary_list.append(False)
            
    # 2. Nós de fronteira defasados (staggered / centrados em subintervalos)
    # Nenhum nó é alocado sobre os 4 cantos/vértices do domínio.
    # Borda inferior (y = 0): tangente [1, 0]
    for k in range(Nx - 1):
        pos = np.array([(k + 0.5) * dx, 0.0], dtype=float)
        coords_list.append(pos)
        vectors_list.append(np.array([1.0, 0.0], dtype=float))
        is_boundary_list.append(True)
        
    # Borda superior (y = Ly): tangente [1, 0]
    for k in range(Nx - 1):
        pos = np.array([(k + 0.5) * dx, Ly], dtype=float)
        coords_list.append(pos)
        vectors_list.append(np.array([1.0, 0.0], dtype=float))
        is_boundary_list.append(True)
        
    # Borda esquerda (x = 0): tangente [0, 1]
    for k in range(Ny - 1):
        pos = np.array([0.0, (k + 0.5) * dy], dtype=float)
        coords_list.append(pos)
        vectors_list.append(np.array([0.0, 1.0], dtype=float))
        is_boundary_list.append(True)
        
    # Borda direita (x = Lx): tangente [0, 1]
    for k in range(Ny - 1):
        pos = np.array([Lx, (k + 0.5) * dy], dtype=float)
        coords_list.append(pos)
        vectors_list.append(np.array([0.0, 1.0], dtype=float))
        is_boundary_list.append(True)
        
    return np.array(coords_list), np.array(vectors_list), np.array(is_boundary_list, dtype=bool)
