import numpy as np
# -------------------------- 2D Derivative ---------------------------------
def derivative_2d_fhit(T, order, Tname):
    """
    Calculate spatial derivatives for 2D_FHIT
    Derivatives are calculated in spectral space
    Boundary conditions are periodic in x and y spatial dimensions
    Length of domain: 2*pi
    
    Args:
        T (ndarray): A 2D array of size M x N
        order (tuple): A tuple (orderX, orderY) specifying the order of the x and y derivatives, respectively
        Tname (str): A string representing the name of the input variable T
    
    Returns:
        ndarray: A 2D array containing the derivatives of T with respect to x and y
        str: A string representing the name of the output variable
    """
    orderX, orderY = order

    # Validating user input
    if orderX < 0 or orderY < 0:
        raise ValueError("Order of derivatives must be 0 or positive")
    elif orderX == 0 and orderY == 0:
        raise ValueError("Both order of derivatives are 0, at least one of them should be positive")

    Ngrid = T.shape
    Lx = 2 * np.pi  # Length of domain

    # Create wave numbers manually
    kx_pos = np.arange(0, Ngrid[0] / 2 + 1)
    kx_neg = np.arange(-Ngrid[0] / 2 + 1, 0)
    kx = np.concatenate((kx_pos, kx_neg)) * (2 * np.pi / Lx)
    
    # Making meshgrid
    Ky, Kx = np.meshgrid(kx, kx)
    
    ## This part needs to be checked 
    # Ngrid = T.shape
    # Lx = 2 * np.pi  # Length of domain
    
    # # Calculating the wave numbers
    # kx = 2 * np.pi * np.fft.fftfreq(Ngrid[0])
    # ky = 2 * np.pi * np.fft.fftfreq(Ngrid[1])

    # # Making meshgrid
    # Ky, Kx = np.meshgrid(ky, kx)



    # Calculating derivatives in spectral space
    T_hat = np.fft.fft2(T)
    Tdash_hat = ((1j * Kx) ** orderX) * ((1j * Ky) ** orderY) * T_hat
    Tdash = np.real(np.fft.ifft2(Tdash_hat))

    # Naming the variable
    TnameOut = Tname + ''.join(['x' * orderX, 'y' * orderY])

    return Tdash, TnameOut